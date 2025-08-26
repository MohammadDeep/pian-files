# -*- coding: utf-8 -*-
"""
Pain-SSL Pipeline (PyTorch) — Full Script
=========================================
هدف: پیش‌آموزشِ خودنظارتی روی سیگنال‌های فیزیولوژیکِ چندکاناله با دو روش
(۳.۱) Contrastive (سبک SimCLR) و (۳.۲) Masked Reconstruction (سبک MAE)،
و سپس استخراج embedding برای آموزش یک طبقه‌بند خطی کوچک با LOOCV.

روند اجرا (TL;DR):
1) سیگنال‌های هر رکورد را به شکل [C,T] (کانال × زمان) آماده و نرمال‌سازی کنید.
2) با windowing آن‌ها را به [N,C,win] تبدیل کنید.
3) یکی از دو پیش‌آموزش را اجرا کنید:
   - train_contrastive(...): یادگیری فضای شباهت
   - train_masked_reconstruction(...): یادگیری ساختار سیگنال با بازسازی ناحیه ماسک‌شده
4) از encoderِ آموزش‌دیده embedding بگیرید و با LOOCV یک سرِ خطی کوچک (Logistic/SVM) را آموزش/ارزیابی کنید.

نکات مهم:
- Augmentationها ملایم و سازگار با فیزیولوژی‌اند (نویز کم، اسکیل دامنه، شیفت زمانی، channel-dropout).
- برای ECG از warping شدید اجتناب کنید؛ زمان‌بندی P-QRS-T اهمیت پزشکی دارد.
- حتماً split را روی «رکوردها» بزنید (نه پنجره‌ها) تا leakage رخ ندهد.
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, f1_score, recall_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# -------------------------
# 0) Device & Seeds
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(123)
np.random.seed(123)


# -------------------------
# 1) Augmentations (label‑preserving)
# -------------------------
def jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """نویز گاوسی کوچک به ازای هر کانال: x شکل [C,T]."""
    return x + torch.randn_like(x) * sigma

def scale_amplitude(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    """اسکیل دامنهٔ تصادفیِ ملایم برای هر کانال."""
    C, _ = x.shape
    factors = torch.empty(C, device=x.device).uniform_(low, high).view(C, 1)
    return x * factors

def time_shift(x: torch.Tensor, max_shift: int = 12) -> torch.Tensor:
    """جابجایی زمانیِ دَوَرانی در بازهٔ ±max_shift نمونه."""
    if max_shift <= 0:
        return x
    shift = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)

def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """گاهی یک کانال را صفر می‌کند تا مدل به همهٔ کانال‌ها متکی باشد."""
    if p <= 0.0:
        return x
    C, _ = x.shape
    mask = (torch.rand(C, device=x.device) > p).float().view(C, 1)
    return x * mask

def light_augment(x: torch.Tensor) -> torch.Tensor:
    """ترکیب اگمنت‌های امن برای سیگنال‌های فیزیولوژیک."""
    x = jitter(x, sigma=0.02)
    x = scale_amplitude(x, 0.9, 1.1)
    x = time_shift(x, max_shift=12)
    x = channel_dropout(x, p=0.1)
    return x


# -------------------------
# 2) Windowing utils
# -------------------------
def windowize(signal: np.ndarray, win: int, step: int) -> np.ndarray:
    """یک سیگنال [C,T] را به پنجره‌های [N,C,win] تبدیل می‌کند؛ اگر کوتاه بود padding می‌کند."""
    C, T = signal.shape
    windows = []
    idx = 0
    while idx + win <= T:
        windows.append(signal[:, idx:idx+win])
        idx += step
    if not windows:  # کوتاه‌تر از win
        pad = np.zeros((C, win), dtype=signal.dtype)
        pad[:, :T] = signal
        windows.append(pad)
    return np.stack(windows, axis=0)

def build_windows(signals: List[np.ndarray], win: int, overlap: float = 0.5) -> np.ndarray:
    """لیست سیگنال‌های [C,T] را به یک آرایهٔ پنجره‌ها [N,C,win] تبدیل می‌کند."""
    step = max(1, int(win * (1 - overlap)))
    all_w = [windowize(s, win, step) for s in signals]
    return np.concatenate(all_w, axis=0)

def make_win_to_record_idx(signals: List[np.ndarray], win: int, overlap: float = 0.5) -> np.ndarray:
    """برای هر پنجره مشخص می‌کند متعلق به کدام رکورد است (برای تجمیع/LOOCV)."""
    step = max(1, int(win * (1 - overlap)))
    idxs = []
    for ridx, s in enumerate(signals):
        _, T = s.shape
        n, i = 0, 0
        while i + win <= T:
            n += 1; i += step
        if n == 0:
            n = 1
        idxs.extend([ridx]*n)
    return np.array(idxs, dtype=np.int64)


# -------------------------
# 3) Datasets
# -------------------------
class ContrastiveWindows(Dataset):
    """برای SimCLR: از همان پنجره دو نمای اگمنتی می‌سازد."""
    def __init__(self, windows: np.ndarray):
        self.windows = windows.astype(np.float32)
    def __len__(self) -> int: return self.windows.shape[0]
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])  # [C,T]
        return light_augment(x), light_augment(x)

class MaskedWindows(Dataset):
    """برای MAE: (ورودی ماسک‌شده، هدف بازسازی، ماسک) را برمی‌گرداند."""
    def __init__(self, windows: np.ndarray, mask_ratio: float = 0.25, span: int = 16):
        self.windows = windows.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.span = span
    def __len__(self) -> int: return self.windows.shape[0]
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])  # [C,T]
        target = x.clone()
        C, T = x.shape
        total_mask = int(T * self.mask_ratio)
        mask = torch.zeros(T, dtype=torch.bool)
        masked = 0
        while masked < total_mask:
            start = np.random.randint(0, max(1, T - self.span + 1))
            end = min(T, start + self.span)
            mask[start:end] = True
            masked = int(mask.sum().item())
        x_masked = x.clone()
        x_masked[:, mask] = 0.0
        return x_masked, target, mask.view(1, -1)  # [1,T]


# -------------------------
# 4) Models
# -------------------------
class SmallEncoder1D(nn.Module):
    """1D-CNN سبک با GlobalAvgPool → embedding 128بُعدی."""
    def __init__(self, in_ch: int, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),  nn.BatchNorm1d(64), nn.GELU(), nn.AvgPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, emb_dim))
    def forward(self, x):  # x: [B,C,T]
        return self.head(self.features(x))  # [B,128]

class ProjectionHead(nn.Module):
    """MLP دو لایه برای Contrastive."""
    def __init__(self, in_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim)
        )
    def forward(self, x): return self.mlp(x)

class SmallDecoder1D(nn.Module):
    """دیکودر بسیار ساده برای بازسازی؛ می‌توانید بعداً به ConvTranspose ارتقا دهید."""
    def __init__(self, out_ch: int, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv1d(out_ch, out_ch, kernel_size=1)
    def forward(self, z: torch.Tensor, T: int, C: int) -> torch.Tensor:
        x = self.fc(z).unsqueeze(-1).repeat(1, 1, T)  # [B,512,T]
        x = x[:, :C, :]
        return self.head(x)                           # [B,C,T]


# -------------------------
# 5) Losses
# -------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """NT-Xent با شباهت کسینوسی و دمای tau (سبک SimCLR)."""
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    Z = torch.cat([z1, z2], dim=0)             # [2B,D]
    S = torch.matmul(Z, Z.T) / tau             # [2B,2B]
    S = S.masked_fill(torch.eye(2*B, dtype=torch.bool, device=Z.device), -1e9)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(Z.device)
    return F.cross_entropy(S, targets)

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE فقط روی نقاط ماسک‌شده (mask: [B,1,T])."""
    diff = (pred - target) ** 2
    diff = diff * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


# -------------------------
# 6) Trainers
# -------------------------
def train_contrastive(
    windows: np.ndarray, in_ch: int, proj_dim: int = 64,
    batch_size: int = 128, epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
    tau: float = 0.1, device: str = DEVICE
) -> Tuple[nn.Module, nn.Module]:
    ds = ContrastiveWindows(windows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    encoder = SmallEncoder1D(in_ch=in_ch, emb_dim=128).to(device)
    head = ProjectionHead(128, proj_dim).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    encoder.train(); head.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = head(encoder(v1)), head(encoder(v2))
            loss = nt_xent_loss(z1, z2, tau)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * v1.size(0)
        print(f"[Contrastive] Epoch {ep:03d}/{epochs} | Loss: {total/len(ds):.4f}")
    return encoder, head

def train_masked_reconstruction(
    windows: np.ndarray, in_ch: int, mask_ratio: float = 0.25, span: int = 16,
    batch_size: int = 128, epochs: int = 120, lr: float = 1e-3, weight_decay: float = 1e-4,
    device: str = DEVICE
) -> Tuple[nn.Module, nn.Module]:
    ds = MaskedWindows(windows, mask_ratio=mask_ratio, span=span)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    encoder = SmallEncoder1D(in_ch=in_ch, emb_dim=128).to(device)
    decoder = SmallDecoder1D(out_ch=in_ch, emb_dim=128).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=weight_decay)
    encoder.train(); decoder.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for x_masked, target, mask in dl:
            x_masked, target, mask = x_masked.to(device), target.to(device), mask.to(device)
            z = encoder(x_masked)
            pred = decoder(z, T=target.shape[-1], C=target.shape[1])
            loss = masked_mse(pred, target, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x_masked.size(0)
        print(f"[Masked]       Epoch {ep:03d}/{epochs} | Loss: {total/len(ds):.6f}")
    return encoder, decoder


# -------------------------
# 7) Embedding extraction
# -------------------------
def extract_embeddings(encoder: nn.Module, windows: np.ndarray, batch_size: int = 256, device: str = DEVICE) -> np.ndarray:
    dl = DataLoader(torch.from_numpy(windows.astype(np.float32)), batch_size=batch_size, shuffle=False)
    encoder.eval()
    Z = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            z = encoder(x).cpu().numpy()
            Z.append(z)
    return np.vstack(Z)  # [N,128]


# -------------------------
# 8) Linear eval (LOOCV scaffold)
# -------------------------
def loocv_linear_eval(Z: np.ndarray, win_to_record_idx: np.ndarray, record_labels: np.ndarray) -> Dict[str, float]:
    """
    LOOCV روی «رکوردها» با طبقه‌بند خطی (Logistic Regression).
    Z: embedding پنجره‌ها [N,128]
    win_to_record_idx: نگاشت پنجره→رکورد [N]
    record_labels: برچسب رکوردها [R] (۰/۱)
    """
    R = len(record_labels)
    y_true_records = record_labels.astype(int)
    y_pred_records = []
    scaler = StandardScaler()

    for test_r in range(R):
        train_r = [r for r in range(R) if r != test_r]
        train_mask = np.isin(win_to_record_idx, train_r)
        test_mask  = (win_to_record_idx == test_r)

        Z_tr, Z_te = Z[train_mask], Z[test_mask]
        y_tr_win = record_labels[win_to_record_idx[train_mask]]  # برچسب پنجره = برچسب رکورد

        scaler.fit(Z_tr)
        Z_tr = scaler.transform(Z_tr)
        Z_te = scaler.transform(Z_te)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(Z_tr, y_tr_win)

        # تجمیع پیش‌بینی پنجره‌ها به سطح رکورد
        p = clf.predict_proba(Z_te)[:, 1].mean()
        yhat_r = int(p >= 0.5)
        y_pred_records.append(yhat_r)

    acc  = accuracy_score(y_true_records, y_pred_records)
    f1   = f1_score(y_true_records, y_pred_records)
    sens = recall_score(y_true_records, y_pred_records)                    # حساسیت
    spec = recall_score(1 - y_true_records, 1 - np.array(y_pred_records))  # ویژگی
    return {"acc": acc, "f1": f1, "sens": sens, "spec": spec} # type: ignore


# -------------------------
# 9) Demo (__main__): جایگزین با دادهٔ واقعی شما
# -------------------------
if __name__ == "__main__":
    # --- جای شما: سیگنال‌های واقعی [C,T] بعد از فیلتر/نرمال‌سازی ---
    num_records = 10
    C = 6
    T = 4096
    rng = np.random.default_rng(0)
    signals = [rng.normal(size=(C, T)).astype(np.float32) for _ in range(num_records)]

    # Windowing
    win = 512         # مثلاً 8s×fs اگر fs=64Hz
    overlap = 0.5
    windows = build_windows(signals, win=win, overlap=overlap)  # [N,C,win]
    win_to_record_idx = make_win_to_record_idx(signals, win=win, overlap=overlap)
    print("Windows:", windows.shape)

    # --- 3.1 Contrastive (دموی کوتاه: epochs=2؛ در عمل 100–200) ---
    print("\n=== Contrastive pretraining (demo) ===")
    enc_c, head = train_contrastive(windows, in_ch=C, proj_dim=64, batch_size=64, epochs=2, lr=1e-3, weight_decay=1e-4)

    # استخراج embedding و ارزیابی خطی (برچسب‌های نمایشی)
    Z_c = extract_embeddings(enc_c, windows, batch_size=256)
    record_labels = np.array([0,1,0,1,0,1,0,1,0,1], dtype=int)  # جای شما
    metrics_c = loocv_linear_eval(Z_c, win_to_record_idx, record_labels)
    print("Contrastive LOOCV (demo labels):", metrics_c)

    # --- 3.2 Masked Reconstruction (دموی کوتاه: epochs=2؛ در عمل 100–150) ---
    print("\n=== Masked reconstruction pretraining (demo) ===")
    enc_m, dec = train_masked_reconstruction(windows, in_ch=C, mask_ratio=0.25, span=24, batch_size=64, epochs=2, lr=1e-3, weight_decay=1e-4)

    Z_m = extract_embeddings(enc_m, windows, batch_size=256)
    metrics_m = loocv_linear_eval(Z_m, win_to_record_idx, record_labels)
    print("Masked LOOCV (demo labels):", metrics_m)

    # ذخیرهٔ آرتیفکت‌ها
    os.makedirs("./artifacts", exist_ok=True)
    torch.save(enc_c.state_dict(), "./artifacts/encoder_contrastive.pt")
    torch.save(enc_m.state_dict(), "./artifacts/encoder_masked.pt")
    np.save("./artifacts/embeddings_contrastive.npy", Z_c)
    np.save("./artifacts/embeddings_masked.npy", Z_m)
    print("\nArtifacts saved in ./artifacts")
