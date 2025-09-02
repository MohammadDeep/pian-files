import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import os
from torchmetrics.classification import F1Score
import re

from vision.Config import dir_history_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device  : {device}')



def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader_test :torch.utils.data.DataLoader,
               results = None,
               number_ep = 1000,
               use_sigmoid = True):
    if results is None:
      # 2. Create empty results dictionary
      results = {
          'number/ len_data': [],
          "train_loss": [],
          "train_acc": [],
          "test_loss": [],
          "test_acc": []
      }
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loop through data loader data batches
    number_data ,number_data1= 0 ,0
    number_all_data = len(dataloader)
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        # Send data to target device
        #y = y.float()

        X, y = X.to(device).float(), y.to(device).float()

        y = y.float()
        #y = y.unsqueeze(1)
        model.to(device)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        if use_sigmoid:
            y_perd_1 = torch.sigmoid(y_pred)
        else:
            y_perd_1 = y_pred
        y_pred_class = torch.where(y_perd_1 >= 0.5, torch.tensor(1,device=device), torch.tensor(0,device=device))
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        number_data = number_data + 1
        number_data1 = number_data1 + 1

        if number_data % number_ep == 0 or number_data == number_all_data :
          number_len = f'{number_data} / {number_all_data}'


          train_loss = train_loss / number_data1
          train_acc = train_acc / number_data1
          number_data1 = 0

          test_loss, test_acc = test_step(model=model,
            dataloader=dataloader_test,
            loss_fn=loss_fn,
            use_sigmoid = use_sigmoid)
          print(
            f"number_len : {number_len}| "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
          # 5. Update results dictionary
          # Ensure all data is moved to CPU and converted to float for storage
          results['number/ len_data'].append(number_len)
          results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
          results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
          results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
          results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
          train_loss, train_acc = 0, 0


    return results





def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              show_plot_and_F1 = False,
              threshold = 0.5,
              use_sigmoid = True):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_targets = []
    # Turn on inference context manager
    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, (X, y) in tqdm(enumerate(dataloader)):

            # Send data to target device
            #y = y.float()
            X, y = X.to(device).float(), y.to(device).float()

            y = y.float()
            #y = y.unsqueeze(1)
            model.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)
            
            # 2. Calculate and accumulate loss
            
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            #test_pred_labels = test_pred_logits.argmax(dim=1)
            if use_sigmoid : 
                #print('use sigmoid funciont on output model')
                # add sigmoid funtion in output
                test_pred_logits = torch.sigmoid(test_pred_logits)

            test_pred_labels = torch.where(test_pred_logits >= threshold, torch.tensor(1,device=device), torch.tensor(0,device=device))
            all_preds.append(test_pred_labels)
            all_targets.append(y)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    if show_plot_and_F1 :
      y_pred_tensor = torch.cat(all_preds)     # شکل: (N_total,)
      y_true_tensor = torch.cat(all_targets)   # شکل: (N_total,)
      class_names = ['note persion ', 'persion']
      # 4. تعریف و محاسبه‌ی ماتریس سردرگمی
      confmat = ConfusionMatrix(num_classes=len(class_names),
                                task='multiclass').to(device)
      # torchmetrics برای محاسبه‌ی تجمعی، می‌توانیم مستقیم ورودی بزرگ بدهیم:
      confmat_tensor = confmat(y_pred_tensor.to(device),
                              y_true_tensor.to(device))
      # حال confmat_tensor یک Tensor از شکل (C, C) است
      # تعریف و محاسبه‌ی F1-Score
      f1 = F1Score(num_classes=2, task='binary').to(device)
      f1_score = f1(y_pred_tensor.to(device), y_true_tensor.to(device)).item()
      print(f"F1-Score on Test Set: {f1_score:.4f}")

      # 5. رسم ماتریس سردرگمی با mlxtend

      fig, ax = plot_confusion_matrix(
          conf_mat=confmat_tensor.cpu().numpy(),  # matplotlib با NumPy کار می‌کند
          class_names=class_names,                # برچسب‌های سطر و ستون
          figsize=(10, 7),                        # اندازه شکل
          show_normed=False                       # اگر True باشد اعداد را نرمال می‌کند
      )
      plt.title("Confusion Matrix on Test Set")
      plt.show()
          # Adjust metrics to get average loss and accuracy per batch
      test_loss = test_loss / len(dataloader)
      test_acc = test_acc / len(dataloader)
      return test_loss, test_acc, f1_score ,confmat_tensor
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc




# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          model_name:str = None,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          results = None,
          epochs: int = 5,
          number_ep = 1000,
          use_sigmoid = True,
          dir_history_model = dir_history_model,
          latest_epoch = -1,
          strict = True):
    if model_name == None:
        model_name = type(model).__name__ 
    
    checkpoint_dir = os.path.join(dir_history_model, model_name)
    latest_checkpoint_path = None
    print(f'checkpoint for model and history {checkpoint_dir}')
    if os.path.isdir(checkpoint_dir):
        # پیدا کردن جدیدترین checkpoint بر اساس شماره epoch
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        if latest_epoch == -1:
            for f in checkpoint_files:
                # استخراج شماره epoch از نام فایل با استفاده از regex
                match = re.search(r'model_epoch_(\d+).pth', f)
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint_path = os.path.join(checkpoint_dir, f)
        else:
            latest_checkpoint_path = f"{dir_history_model}/{model_name}/model_epoch_{latest_epoch}.pth"
    if latest_checkpoint_path:
        try:
            print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)
            
            model.load_state_dict(checkpoint['model_state_dict'], strict = strict)
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch شروع را برابر با epoch ذخیره شده + 1 قرار می‌دهیم
            latest_epoch = checkpoint['epoch'] 
            tlos , tac = test_step(model,
                                    test_dataloader,
                                    loss_fn)
            print('-' * 50)
            print(f'Model load test loss : { tlos}  ,  test acc : { tac}')
            print(f"Model and optimizer state loaded. Resuming from epoch {latest_epoch}")
        except: 
            print(f'cant read model : {latest_checkpoint_path}')
        print('-' * 50)
        print('try to read history...')
        try : 
             df_loaded = pd.read_csv(f"{dir_history_model}/{model_name}/history_model_epoch_{latest_epoch}.csv")
             print(f'read history file : {dir_history_model}/{model_name}/history_model_epoch_{latest_epoch}.csv')
             # تبدیل به دیکشنری از لیست‌ها
             results= df_loaded.to_dict(orient='list')
        except:
            print(f' cant read history ')
        print('-' * 50)
    else:
        print("No checkpoint found. Starting training from scratch.")
        # اگر checkpoint وجود نداشت، start_epochs همان مقدار ورودی (معمولا 0) باقی می‌ماند
    





    print('Start taining...')
    for epoch in range( latest_epoch + 1,latest_epoch + epochs + 1):
        print(f'epoch : {epoch}')
        results = train_step(model=model,
                              dataloader=train_dataloader,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              dataloader_test = test_dataloader,
                              results = results,
                              number_ep = number_ep,
                              use_sigmoid = use_sigmoid
                             )
        # ذخیره Checkpoint
        
        checkpoint_path = f"{dir_history_model}/{model_name}/model_epoch_{epoch}.pth"
        # اگر پوشه‌ی پدر وجود نداشت، بسازش
        print(f'save modle : {checkpoint_path}')
        modelse_save_dir = os.path.join(dir_history_model , model_name)
        os.makedirs(modelse_save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f'save hsietory :  {dir_history_model}/{model_name}/history_model_epoch_{epoch}.csv')
        df = pd.DataFrame(results)
        df.to_csv( f"{dir_history_model}/{model_name}/history_model_epoch_{epoch}.csv", index=False, encoding='utf-8-sig')




    # 6. Return the filled results at the end of the epochs
    return results




# Save and lod models
def save_models_historyes(
    model,
    history,
    dir_save,
    name_file,
    ):
  file_dir = f'{dir_save}/{name_file}'
  # Save the model's state_dict to a file in Google Drive
  torch.save(model.state_dict(), f'{file_dir}.pth')
  print(f"Model saved :{file_dir}.pth")


  # Create dataframe
  df = pd.DataFrame(history)
  # Save dataFrame
  df.to_csv(f'{file_dir}.csv', index=False, encoding='utf-8-sig')
  print(f'History saved : {file_dir}.csv')



def load_model(model_class, checkpoint_path, device=None):
    """
    بارگذاری یک مدل PyTorch از فایل checkpoint.

    Args:
        model_class: کلاس مدل شما (مثلاً MyModel).
        checkpoint_path: مسیر فایل .pth یا .pt حاوی state_dict یا دیکشنری checkpoint.
        device (optional): torch.device یا رشته 'cpu'/'cuda'. اگر None باشد،
                             خودکار CUDA را در صورت امکان استفاده می‌کند.

    Returns:
        مدل در وضعیت eval و منتقل‌شده به device مشخص‌شده.
    """
    # تعیین دستگاه
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # نمونه‌سازی مدل
    model = model_class()

    # بارگذاری checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # اگر checkpoint شامل کل state_dict نیست
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
