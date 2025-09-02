import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm


def prepare_single_csv(file_path: Path, output_dir: Path) -> Path | None:
    """Convert raw CSV into epoch-based CSV and save to output_dir. Return new file path."""
    try:
        df = pd.read_csv(file_path)
        if 'number/ len_data' not in df.columns:
            return None

        list_data = [i.split('/') for i in df['number/ len_data']]
        list_n = [int(j[0]) / int(j[1]) for j in list_data]
        n_list = [0]
        for number in list_n:
            n_list.append(number)
            if number == 1:
                break

        df_n_list = np.array(n_list[1:]) - np.array(n_list[:-1])
        df.drop('number/ len_data', axis=1, inplace=True)

        for i in range(len(df)):
            df.iloc[i] = df.iloc[i] * df_n_list[(i + 1) % len(df_n_list)]

        if len(df) % len(df_n_list) != 0:
            number = int((len(df) // len(df_n_list)) * len(df_n_list))
            df = df[:number]

        df_new = df.groupby(df.index // len(df_n_list)).sum()
        df_new["epoches"] = range(1, len(df_new) + 1)
        cols = ["epoches"] + [col for col in df_new.columns if col != "epoches"]
        df_new = df_new[cols]

        output_dir.mkdir(exist_ok=True, parents=True)
        out_path = output_dir / file_path.name
        df_new.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return None


def plot_single_csv(csv_path: Path, output_dir: Path) -> None:
    """Create and save train/test loss and accuracy plots for one processed CSV file."""
    df = pd.read_csv(csv_path)
    if 'epoches' not in df.columns:
        return

    x = df['epoches']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, df['train_loss'], label='Train Loss')
    ax1.plot(x, df['test_loss'], label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, df['train_acc'], label='Train Accuracy')
    ax2.plot(x, df['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f'Model Training - {csv_path.stem}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = output_dir / f"{csv_path.stem}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def collect_metrics_for_summary(csv_path: Path) -> dict | None:
    """Extract summary metrics from a processed CSV."""
    try:
        df = pd.read_csv(csv_path)
        if 'epoches' not in df.columns:
            return None
        return {
            'Model': csv_path.stem,
            'Max Train Acc': round(df['train_acc'].max(), 4),
            'Max Test Acc': round(df['test_acc'].max(), 4),
            'Min Train Loss': round(df['train_loss'].min(), 4),
            'Min Test Loss': round(df['test_loss'].min(), 4)
        }
    except Exception as e:
        print(f"⚠️ Failed to summarize {csv_path.name}: {e}")
        return None


def process_all_csvs(input_dir: Path, output_dir: Path):
    """
    Main controller: processes all CSVs in input_dir and writes results to output_dir.
    Includes:
      - preprocessing
      - plotting
      - summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_paths = []

    for csv_file in tqdm(input_dir.glob("*.csv"), desc="Processing all CSVs"):
        processed_csv = prepare_single_csv(csv_file, output_dir)
        if processed_csv:
            plot_single_csv(processed_csv, output_dir)
            processed_paths.append(processed_csv)

    # Generate summary
    summaries = []
    for p in processed_paths:
        result = collect_metrics_for_summary(p)
        if result:
            summaries.append(result)

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary = df_summary.sort_values(
            by=['Max Test Acc', 'Min Test Loss'],
            ascending=[False, True]
        ).reset_index(drop=True)
        df_summary.to_csv(output_dir / "summary.csv", index=False)
        print("\n✅ Summary saved to:", output_dir / "summary.csv")
        print(df_summary)
    else:
        print("⚠️ No valid summaries generated.")

def plot_comparison_across_models(output_dir: Path):
    """
    Plot line charts comparing each metric (acc/loss) across all models.
    """
    metrics = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    model_data = {}
    csv_paths = list(output_dir.glob("*.csv"))
    # جمع‌آوری داده‌ها
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if 'epoches' not in df.columns:
                continue
            model_data[path.stem] = (df['epoches'], df)
        except Exception as e:
            print(f"⚠️ Failed to read {path.name}: {e}")

    # رسم نمودار برای هر متریک
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model_name, (x, df) in model_data.items():
            if metric in df.columns:
                plt.plot(x, df[metric], label=model_name)

        plt.title(f"Comparison of {metric} across models")
        plt.xlabel("Epochs")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = output_dir / f"1_comparison_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    from vision.Config import dir_history_model  # مسیر پوشه ورودی
    input_dir = Path(dir_history_model)
    output_dir = input_dir / "processed"

    process_all_csvs(input_dir, output_dir)
        # رسم نمودار مقایسه‌ای بین مدل‌ها
    plot_comparison_across_models(output_dir)
