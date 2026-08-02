import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_dir / "ablation_results.csv")

df_sorted = df.sort_values("test_f1_macro", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_sorted["experiment_name"], df_sorted["test_f1_macro"])
plt.xlabel("Test Macro-F1")
plt.ylabel("Experiment")
plt.title("Traditional Method Ablation: Test Macro-F1")
plt.tight_layout()
plt.savefig(plots_dir / "macro_f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(df_sorted["experiment_name"], df_sorted["test_accuracy"])
plt.xlabel("Test Accuracy")
plt.ylabel("Experiment")
plt.title("Traditional Method Ablation: Test Accuracy")
plt.tight_layout()
plt.savefig(plots_dir / "accuracy_comparison.png", dpi=300)
plt.close()

df_top5 = df.dropna(subset=["test_top5_accuracy"])
df_top5 = df_top5.sort_values("test_top5_accuracy", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_top5["experiment_name"], df_top5["test_top5_accuracy"])
plt.xlabel("Test Top-5 Accuracy")
plt.ylabel("Experiment")
plt.title("Traditional Method Ablation: Test Top-5 Accuracy")
plt.tight_layout()
plt.savefig(plots_dir / "top5_accuracy_comparison.png", dpi=300)
plt.close()

print("Plots saved to results/plots/")