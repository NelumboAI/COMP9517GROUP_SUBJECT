import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

analysis_dir = Path("results/error_analysis")
best_exp = "sift_vocab200_sgd_svm_c1"

df = pd.read_csv(analysis_dir / f"{best_exp}_top_confused_pairs.csv")
top = df.head(10).copy()

top["pair"] = top["true_label_name"] + " → " + top["pred_label_name"]

plt.figure(figsize=(12, 6))
plt.barh(top["pair"][::-1], top["count"][::-1])
plt.xlabel("Number of misclassifications")
plt.ylabel("True class → Predicted class")
plt.title("Top 10 Most Confused Species Pairs")
plt.tight_layout()
plt.savefig(analysis_dir / f"{best_exp}_top_confused_pairs_bar.png", dpi=300)
plt.close()

print("Saved top confused pairs bar chart.")