"""Regenerate results/robustness.png from results/results_by_attack.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/results_by_attack.csv")
x = np.arange(len(df))
fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})

w = 0.27
a.bar(x - w, df["TPR_at_0_5"], w, label="Detection rate at default threshold (0.5)")
a.bar(x, df["AUC"], w, label="AUC (95% bootstrap CI)",
      yerr=[df["AUC"] - df["AUC_lo"], df["AUC_hi"] - df["AUC"]], capsize=2)
a.bar(x + w, df["TPR_at_1pct_FPR"], w, label="TPR at ≤1% FPR (threshold chosen post hoc)")
a.set_ylim(0, 1.32); a.set_yticks(np.arange(0, 1.01, 0.25))
a.set_ylabel("Detection"); a.legend(loc="upper center", ncol=1, fontsize=8.5, frameon=False)
a.set_title("AudioSeal under processing: detection (n = 60 watermarked + 60 clean clips)", fontsize=11)

b.bar(x, df["bit_acc"], 0.55, color="#2ca02c")
b.axhline(0.5, ls=":", c="grey", lw=1); b.text(len(df) - 0.5, 0.51, "chance (0.5)", ha="right", va="bottom", fontsize=8, color="grey")
b.set_ylim(0, 1.05); b.set_ylabel("Message bit accuracy")
b.set_title("Message recovery (watermarked clips only)", fontsize=11)
b.set_xticks(x); b.set_xticklabels(df["attack"], rotation=40, ha="right")
plt.tight_layout(); plt.savefig("results/robustness.png", dpi=200)
