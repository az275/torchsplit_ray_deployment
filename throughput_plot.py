import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_a = "./mono_data_bsize1/runtime_summary.csv"
csv_b = "./componentized_data_bsize1/runtime_summary.csv"

label_a = "Monolithic Deployment"
label_b = "Componentized Deployment"

df_a = pd.read_csv(csv_a)
df_b = pd.read_csv(csv_b)

merged = pd.merge(
    df_a[["qps_target", "achieved_qps"]],
    df_b[["qps_target", "achieved_qps"]],
    on="qps_target",
    suffixes=("_a", "_b"),
)

qps_targets = merged["qps_target"].values
achieved_a = merged["achieved_qps_a"].values
achieved_b = merged["achieved_qps_b"].values

x = np.arange(len(qps_targets))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    achieved_a,
    width,
    label=label_a,
    color="blue",
)

plt.bar(
    x + width / 2,
    achieved_b,
    width,
    label=label_b,
    color="green",
)

plt.xlabel("Send Rate (qps)")
plt.ylabel("Throughput (qps)")
plt.title("Throughput vs Send Rate")
plt.xticks(x, qps_targets)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

output_path = "throughput_vs_send_rate.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to {output_path}")
