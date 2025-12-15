import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_a = "./mono_data_bsize1/runtime_summary.csv"
csv_b = "./componentized_data_bsize1/runtime_summary.csv"

label_a = "Monolithic Deployment"
label_b = "Componentized Deployment"

df_a = pd.read_csv(csv_a).sort_values("qps_target")
df_b = pd.read_csv(csv_b).sort_values("qps_target")

merged = pd.merge(
    df_a[["qps_target", "p50_s", "p99_s"]],
    df_b[["qps_target", "p50_s", "p99_s"]],
    on="qps_target",
    suffixes=("_a", "_b"),
)

qps = merged["qps_target"].values
p50_a = merged["p50_s_a"].values
p99_a = merged["p99_s_a"].values

err_a = np.vstack([
    np.zeros_like(p50_a),
    p99_a - p50_a,
])

p50_b = merged["p50_s_b"].values
p99_b = merged["p99_s_b"].values

err_b = np.vstack([
    np.zeros_like(p50_b),
    p99_b - p50_b,
])

plt.figure(figsize=(10, 6))

plt.errorbar(
    qps,
    p50_a,
    yerr=err_a,
    fmt="-o",
    capsize=4,
    label=label_a,
    color="blue",
)

plt.errorbar(
    qps,
    p50_b,
    yerr=err_b,
    fmt="-o",
    capsize=4,
    label=label_b,
    color="green",
)

plt.xlabel("Send Rate (qps)")
plt.ylabel("Latency (seconds)")
plt.title("Latency vs Send Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.figtext(
    0.5,
    -0.08,
    "Line denotes median latency; tick above denotes 99th percentile.",
    ha="center",
    fontsize=10,
)

output_path = "latency_vs_send_rate.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to {output_path}")
