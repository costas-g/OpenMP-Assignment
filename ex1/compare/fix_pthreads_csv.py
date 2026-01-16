import pandas as pd

# Load CSV
df = pd.read_csv("poly_results.csv")

# Group by degree and workers, compute means
df_summary = df.groupby(["degree", "workers"], as_index=False).agg(
    gen_mean=("gen_time", "mean"),
    serial_mean=("serial_time", "mean"),
    parallel_mean=("parallel_time", "mean")
)

# Compute speedup from mean times
df_summary["speedup"] = df_summary["serial_mean"] / df_summary["parallel_mean"]

# Save to CSV
df_summary.to_csv("pth_poly_results_stats.csv", index=False)
