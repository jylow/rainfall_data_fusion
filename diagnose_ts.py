"""Diagnostic script — run with: conda run -n fyp python3 diagnose_ts.py"""
import pandas as pd
import numpy as np

from src.utils import read_config
from src.raingauge.utils import load_raingauge_dataset
from src.radar.utils import load_processed_dataset
from src.cml.utils import load_cml_dataset

config = read_config("config.yaml")
uptime_threshold = config['filters']['uptime_threshold']
start_year = config['dataset_parameters']['start_year']
end_year   = config['dataset_parameters']['end_year']

raingauge_df, _ = load_raingauge_dataset(start=start_year, end=end_year, uptime_threshold=uptime_threshold)
radar_df = load_processed_dataset("database/processed_radar_dataset.pkl")
cml_df, _ = load_cml_dataset(config['dataset_parameters']['cml_folder'])
cml_df = cml_df.fillna(0)

print(f"raingauge timestamps : {raingauge_df['timestamp'].nunique()}")
print(f"radar timestamps     : {radar_df['timestamp'].nunique()}")
print(f"cml timestamps       : {cml_df['timestamp'].nunique()}")

merged = radar_df.merge(raingauge_df, on=['timestamp'], how='inner')
print(f"After radar x raingauge inner join: {merged['timestamp'].nunique()} timestamps")

merged2 = merged.merge(cml_df, on=['timestamp'], how='inner')
print(f"After + cml inner join:             {merged2['timestamp'].nunique()} timestamps")

# ---- Gap analysis ----
ts = pd.to_datetime(merged2['timestamp'].drop_duplicates().sort_values().values)
diffs = pd.Series(ts).diff().dropna()
diffs_min = diffs.dt.total_seconds() / 60

print(f"\nGap analysis on {len(ts)} merged timestamps")
print(f"  Min gap : {diffs.min()}")
print(f"  Max gap : {diffs.max()}")
print(f"  Mean    : {diffs.mean()}")
print(f"  Median  : {diffs.median()}")

bins   = [0, 10, 16, 30, 60, 120, 300, 1440, int(1e9)]
labels = ['<10m','10-16m','16-30m','30-60m','1-2h','2-5h','5h-1d','>1d']
cut = pd.cut(diffs_min, bins=bins, labels=labels, right=False)
print("\nGap distribution:")
print(cut.value_counts().sort_index().to_string())

# ---- Contiguous run lengths ----
gap_threshold = pd.Timedelta(minutes=16)
run_lengths = []
run = 1
for g in diffs:
    if g <= gap_threshold:
        run += 1
    else:
        run_lengths.append(run)
        run = 1
run_lengths.append(run)
run_lengths = np.array(run_lengths)

print(f"\nContiguous 15-min runs (gap <= 16 min):")
print(f"  Number of runs : {len(run_lengths)}")
print(f"  Longest run    : {run_lengths.max()} timesteps  ({run_lengths.max()*15/60:.1f} h)")
print(f"  Median run     : {np.median(run_lengths):.0f} timesteps")
print(f"  Runs >= 6  ts  : {(run_lengths >= 6).sum()}")
print(f"  Runs >= 12 ts  : {(run_lengths >= 12).sum()}")
print(f"  Runs >= 48 ts  : {(run_lengths >= 48).sum()}")
print(f"  Runs >= 120 ts : {(run_lengths >= 120).sum()}")

# ---- Valid windows per window_size ----
print("\nValid target windows per window_size (max_gap=16 min):")
for w in [1, 3, 6, 12, 24, 48, 120]:
    valid = sum(
        1 for t in range(w, len(ts))
        if (pd.Series(ts[t-w:t+1]).diff().dropna() <= gap_threshold).all()
    )
    print(f"  window={w:4d}  =>  {valid} valid windows")

# ---- Check if the timestamps array passed to SpatioTemporalDataset
#      actually has T entries equal to what heterodata expects ----
print(f"\nTimestamp array that would be passed to SpatioTemporalDataset: {len(ts)} entries")
print(f"  (This must equal the T dimension of heterodata[node_type].x)")

# ---- Show top 20 largest gaps for inspection ----
large_gaps = diffs[diffs > pd.Timedelta(minutes=16)].sort_values(ascending=False).head(20)
print(f"\nTop 20 largest gaps (> 16 min):")
gap_df = pd.DataFrame({
    'gap': large_gaps.values,
    'before': ts[large_gaps.index - 1],
    'after':  ts[large_gaps.index],
})
print(gap_df.to_string(index=False))
