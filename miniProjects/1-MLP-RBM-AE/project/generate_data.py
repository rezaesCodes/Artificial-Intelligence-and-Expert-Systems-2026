from pathlib import Path
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

SEED_DEFAULT = 12345678
SOURCE_URLS = [
    "https://raw.githubusercontent.com/coderkhongodo/data_wind/main/T1.csv",
    "https://raw.githubusercontent.com/coderkhongodo/data_wind/master/T1.csv",
]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TIMESTAMP = "Date/Time"
NUMERIC_COLUMNS = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
]

def obtain_source(explicit_path=None):
    candidates = [Path(explicit_path)] if explicit_path else []
    candidates += [ROOT / "T1.csv", RAW_DIR / "T1.csv"]
    for path in candidates:
        if path and path.is_file() and path.stat().st_size:
            return path, "local"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "T1.csv"
    errors = []
    for url in SOURCE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if len(payload) < 1000 or b"Date/Time" not in payload[:1000]:
                raise RuntimeError("Downloaded file does not look like T1.csv.")
            target.write_bytes(payload)
            return target, url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not obtain T1.csv. Put the original file at proj1_1/T1.csv.\n" + "\n".join(errors))

def load_source(path):
    frame = pd.read_csv(path)
    missing = [c for c in [TIMESTAMP, *NUMERIC_COLUMNS] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame[[TIMESTAMP, *NUMERIC_COLUMNS]].copy()
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], format="%d %m %Y %H:%M", errors="coerce")
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP]).sort_values(TIMESTAMP)
    frame = frame.drop_duplicates(subset=[TIMESTAMP], keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid timestamped rows were found.")
    return frame

def inject_faults(clean, seed, missing_fraction, spike_fraction, noise_fraction, noise_scale):
    rng = np.random.default_rng(seed)
    student = clean.copy()
    values = student[NUMERIC_COLUMNS].to_numpy(dtype=float, copy=True)
    n_rows, n_cols = values.shape
    total = n_rows * n_cols
    missing_n = max(1, int(round(total * missing_fraction))) if missing_fraction > 0 else 0
    spike_n = max(1, int(round(total * spike_fraction))) if spike_fraction > 0 else 0
    noise_n = max(1, int(round(total * noise_fraction))) if noise_fraction > 0 else 0
    all_positions = rng.permutation(total)
    missing_positions = all_positions[:missing_n]
    spike_positions = all_positions[missing_n:missing_n + spike_n]
    noise_positions = all_positions[missing_n + spike_n:missing_n + spike_n + noise_n]
    missing_rows = np.zeros(n_rows, dtype=bool)
    spike_rows = np.zeros(n_rows, dtype=bool)
    noise_rows = np.zeros(n_rows, dtype=bool)
    for pos in missing_positions:
        row, col = divmod(int(pos), n_cols)
        values[row, col] = np.nan
        missing_rows[row] = True
    for pos in spike_positions:
        row, col = divmod(int(pos), n_cols)
        if np.isnan(values[row, col]):
            continue
        finite = clean[NUMERIC_COLUMNS[col]].dropna()
        spread = float(finite.quantile(0.75) - finite.quantile(0.25))
        if not np.isfinite(spread) or spread == 0:
            spread = float(finite.std())
        direction = -1.0 if rng.random() < 0.5 else 1.0
        values[row, col] += direction * max(spread * 6.0, 1.0)
        spike_rows[row] = True
    for pos in noise_positions:
        row, col = divmod(int(pos), n_cols)
        if np.isnan(values[row, col]):
            continue
        finite = clean[NUMERIC_COLUMNS[col]].dropna()
        spread = float(finite.quantile(0.75) - finite.quantile(0.25))
        if not np.isfinite(spread) or spread == 0:
            spread = float(finite.std())
        values[row, col] += rng.normal(0.0, max(spread * noise_scale, 1e-6))
        noise_rows[row] = True
    student.loc[:, NUMERIC_COLUMNS] = values
    labels = pd.DataFrame({
        "missing_injected": missing_rows.astype(int),
        "spike_injected": spike_rows.astype(int),
        "noise_injected": noise_rows.astype(int),
        "injected_fault": (missing_rows | spike_rows | noise_rows).astype(int),
    })
    return student, labels, missing_n, spike_n, noise_n

def main():
    parser = argparse.ArgumentParser(description="Prepare the student-facing wind-turbine dataset.")
    parser.add_argument("--source", type=str, default=None, help="Optional path to T1.csv.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--missing-fraction", type=float, default=0.003)
    parser.add_argument("--spike-fraction", type=float, default=0.001)
    parser.add_argument("--noise-fraction", type=float, default=0.01)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    args = parser.parse_args()
    if any(not 0 <= value < 1 for value in [args.missing_fraction, args.spike_fraction, args.noise_fraction]):
        raise ValueError("Fault fractions must be between 0 and 1.")
    if args.noise_scale < 0:
        raise ValueError("Noise scale must be non-negative.")
    source_path, source_origin = obtain_source(args.source)
    clean = load_source(source_path)
    student, labels, missing_n, spike_n, noise_n = inject_faults(
        clean, args.seed, args.missing_fraction, args.spike_fraction,
        args.noise_fraction, args.noise_scale
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DATA_DIR / "clean_data.csv", index=False)
    student.to_csv(DATA_DIR / "student_data.csv", index=False)
    labels.to_csv(DATA_DIR / "injected_fault_labels.csv", index=False)
    gaps = clean[TIMESTAMP].diff().dropna()
    metadata = {
        "seed": args.seed,
        "source_file": source_path.name,
        "source_origin": source_origin,
        "source_dataset": "T1.csv wind turbine SCADA dataset from coderkhongodo/data_wind",
        "timestamp_column": TIMESTAMP,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": "LV ActivePower (kW)",
        "mlp_features": ["Wind Speed (m/s)", "Wind Direction (°)"],
        "theoretical_reference_column": "Theoretical_Power_Curve (KWh)",
        "rows": int(len(clean)),
        "start_time": str(clean[TIMESTAMP].min()),
        "end_time": str(clean[TIMESTAMP].max()),
        "most_common_sampling_interval": str(gaps.mode().iloc[0]) if len(gaps) else None,
        "nonstandard_time_gaps": int((gaps != gaps.mode().iloc[0]).sum()) if len(gaps) else 0,
        "missing_cells_injected": int(missing_n),
        "spike_cells_injected": int(spike_n),
        "gaussian_noise_cells_injected": int(noise_n),
        "gaussian_noise_std_as_column_iqr_fraction": args.noise_scale,
        "synthetic_faults_only": True,
        "note": "The source file contains no real row-level fault labels. Missing values, spikes, and small Gaussian sensor perturbations are synthetic teaching examples.",
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(clean):,} rows from {source_path.name}.")
    print(f"Injected missing cells: {missing_n:,}; spikes: {spike_n:,}; noisy cells: {noise_n:,}.")
    print(f"Output directory: {DATA_DIR}")

if __name__ == "__main__":
    main()
