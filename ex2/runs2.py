#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------

ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")

EXE_PATH = os.path.join(BIN_DIR, "main")
if not os.path.exists(EXE_PATH) and os.path.exists(EXE_PATH + ".exe"):
    EXE_PATH = EXE_PATH + ".exe"

MATRIX_SIZE = [10**3, 10**4]

# IMPORTANT: your main expects sparsity as float in [0,1]
SPARSITY = [x / 100.0 for x in [0, 25, 50, 75, 90, 95, 99]]  # 0.01, 0.06, ..., 0.96

NUM_MULTS = [1, 5, 10, 20]
THREADS   = [1, 2, 4, 8]

DATA_DIR  = os.path.join(ROOT, "data")
RUNS_DIR  = os.path.join(DATA_DIR, "runs")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MEANS_CSV_PATH = os.path.join(RUNS_DIR, "matrix_results_means.csv")


# ---------------- REGEXES (match your sample output) ----------------

FLOAT_PAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"

csr_build_serial   = re.compile(r"Serial CSR build time \(s\):\s*" + FLOAT_PAT)
csr_build_parallel = re.compile(r"Parallel CSR build time \(s\):\s*" + FLOAT_PAT)

dense_mult_serial   = re.compile(r"Dense matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT)
dense_mult_parallel = re.compile(r"Dense matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT)

csr_mult_serial   = re.compile(r"Sparse matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT)
csr_mult_parallel = re.compile(r"Sparse matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT)


def _last_float(pat: re.Pattern, text: str) -> Optional[float]:
    m = None
    for m in pat.finditer(text):
        pass
    return float(m.group(1)) if m else None


def parse_output(output: str) -> Dict[str, Optional[float]]:
    return {
        "csr_build_serial_s":   _last_float(csr_build_serial, output),
        "csr_build_parallel_s": _last_float(csr_build_parallel, output),
        "dense_mult_serial_s":  _last_float(dense_mult_serial, output),
        "dense_mult_parallel_s":_last_float(dense_mult_parallel, output),
        "csr_mult_serial_s":    _last_float(csr_mult_serial, output),
        "csr_mult_parallel_s":  _last_float(csr_mult_parallel, output),
    }


def run_once(exe_path: str, N: int, sp: float, k: int, t: int) -> Dict[str, Any]:
    cmd = [exe_path, str(N), f"{sp:.6f}", str(k), str(t)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_output(output)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "output": output,
        **parsed
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return mean, std


def run_experiment_means(exe_path: str, N: int, sp: float, k: int, t: int, repeats: int, log_mode: str, run_id: str) -> Dict[str, Any]:
    """
    Runs repeats; keeps only successful parses for averaging.
    If a run crashes (rc!=0) or missing metrics -> counted as failed.
    """
    key_fields = [
        "csr_build_serial_s","csr_build_parallel_s",
        "dense_mult_serial_s","dense_mult_parallel_s",
        "csr_mult_serial_s","csr_mult_parallel_s"
    ]

    ok = {f: [] for f in key_fields}
    rc_list = []

    for rep in range(1, repeats + 1):
        r = run_once(exe_path, N, sp, k, t)
        rc = int(r["returncode"])
        rc_list.append(rc)

        missing = [f for f in key_fields if r.get(f) is None]
        is_fail = (rc != 0) or (len(missing) > 0)

        if is_fail:
            continue

        for f in key_fields:
            ok[f].append(float(r[f]))

    n_ok = len(ok[key_fields[0]])
    n_fail = repeats - n_ok

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "matrix_size": N,
        "sparsity": sp,
        "num_mults": k,
        "threads": t,
        "repeats": repeats,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "rc_last": rc_list[-1] if rc_list else 0,
    }

    # If nothing succeeded, keep NaNs
    if n_ok == 0:
        for f in key_fields:
            row[f"{f}_mean"] = np.nan
            row[f"{f}_std"]  = np.nan
        row["csr_build_speedup"] = np.nan
        row["dense_speedup"] = np.nan
        row["csr_speedup"] = np.nan
        row["ratio_dense_over_csr_serial"] = np.nan
        row["ratio_dense_over_csr_parallel"] = np.nan
        return row

    # Means/stds
    for f in key_fields:
        m, s = mean_std(ok[f])
        row[f"{f}_mean"] = m
        row[f"{f}_std"]  = s

    # Speedups (use means)
    row["csr_build_speedup"] = row["csr_build_serial_s_mean"] / row["csr_build_parallel_s_mean"]
    row["dense_speedup"] = row["dense_mult_serial_s_mean"] / row["dense_mult_parallel_s_mean"]
    row["csr_speedup"]   = row["csr_mult_serial_s_mean"] / row["csr_mult_parallel_s_mean"]

    # CSR vs Dense (per-mult) ratios (>1 => CSR faster)
    row["ratio_dense_over_csr_serial"] = row["dense_mult_serial_s_mean"] / row["csr_mult_serial_s_mean"]
    row["ratio_dense_over_csr_parallel"] = row["dense_mult_parallel_s_mean"] / row["csr_mult_parallel_s_mean"]

    return row


def append_row_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--log-mode", choices=["none","fail","all"], default="none",
                    help="Save logs: none (default), fail, or all.")
    ap.add_argument("--skip-experiments", action="store_true",
                    help="Skip running and only plot from existing means CSV.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(EXE_PATH):
        raise SystemExit(f"[ERROR] Executable not found: {EXE_PATH}")

    if args.skip_experiments:
        if not os.path.exists(MEANS_CSV_PATH):
            raise SystemExit(f"[ERROR] Missing means CSV: {MEANS_CSV_PATH}")
    else:
        # Overwrite means CSV each new full run
        if os.path.exists(MEANS_CSV_PATH):
            os.remove(MEANS_CSV_PATH)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        total = len(MATRIX_SIZE) * len(SPARSITY) * len(NUM_MULTS) * len(THREADS)
        c = 0

        # fixed header
        fieldnames = [
            "timestamp","matrix_size","sparsity","num_mults","threads","repeats","n_ok","n_fail","rc_last",
            "csr_build_serial_s_mean","csr_build_serial_s_std",
            "csr_build_parallel_s_mean","csr_build_parallel_s_std",
            "dense_mult_serial_s_mean","dense_mult_serial_s_std",
            "dense_mult_parallel_s_mean","dense_mult_parallel_s_std",
            "csr_mult_serial_s_mean","csr_mult_serial_s_std",
            "csr_mult_parallel_s_mean","csr_mult_parallel_s_std",
            "csr_build_speedup","dense_speedup","csr_speedup",
            "ratio_dense_over_csr_serial","ratio_dense_over_csr_parallel"
        ]

        for N in MATRIX_SIZE:
            for sp in SPARSITY:
                for k in NUM_MULTS:
                    for t in THREADS:
                        c += 1

                        # Control repeats to reduce test time
                        if N >= 10**4 and sp < 0.95:
                            final_repeats = 2 # REDUCED REPEATS
                            if k >= 10 and t < 4:
                                final_repeats = 1 # MINIMUM REPEATS
                        else:
                            final_repeats = 3 # DEFAULT REPEATS
                        
                        if args.repeats > 1:
                            final_repeats = args.repeats # Repeats hard override by explicit argument

                        print(f"[INFO] {c}/{total} | N={N} sp={sp:.2f} k={k} t={t} (running {final_repeats} times)")

                        row = run_experiment_means(EXE_PATH, N, sp, k, t, final_repeats, args.log_mode, run_id)

                        # If it crashed at least once, note it
                        if row["n_ok"] == 0 or row["n_fail"] > 0 or row["rc_last"] != 0:
                            print(f"[WARN] Partial/failed runs for N={N} sp={sp:.2f} k={k} t={t} "
                                  f"(n_ok={row['n_ok']}, n_fail={row['n_fail']}, rc_last={row['rc_last']})")

                        append_row_csv(MEANS_CSV_PATH, row, fieldnames)

        print(f"[INFO] Saved means CSV: {MEANS_CSV_PATH}")
    '''
    if not args.no_plots:
        make_plots(MEANS_CSV_PATH)
    '''

if __name__ == "__main__":
    main()
