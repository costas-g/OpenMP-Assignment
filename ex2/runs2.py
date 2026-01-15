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

MATRIX_SIZE = [10**3, 5000]

# IMPORTANT: your main expects sparsity as float in [0,1]
SPARSITY = [x / 100.0 for x in [0, 90, 99]]  # 0.01, 0.06, ..., 0.96

NUM_MULTS = [1, 5, 10]
THREADS   = [2, 4]

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
    
    
'''
# ---------- RUNNING THE PROGRAM ----------
def run_single(degree: int, threads: int):
    """
    Run ./bin/main <degree> <threads> once and return (gen_time, serial_time, parallel_time).
    Raises RuntimeError if results don't match or times can't be parsed.
    """
    cmd = [EXE_PATH, str(degree), str(threads)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Running {cmd}:")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    gen_time = serial_time = parallel_time = None
    results_match = False

    for line in output.splitlines():
        m = gen_regex.search(line)
        if m:
            gen_time = float(m.group(1))

        m = serial_regex.search(line)
        if m:
            serial_time = float(m.group(1))

        m = parallel_regex.search(line)
        if m:
            parallel_time = float(m.group(1))

        if match_regex.search(line):
            results_match = True

    if gen_time is None or serial_time is None or parallel_time is None:
        print("\n[ERROR] Could not parse times from output:")
        print(output)
        raise RuntimeError("Failed to parse timing lines")

    if not results_match:
        print(f"\n[ERROR] Results did NOT match for degree={degree}, threads={threads}.")
        print("Output:\n", output)
        raise RuntimeError("Results mismatch")

    return gen_time, serial_time, parallel_time
'''


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
        row["ratio_dense_over_csr_serial_per_mult"] = np.nan
        row["ratio_dense_over_csr_parallel_per_mult"] = np.nan
        return row

    # Means/stds
    for f in key_fields:
        m, s = mean_std(ok[f])
        row[f"{f}_mean"] = m
        row[f"{f}_std"]  = s

    # Speedups (use means)
    row["csr_build_speedup"] = row["csr_build_serial_s_mean"] / row["csr_build_parallel_s_mean"]
    row["dense_speedup"] = (row["dense_mult_serial_s_mean"] / k) / (row["dense_mult_parallel_s_mean"] / k)
    row["csr_speedup"]   = (row["csr_mult_serial_s_mean"] / k) / (row["csr_mult_parallel_s_mean"] / k)

    # CSR vs Dense (per-mult) ratios (>1 => CSR faster)
    row["ratio_dense_over_csr_serial_per_mult"] = (row["dense_mult_serial_s_mean"] / k) / (row["csr_mult_serial_s_mean"] / k)
    row["ratio_dense_over_csr_parallel_per_mult"] = (row["dense_mult_parallel_s_mean"] / k) / (row["csr_mult_parallel_s_mean"] / k)

    return row


def append_row_csv(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def make_plots(means_csv: str) -> None:
    df = pd.read_csv(means_csv)

    # Keep only rows with valid metrics
    df_ok = df[df["n_ok"] > 0].copy()

    # 1) CSR build scaling speedup vs threads (show few sparsities)
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        sparsities = sorted(subN["sparsity"].unique())
        reps = sparsities[:6]  # keep plot readable

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].sort_values("threads")
            ax.plot(g["threads"], g["csr_build_speedup"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (CSR build)")
        ax.set_title(f"CSR build speedup vs threads (N={N})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_build_speedup_N{N}.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    # 2) CSR mult scaling vs threads (avg over num_mults)
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique())[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].groupby("threads")["csr_speedup"].mean().reset_index().sort_values("threads")
            ax.plot(g["threads"], g["csr_speedup"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (CSR mult)")
        ax.set_title(f"CSR mult speedup vs threads (N={N}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_mult_speedup_N{N}.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    # 3) Dense mult scaling vs threads (avg over num_mults)
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        reps = sorted(subN["sparsity"].unique())[:6]

        fig, ax = plt.subplots()
        for sp in reps:
            g = subN[subN["sparsity"] == sp].groupby("threads")["dense_speedup"].mean().reset_index().sort_values("threads")
            ax.plot(g["threads"], g["dense_speedup"], "o--", label=f"sp={sp:.2f}")

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (Dense mult)")
        ax.set_title(f"Dense mult speedup vs threads (N={N}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"dense_mult_speedup_N{N}.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    # 4) CSR vs Dense ratio vs sparsity (t=1 and t=max threads), avg over num_mults
    for N in sorted(df_ok["matrix_size"].unique()):
        subN = df_ok[df_ok["matrix_size"] == N].copy()
        maxT = int(subN["threads"].max())

        serial = subN[subN["threads"] == 1].groupby("sparsity")["ratio_dense_over_csr_serial_per_mult"].mean().reset_index().sort_values("sparsity")
        par    = subN[subN["threads"] == maxT].groupby("sparsity")["ratio_dense_over_csr_parallel_per_mult"].mean().reset_index().sort_values("sparsity")

        fig, ax = plt.subplots()
        ax.plot(serial["sparsity"], serial["ratio_dense_over_csr_serial_per_mult"], "o--", label="t=1")
        ax.plot(par["sparsity"], par["ratio_dense_over_csr_parallel_per_mult"], "o--", label=f"t={maxT}")
        ax.axhline(1.0, linestyle="--")

        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Dense_per_mult / CSR_per_mult  (>1 => CSR faster)")
        ax.set_title(f"CSR vs Dense ratio vs sparsity (N={N}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_vs_dense_ratio_N{N}.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Plots saved in: {PLOTS_DIR}")


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
            "ratio_dense_over_csr_serial_per_mult","ratio_dense_over_csr_parallel_per_mult"
        ]

        for N in MATRIX_SIZE:
            for sp in SPARSITY:
                for k in NUM_MULTS:
                    
                    for t in THREADS:
                        c += 1
                        print(f"[INFO] {c}/{total} | N={N} sp={sp:.2f} k={k} t={t}")

                        row = run_experiment_means(EXE_PATH, N, sp, k, t, args.repeats, args.log_mode, run_id)

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
