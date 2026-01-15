#!/usr/bin/env python3
"""
Benchmark driver for the polynomial multiplication experiments.

Usage:
    py runs.py           # run experiments, build stats, write stats CSV
"""
import os
import re
import subprocess
import numpy as np
import pandas as pd
import argparse

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")

parser = argparse.ArgumentParser(description="Run polynomial multiplication (serial vs parallel)")
parser.add_argument("-s", "--skip-experiments", action="store_true",
                    help="Skip running experiments and use the temporary raw CSV")
parser.add_argument("--keep-raw", action="store_true",
                    help="Keep the temporary raw CSV (otherwise it is deleted at the end).")
args = parser.parse_args()

EXE_PATH = os.path.join(BIN_DIR, "main")

DEGREE_VALUES = [10 ** p for p in range(3, 6)]  # 1e3..1e5
THREAD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]
THREAD_VALUES_SHORT = [1, 2, 4, 8]

REPEATS = 5
REPEATS_SHORT = 3

# Times
gen_regex      = re.compile(r"Generate Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)")
serial_regex   = re.compile(r"Serial Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)")
parallel_regex = re.compile(r"Parallel Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)")

# Speedup
speedup_regex  = re.compile(r"Speedup:\s*([0-9]+(?:\.[0-9]+)?)")

# Correctness / errors
match_regex    = re.compile(r"Results match!", re.IGNORECASE)
mismatch_regex = re.compile(r"Mismatch at i=\s*(\d+):\s*serial=\s*(-?\d+),\s*parallel=\s*(-?\d+)", re.IGNORECASE)
error_regex    = re.compile(r"\bERROR\b", re.IGNORECASE)

data_root = os.path.join(ROOT, "data")
os.makedirs(data_root, exist_ok=True)

runs_root = os.path.join(data_root, "runs")
os.makedirs(runs_root, exist_ok=True)

# FINAL output CSV (stats)
STATS_CSV = os.path.join(runs_root, "omp_poly_results_stats.csv")


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


def run_combo_and_stats(degree, threads, repeats):
    """
    Run run_single() 'repeats' times for a single (degree,threads) combo,
    compute stats, and return ONE dict row for the final CSV.
    """
    gen_times = []
    serial_times = []
    parallel_times = []

    for it in range(repeats):
        try:
            gen_time, serial_time, parallel_time = run_single(degree, threads)
        except RuntimeError as e:
            print("[ERROR]", e)
            continue

        gen_times.append(gen_time)
        serial_times.append(serial_time)
        parallel_times.append(parallel_time)

    # If all repeats failed, skip this combo
    if len(parallel_times) == 0 or len(serial_times) == 0:
        return None

    gen_mean = float(np.mean(gen_times))
    gen_std  = float(np.std(gen_times, ddof=1)) if len(gen_times) > 1 else 0.0

    serial_mean = float(np.mean(serial_times))
    serial_std  = float(np.std(serial_times, ddof=1)) if len(serial_times) > 1 else 0.0

    parallel_mean = float(np.mean(parallel_times))
    parallel_std  = float(np.std(parallel_times, ddof=1)) if len(parallel_times) > 1 else 0.0
    parallel_min  = float(np.min(parallel_times))
    parallel_max  = float(np.max(parallel_times))
    
    
    speedup = serial_mean / parallel_mean if parallel_mean > 0 else float("inf")
    
    return {
        "degree": degree,
        "threads": threads,

        "gen_mean": gen_mean,
        "gen_std": gen_std,

        "serial_mean": serial_mean,
        "serial_std": serial_std,

        "parallel_mean": parallel_mean,
        "parallel_std": parallel_std,
        "parallel_min": parallel_min,
        "parallel_max": parallel_max,

        "speedup": speedup,
        
        "repeats": len(parallel_times),
    }


def run_experiments_write_stats():
    """
    For each (degree, threads):
      run repeats -> compute stats -> append ONE row to STATS_CSV
    """
    # Fresh run: delete stats file if it exists
    if os.path.exists(STATS_CSV):
        os.remove(STATS_CSV)

    file_exists = False

    split_value = 10**5
    idx = np.searchsorted(DEGREE_VALUES, split_value, side="left")
    small_degrees = DEGREE_VALUES[:idx]
    big_degrees = DEGREE_VALUES[idx:]

    full_combos = len(small_degrees) * len(THREAD_VALUES)
    full_runs  = full_combos * REPEATS
    short_combos = len(big_degrees)  * len(THREAD_VALUES_SHORT)
    short_runs = short_combos * REPEATS_SHORT
    total_runs = full_runs + short_runs
    total_combos = full_combos + short_combos
    combo_counter = 0

    for degree in DEGREE_VALUES:
        if degree >= split_value:
            my_repeats = REPEATS_SHORT
            my_threads = THREAD_VALUES_SHORT
        else:
            my_repeats = REPEATS
            my_threads = THREAD_VALUES

        for threads in my_threads:
            combo_counter += 1
            print(f"[INFO] Combo {combo_counter}/{total_combos} | degree={degree}, threads={threads}, repeats={my_repeats}")

            row = run_combo_and_stats(degree, threads, my_repeats)
            if row is None:
                print(f"[WARN] No successful samples for degree={degree}, threads={threads}. Skipping.")
                continue

            pd.DataFrame([row]).to_csv(
                STATS_CSV,
                mode="a",
                header=not file_exists,
                index=False
            )
            file_exists = True
            print(f"[INFO] Appended -> {STATS_CSV}")

# ---------- MAIN ----------
def main():
    print(f"[INFO] Executable: {EXE_PATH}")
    print(f"[INFO] Degrees: {DEGREE_VALUES}")
    print(f"[INFO] Threads: {THREAD_VALUES}")
    print(f"[INFO] REPEATS: {REPEATS}")
    print(f"[INFO] Stats CSV: {STATS_CSV}")

    if args.skip_experiments:
        if not os.path.exists(STATS_CSV):
            raise FileNotFoundError(f"--skip-experiments but stats CSV not found: {STATS_CSV}")
        stats_df = pd.read_csv(STATS_CSV)
        print(f"[INFO] Loaded stats CSV (rows={len(stats_df)})")
        return

    run_experiments_write_stats()

    stats_df = pd.read_csv(STATS_CSV)
    print(f"[INFO] Done. Final stats rows: {len(stats_df)}")
    # TODO: plots/tables using stats_df

if __name__ == "__main__":
    main()
