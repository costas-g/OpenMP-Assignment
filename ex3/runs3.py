#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import numpy as np
import pandas as pd
from typing import Optional # for optional function arguments

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Run benchmarks (serial vs parallel)")
parser.add_argument("-s", "--skip-experiments", action="store_true",
                    help="Skip running experiments and only load the stats CSV")
args = parser.parse_args()

# ---------------- PATHS ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")

EXE_PATH = os.path.join(BIN_DIR, "main")
# if not os.path.exists(EXE_PATH) and os.path.exists(EXE_PATH + ".exe"):
#     EXE_PATH = EXE_PATH + ".exe"
if not os.path.exists(EXE_PATH):
    raise FileNotFoundError(f"Executable not found: {EXE_PATH}")

DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
# LOGS_DIR = os.path.join(RUNS_DIR, "logs")
# os.makedirs(LOGS_DIR, exist_ok=True)

STATS_CSV = os.path.join(RUNS_DIR, "results_stats.csv")

# ---------------- CONFIG ----------------
DEGREES = [10**7, 10**8]
THREADS = [1, 2, 3, 4, 5, 6, 8]
REPEATS = 2

# ---------------- REGEX ----------------
gen_regex = re.compile(r"^\s*Generate Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
serial_regex = re.compile(r"^\s*Serial Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
parallel_regex = re.compile(r"^\s*Parallel Time\s*\(s\):\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)

sorted_ok_regex = re.compile(r"Correct sorting!", re.IGNORECASE)
not_sorted_regex = re.compile(r"Incorrect sorting|Not sorted|ERROR:\s*Incorrect sorting", re.IGNORECASE)


def run_single(degree: int, method: str, threads: Optional[int]):
    """
    Run once. Returns (gen_time, serial_time, parallel_time, ok_sorted, output).
    Program prints only one of serial/parallel times depending on method.
    """
    if method == "s":
        cmd = [EXE_PATH, str(degree), "s"]
    elif method == "p":
        if threads is None:
            raise ValueError("threads must be provided for parallel runs")
        cmd = [EXE_PATH, str(degree), "p", str(threads)]
    else:
        raise ValueError("method must be 's' or 'p'")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        raise RuntimeError(f"Run failed for cmd={cmd}\n{out}") from e

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    gen_time = None
    serial_time = None
    parallel_time = None
    ok_sorted = False
    bad_sorted = False

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

        if sorted_ok_regex.search(line):
            ok_sorted = True
        if not_sorted_regex.search(line):
            bad_sorted = True

    if bad_sorted:
        ok_sorted = False

    return gen_time, serial_time, parallel_time, ok_sorted, output


def run_serial_baseline(degree: int, repeats: int):
    gen_times = []
    serial_times = []

    for it in range(repeats):
        gen_t, ser_t, par_t, ok, out = run_single(degree, "s", None)
        if not ok or gen_t is None or ser_t is None:
            continue
        gen_times.append(gen_t)
        serial_times.append(ser_t)

    if len(serial_times) == 0:
        return None

    return {
        "serial_mean": float(np.mean(serial_times)),
        "serial_std": float(np.std(serial_times, ddof=1)) if len(serial_times) > 1 else 0.0,
    }

def run_parallel_combo_and_stats(degree: int, threads: int, repeats: int, serial_mean: float, serial_std: float):
    gen_times = []
    parallel_times = []

    for it in range(repeats):
        gen_t, ser_t, par_t, ok, out = run_single(degree, "p", threads)
        if not ok or gen_t is None or par_t is None:
            continue
        gen_times.append(gen_t)
        parallel_times.append(par_t)

    if len(parallel_times) == 0:
        return None

    gen_mean = float(np.mean(gen_times))
    gen_std  = float(np.std(gen_times, ddof=1)) if len(gen_times) > 1 else 0.0

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
    cols = [
        "degree,threads,gen_mean,gen_std,serial_mean,serial_std,"
        "parallel_mean,parallel_std,parallel_min,parallel_max,speedup,repeats"
    ][0].split(",")

    if os.path.exists(STATS_CSV):
        os.remove(STATS_CSV)

    file_exists = False

    for degree in DEGREES:
        print(f"[INFO] Serial baseline degree={degree} repeats={REPEATS}")
        base = run_serial_baseline(degree, REPEATS)
        if base is None:
            raise RuntimeError(f"No successful serial samples for degree={degree}")

        ser_mean = base["serial_mean"]
        ser_std  = base["serial_std"]

        for threads in THREADS:
            print(f"[INFO] Parallel combo degree={degree}, threads={threads}, repeats={REPEATS}")

            row = run_parallel_combo_and_stats(degree, threads, REPEATS, ser_mean, ser_std)
            if row is None:
                print(f"[WARN] No successful parallel samples for degree={degree}, threads={threads}. Skipping.")
                continue

            pd.DataFrame([row], columns=cols).to_csv(
                STATS_CSV,
                mode="a",
                header=not file_exists,
                index=False
            )
            file_exists = True
            print(f"[INFO] Appended -> {STATS_CSV}")


def main():
    print(f"[INFO] Executable: {EXE_PATH}")
    print(f"[INFO] Output CSV: {STATS_CSV}")

    if args.skip_experiments:
        if not os.path.exists(STATS_CSV):
            raise FileNotFoundError(f"--skip-experiments but CSV not found: {STATS_CSV}")
        df = pd.read_csv(STATS_CSV)
        print(df)
        return

    df = run_experiments_write_stats()
    print(df)


if __name__ == "__main__":
    main()
