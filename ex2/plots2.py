#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



# ---------------- CONFIG ----------------

ROOT = os.path.abspath(os.path.dirname(__file__))
# BIN_DIR = os.path.join(ROOT, "bin")

# EXE_PATH = os.path.join(BIN_DIR, "main")
# if not os.path.exists(EXE_PATH) and os.path.exists(EXE_PATH + ".exe"):
#     EXE_PATH = EXE_PATH + ".exe"

# MATRIX_SIZE = [10**3, 10**4]

# # IMPORTANT: your main expects sparsity as float in [0,1]
# SPARSITY = [x / 100.0 for x in range(1, 99, 20)]  # 0.01, 0.06, ..., 0.96

# NUM_MULTS = list(range(1, 20, 5))
# THREADS   = list(range(1, 9))

DATA_DIR  = os.path.join(ROOT, "data")
RUNS_DIR  = os.path.join(DATA_DIR, "runs")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MEANS_CSV_PATH = os.path.join(RUNS_DIR, "matrix_results_means.csv")


# # ---------------- REGEXES (match your sample output) ----------------

# FLOAT_PAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"

# csr_build_serial   = re.compile(r"Serial CSR build time \(s\):\s*" + FLOAT_PAT)
# csr_build_parallel = re.compile(r"Parallel CSR build time \(s\):\s*" + FLOAT_PAT)

# dense_mult_serial   = re.compile(r"Dense matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT)
# dense_mult_parallel = re.compile(r"Dense matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT)

# csr_mult_serial   = re.compile(r"Sparse matrix \d+x mult Serial time \(s\):\s*" + FLOAT_PAT)
# csr_mult_parallel = re.compile(r"Sparse matrix \d+x mult Parallel time \(s\):\s*" + FLOAT_PAT)



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


        # <--- Minimal fix: force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (CSR build)")
        ax.set_title(f"CSR build speedup vs threads (N={N:,})")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_build_speedup_N{N//1000}K.svg")
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

        # <--- Minimal fix: force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (CSR mult)")
        ax.set_title(f"CSR mult speedup vs threads (N={N:,}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_mult_speedup_N{N//1000}K.svg")
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

        # <--- Minimal fix: force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (Dense mult)")
        ax.set_title(f"Dense mult speedup vs threads (N={N:,}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"dense_mult_speedup_N{N//1000}K.svg")
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
        ax.set_title(f"CSR vs Dense ratio vs sparsity (N={N:,}) (avg over num_mults)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"csr_vs_dense_ratio_N{N//1000}K.svg")
        fig.savefig(out, format="svg", bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Plots saved in: {PLOTS_DIR}")


def main():
    make_plots(MEANS_CSV_PATH)
        
if __name__ == "__main__":
    main()
