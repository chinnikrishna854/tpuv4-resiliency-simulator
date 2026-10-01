#!/usr/bin/env python3
"""Small simulator for gang-scheduled accelerator allocation.

The model is intentionally abstract: each cube contains 64 accelerator
resources. A request succeeds only when its resources are available at the
same time. Static allocation requires a contiguous run of complete cubes;
reconfigurable allocation can combine healthy free resources across cubes.

Usage:
    python3 tpuv4_simulator.py --outdir results
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt


TPUS_PER_CUBE = 64


@dataclass
class Config:
    cubes: int = 64
    tpus_per_cube: int = TPUS_PER_CUBE
    failure_probability: float = 0.001
    occupancy_probability: float = 0.01
    trials: int = 5000
    reconfig_overhead: float = 0.0
    correlated_failure_probability: float = 0.0


def make_state(cfg: Config, rng: random.Random) -> list[list[bool]]:
    """Return a cube-by-TPU matrix; True means usable and free."""
    state = []
    correlated = rng.random() < cfg.correlated_failure_probability
    bad_cube = rng.randrange(cfg.cubes) if correlated else None

    for cube in range(cfg.cubes):
        cube_state = []
        cube_failed = cube == bad_cube
        for _ in range(cfg.tpus_per_cube):
            failed = cube_failed or rng.random() < cfg.failure_probability
            occupied = rng.random() < cfg.occupancy_probability
            cube_state.append(not failed and not occupied)
        state.append(cube_state)
    return state


def static_success(state: list[list[bool]], job_size: int, tpus_per_cube: int) -> bool:
    """Static pods require complete, contiguous cubes."""
    needed_cubes = (job_size + tpus_per_cube - 1) // tpus_per_cube
    run = 0
    for cube in state:
        if all(cube) and job_size >= 0:
            run += 1
            if run >= needed_cubes:
                return True
        else:
            run = 0
    return False


def reconfigurable_success(
    state: list[list[bool]], job_size: int, overhead: float
) -> bool:
    """Reconfiguration combines healthy free resources across cubes.

    ``overhead`` represents resources reserved for routing/control and is
    expressed as a fraction of the total capacity.
    """
    usable = sum(sum(cube) for cube in state)
    capacity_after_overhead = sum(len(cube) for cube in state) * (1 - overhead)
    return usable >= job_size and job_size <= capacity_after_overhead


def estimate_many(cfg: Config, job_sizes: list[int], seed: int = 7) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)
    static_hits = {size: 0 for size in job_sizes}
    reconfig_hits = {size: 0 for size in job_sizes}
    for _ in range(cfg.trials):
        state = make_state(cfg, rng)
        for size in job_sizes:
            static_hits[size] += static_success(state, size, cfg.tpus_per_cube)
            reconfig_hits[size] += reconfigurable_success(state, size, cfg.reconfig_overhead)
    return {
        size: (static_hits[size] / cfg.trials, reconfig_hits[size] / cfg.trials)
        for size in job_sizes
    }


def estimate(cfg: Config, job_size: int, seed: int = 7) -> tuple[float, float]:
    return estimate_many(cfg, [job_size], seed)[job_size]


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def experiment_job_size(outdir: str) -> None:
    cfg = Config(trials=1200)
    rows = []
    sizes = list(range(128, 4097, 256))
    estimates = estimate_many(cfg, sizes, seed=100)
    for size in sizes:
        static, recon = estimates[size]
        rows.append({"job_tpus": size, "static": static, "reconfigurable": recon})
    write_csv(os.path.join(outdir, "experiment_job_size.csv"), rows)

    plt.figure(figsize=(7.2, 4.5))
    plt.plot([r["job_tpus"] for r in rows], [r["static"] for r in rows], "o-", label="Static contiguous pod")
    plt.plot([r["job_tpus"] for r in rows], [r["reconfigurable"] for r in rows], "o-", label="Reconfigurable")
    plt.xlabel("Requested job size (TPUs)")
    plt.ylabel("Probability of successful allocation")
    plt.title("Experiment 1: Larger gang-scheduled jobs")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "experiment_job_size.png"), dpi=180)
    plt.close()


def experiment_scale(outdir: str) -> None:
    rows = []
    for cubes in [8, 16, 32, 64, 96, 128]:
        cfg = Config(cubes=cubes, trials=1200)
        job = int(cubes * TPUS_PER_CUBE * 0.50)
        static, recon = estimate(cfg, job, seed=200 + cubes)
        rows.append({"cubes": cubes, "job_tpus": job, "static": static, "reconfigurable": recon})
    write_csv(os.path.join(outdir, "experiment_scale.csv"), rows)

    plt.figure(figsize=(7.2, 4.5))
    plt.plot([r["cubes"] for r in rows], [r["static"] for r in rows], "o-", label="Static contiguous pod")
    plt.plot([r["cubes"] for r in rows], [r["reconfigurable"] for r in rows], "o-", label="Reconfigurable")
    plt.xlabel("Infrastructure size (cubes)")
    plt.ylabel("Probability of successful allocation")
    plt.title("Experiment 2: Scaling the pod at a fixed 50% job share")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "experiment_scale.png"), dpi=180)
    plt.close()


def experiment_correlated_failures(outdir: str) -> None:
    rows = []
    for correlated in [0.0, 0.01, 0.03, 0.05, 0.10]:
        cfg = Config(correlated_failure_probability=correlated, trials=1500)
        static, recon = estimate(cfg, 4040, seed=300 + int(correlated * 1000))
        rows.append({"correlated_failure_probability": correlated, "static": static, "reconfigurable": recon})
    write_csv(os.path.join(outdir, "experiment_correlated_failures.csv"), rows)

    plt.figure(figsize=(7.2, 4.5))
    plt.plot([r["correlated_failure_probability"] for r in rows], [r["static"] for r in rows], "o-", label="Static contiguous pod")
    plt.plot([r["correlated_failure_probability"] for r in rows], [r["reconfigurable"] for r in rows], "o-", label="Reconfigurable")
    plt.xlabel("Probability of one whole-cube correlated failure")
    plt.ylabel("Probability of successful allocation")
    plt.title("Experiment 3: Correlated failures (additional scenario)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "experiment_correlated_failures.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    experiment_job_size(args.outdir)
    experiment_scale(args.outdir)
    experiment_correlated_failures(args.outdir)
    print(f"Wrote experiments, CSV files, and graphs to {args.outdir}/")


if __name__ == "__main__":
    main()
