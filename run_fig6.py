"""
run_fig6.py — Fig. 6: average sum-rate versus SNR.

Reproduces Fig. 6 from the paper:
  "Grover's Search-Inspired Quantum Reinforcement Learning for Massive MIMO
   User Scheduling" (arXiv:2601.20688v1)

Two user-antenna configurations are evaluated:
  - Config 1: T=4, A=8   (limited configuration)
  - Config 2: T=6, A=12  (better configuration)

For each configuration, three methods are compared: QRL, QNN, CNN.
The x-axis is SNR (dB), ranging from 0 to 30 dB.
The y-axis is average sum-rate (bps/Hz).

Key alignment with paper:
  - Training reward is PF-inspired; plotted metric is sum-rate.
  - All schedulers select exactly n_schedule = ceil(T/2) users.
  - QRL uses Grover oracle with the MIMO system.
  - Each SNR point is trained independently from scratch.
  - Evaluation averages over multiple geometries and fading samples.

Example:
    python run_fig6.py --epochs 300 --snr-range 0 5 10 15 20 25 30
    python run_fig6.py --epochs 500
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from QRL import QRLAgent
from benchmark import CNNScheduler, QNNScheduler
from mMIMO_sys import MassiveMIMOSystem
from baseline import ProportionalFairnessScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Default hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
SNR_RANGE      = [0, 5, 10, 15, 20, 25, 30]   # dB, matching paper x-axis
N_EPOCHS       = 500
N_LAYERS       = 2
LR_QRL         = 0.02
LR_QNN         = 0.02
LR_CNN         = 1e-3
G_ITERS        = 1
TAU            = 80.0          # percentile oracle threshold (same as fig5)
BASELINE_DECAY = 0.9
EVAL_GEOMETRIES = 5
EVAL_STEPS      = 30
SEED            = 9999

# Two configurations shown in Fig. 6 (legend: "T=4,A=8" and "T=6,A=12")
CONFIGS = [
    {"T": 4, "A": 8,  "label": "T=4, A=8"},
    {"T": 6, "A": 12, "label": "T=6, A=12"},
]

# Colour / style scheme consistent with fig4 & fig5
COLORS = {"QRL": "#1a6fb5", "QNN": "#1a8c5a", "CNN": "#c0392b"}
STYLES = {
    ("QRL", 0): "-",   ("QNN", 0): "--",  ("CNN", 0): ":",
    ("QRL", 1): "-",   ("QNN", 1): "--",  ("CNN", 1): ":",
}
MARKS  = {"QRL": "o",  "QNN": "s",  "CNN": "^"}
# Distinguish configs by marker fill
FILL   = {0: "full", 1: "none"}   # config-0 filled, config-1 open markers


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainState:
    pf: ProportionalFairnessScheduler
    baseline: float = 0.0


def _sum_rate(
    sys: MassiveMIMOSystem,
    N: np.ndarray,
    F: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, float]:
    sinr  = sys.compute_sinr(N, F, theta)
    rates = sys.instantaneous_rate(sinr)
    return rates, sys.compute_sum_rate(rates, theta)


def _pf_advantage(
    state: TrainState,
    theta: np.ndarray,
    rates: np.ndarray,
    epoch: int,
) -> tuple[float, float]:
    train_reward   = state.pf.calculate_pf_reward(theta, rates)
    state.baseline = (
        train_reward
        if epoch == 1
        else BASELINE_DECAY * state.baseline + (1 - BASELINE_DECAY) * train_reward
    )
    advantage = train_reward - state.baseline
    state.pf.update(theta, rates)
    return train_reward, advantage


def evaluate(
    qrl, qnn, cnn,
    A: int, T: int, n_sched: int, snr_db: float, seed: int,
    eval_geometries: int = EVAL_GEOMETRIES,
    eval_steps: int = EVAL_STEPS,
) -> dict[str, float]:
    """Evaluate all three methods on identical channels (no PF updates)."""
    out = {"QRL": [], "QNN": [], "CNN": []}
    sys = MassiveMIMOSystem(A=A, T=T, snr_db=snr_db, seed=seed)

    for _ in range(eval_geometries):
        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        for _ in range(eval_steps):
            N = sys.generate_channel()

            theta_qrl = qrl.select(N, n_sched, mimo_sys=sys, F=F)
            _, sr_qrl = _sum_rate(sys, N, F, theta_qrl)
            out["QRL"].append(sr_qrl)

            theta_qnn = qnn.select(N, n_sched)
            _, sr_qnn = _sum_rate(sys, N, F, theta_qnn)
            out["QNN"].append(sr_qnn)

            theta_cnn = cnn.select(N, n_sched)
            _, sr_cnn = _sum_rate(sys, N, F, theta_cnn)
            out["CNN"].append(sr_cnn)

    return {k: float(np.mean(v)) for k, v in out.items()}


def train_one_point(
    A: int, T: int, n_sched: int, snr_db: float,
    n_epochs: int, tau: float, seed: int,
    eval_geometries: int = EVAL_GEOMETRIES,
    eval_steps: int = EVAL_STEPS,
) -> dict[str, float]:
    """
    Train QRL / QNN / CNN from scratch for one (A, T, SNR) operating point,
    then evaluate and return average sum-rates.
    """
    qrl = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
    qnn = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
    cnn = CNNScheduler(A=A, T=T, lr=LR_CNN)
    sys = MassiveMIMOSystem(A=A, T=T, snr_db=snr_db, seed=seed)

    states = {
        "QRL": TrainState(ProportionalFairnessScheduler(T)),
        "QNN": TrainState(ProportionalFairnessScheduler(T)),
        "CNN": TrainState(ProportionalFairnessScheduler(T)),
    }

    sys.reset_channel_conditions()
    F = sys.beamforming_vector()

    for epoch in range(1, n_epochs + 1):
        # Refresh large-scale geometry periodically to prevent overfitting
        if epoch == 1 or epoch % 10 == 1:
            sys.reset_channel_conditions()
            F = sys.beamforming_vector()

        N = sys.generate_channel()

        # --- QRL ---
        theta_qrl          = qrl.select(N, n_sched, mimo_sys=sys, F=F)
        rates_qrl, sr_qrl  = _sum_rate(sys, N, F, theta_qrl)
        _, adv_qrl         = _pf_advantage(states["QRL"], theta_qrl, rates_qrl, epoch)
        qrl.update(N, adv_qrl, theta_qrl)

        # --- QNN ---
        theta_qnn          = qnn.select(N, n_sched)
        rates_qnn, sr_qnn  = _sum_rate(sys, N, F, theta_qnn)
        _, adv_qnn         = _pf_advantage(states["QNN"], theta_qnn, rates_qnn, epoch)
        qnn.update(N, adv_qnn, theta_qnn)

        # --- CNN ---
        theta_cnn          = cnn.select(N, n_sched)
        rates_cnn, sr_cnn  = _sum_rate(sys, N, F, theta_cnn)
        _, adv_cnn         = _pf_advantage(states["CNN"], theta_cnn, rates_cnn, epoch)
        cnn.update(N, adv_cnn, theta_cnn)

    # Evaluation on fresh channels after training
    return evaluate(
        qrl, qnn, cnn,
        A=A, T=T, n_sched=n_sched, snr_db=snr_db,
        seed=seed + 100_000,
        eval_geometries=eval_geometries,
        eval_steps=eval_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_epochs: int = N_EPOCHS,
    snr_range: list[int] | None = None,
    tau: float = TAU,
    seed: int = SEED,
    eval_geometries: int = EVAL_GEOMETRIES,
    eval_steps: int = EVAL_STEPS,
    configs: list[dict] | None = None,
):
    if snr_range  is None: snr_range = SNR_RANGE
    if configs    is None: configs   = CONFIGS

    np.random.seed(seed)
    print("=" * 70)
    print("Fig. 6 — Sum-rate vs SNR")
    print(f"  SNR range: {snr_range} dB | epochs per point: {n_epochs}")
    print(f"  Configs: {[(c['T'], c['A']) for c in configs]}")
    print(f"  tau={tau:.1f} (percentile), eval={eval_geometries}×{eval_steps}")
    print("=" * 70)

    # results[cfg_idx][method] = list of avg sum-rates (one per SNR point)
    results: list[dict[str, list[float]]] = [
        {"QRL": [], "QNN": [], "CNN": []} for _ in configs
    ]

    t0 = time.time()

    for cfg_idx, cfg in enumerate(configs):
        A_cfg, T_cfg = cfg["A"], cfg["T"]
        n_sched = (T_cfg + 1) // 2
        print(f"\n{'─'*60}")
        print(f"Config {cfg_idx+1}/{len(configs)}: T={T_cfg}, A={A_cfg}, n_sched={n_sched}")

        for snr_idx, snr_db in enumerate(snr_range):
            point_seed = seed + cfg_idx * 10_000 + snr_idx * 100
            print(f"  SNR={snr_db:5.1f} dB  (seed={point_seed})", end="", flush=True)
            t1 = time.time()

            scores = train_one_point(
                A=A_cfg, T=T_cfg, n_sched=n_sched,
                snr_db=float(snr_db),
                n_epochs=n_epochs, tau=tau,
                seed=point_seed,
                eval_geometries=eval_geometries,
                eval_steps=eval_steps,
            )

            for method in ("QRL", "QNN", "CNN"):
                results[cfg_idx][method].append(scores[method])

            print(
                f"  → QRL={scores['QRL']:.3f}  QNN={scores['QNN']:.3f}"
                f"  CNN={scores['CNN']:.3f}  [{time.time()-t1:.0f}s]"
            )

    total = (time.time() - t0) / 60
    print(f"\nAll done. Total time: {total:.1f} min")
    return snr_range, results, configs


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot(
    snr_range: list[int],
    results: list[dict[str, list[float]]],
    configs: list[dict],
    output_path: str = "fig6_snr.png",
):
    """
    Reproduce Fig. 6 style:
      - Two configurations, each with three curves (QRL / QNN / CNN).
      - Config 1 (limited):  solid line markers filled.
      - Config 2 (better):   dashed line markers open.
      - Legend shows method name + config label.
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")

    line_styles = ["-", "--"]   # solid for config 0, dashed for config 1
    markerfill  = [True, False] # filled for config 0, open for config 1

    for cfg_idx, (cfg, res) in enumerate(zip(configs, results)):
        ls = line_styles[cfg_idx]
        for method in ["CNN", "QNN", "QRL"]:
            y    = res[method]
            mfc  = COLORS[method] if markerfill[cfg_idx] else "none"
            ax.plot(
                snr_range, y,
                color     = COLORS[method],
                linestyle = ls,
                marker    = MARKS[method],
                markersize= 7,
                linewidth = 1.8,
                markerfacecolor = mfc,
                markeredgecolor = COLORS[method],
                label     = f"{method} ({cfg['label']})",
            )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Average Sum-Rate (bps/Hz)", fontsize=12)
    ax.set_title(
        "Fig. 6 — Sum-rate vs SNR for QNN, CNN, and QRL\n"
        "under different user–antenna configurations",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(snr_range)
    ax.set_xlim(min(snr_range) - 1, max(snr_range) + 1)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Fig. 6-style SNR vs sum-rate experiment"
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help="Training epochs per (config, SNR) point (default: 500)"
    )
    parser.add_argument(
        "--snr-range", type=int, nargs="+", default=SNR_RANGE,
        metavar="SNR",
        help="SNR values in dB (default: 0 5 10 15 20 25 30)"
    )
    parser.add_argument(
        "--tau", type=float, default=TAU,
        help="Oracle percentile threshold for QRL (default: 80.0)"
    )
    parser.add_argument(
        "--eval-geometries", type=int, default=EVAL_GEOMETRIES,
        help="Number of geometry realisations for evaluation (default: 5)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=EVAL_STEPS,
        help="Fading samples per geometry for evaluation (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Global random seed (default: 9999)"
    )
    parser.add_argument(
        "--output", type=str, default="fig6_snr.png",
        help="Output filename for the figure (default: fig6_snr.png)"
    )
    args = parser.parse_args()

    snr_range, results, configs = run(
        n_epochs        = args.epochs,
        snr_range       = sorted(set(args.snr_range)),
        tau             = args.tau,
        seed            = args.seed,
        eval_geometries = args.eval_geometries,
        eval_steps      = args.eval_steps,
    )
    plot(snr_range, results, configs, output_path=args.output)