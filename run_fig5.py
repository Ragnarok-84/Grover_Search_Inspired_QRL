"""
run_fig5.py — Fig. 5: average sum-rate versus number of BS antennas.

This version is aligned with the corrected QRL / mMIMO implementation:
  - QRL selection receives mimo_sys and F, so the Grover oracle is actually used;
  - all schedulers select exactly N_SCHED users;
  - training reward is PF-inspired, while the plotted metric is sum-rate;
  - QRL, QNN, and CNN keep separate PF reward states / advantage baselines;
  - evaluation uses the same channel realisations for all methods.

Example:
    python run_fig5.py --epochs 50 --antennas 6 8 10
    python run_fig5.py --epochs 500
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from QRL import QRLAgent
from benchmark import CNNScheduler, QNNScheduler
from mMIMO_sys import MassiveMIMOSystem
from baseline import ProportionalFairnessScheduler


T = 6
SNR_DB = 20.0
N_EPOCHS = 500
N_LAYERS = 2
LR_QRL = 0.02
LR_QNN = 0.02
LR_CNN = 1e-3
G_ITERS = 1
#TAU = 8.0
TAU = 80.0
N_SCHED = (T + 1) // 2
ANT_RANGE = [6, 8, 10, 12, 14, 16]
BASELINE_DECAY = 0.9
EVAL_GEOMETRIES = 5
EVAL_STEPS = 30
SEED = 5678

COLORS = {"QRL": "#1a6fb5", "QNN": "#1a8c5a", "CNN": "#c0392b"}
STYLES = {"QRL": "-", "QNN": "--", "CNN": ":"}
MARKS = {"QRL": "o", "QNN": "s", "CNN": "^"}


@dataclass
class TrainState:
    pf: ProportionalFairnessScheduler
    baseline: float = 0.0


def _sum_rate(sys: MassiveMIMOSystem, N: np.ndarray, F: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, float]:
    sinr = sys.compute_sinr(N, F, theta)
    rates = sys.instantaneous_rate(sinr)
    return rates, sys.compute_sum_rate(rates, theta)


def _pf_advantage(state: TrainState, theta: np.ndarray, rates: np.ndarray, epoch: int) -> tuple[float, float]:
    train_reward = state.pf.calculate_pf_reward(theta, rates)
    state.baseline = train_reward if epoch == 1 else BASELINE_DECAY * state.baseline + (1 - BASELINE_DECAY) * train_reward
    advantage = train_reward - state.baseline
    state.pf.update(theta, rates)
    return train_reward, advantage


def evaluate(qrl, qnn, cnn, A: int, seed: int,
             eval_geometries: int = EVAL_GEOMETRIES,
             eval_steps: int = EVAL_STEPS) -> dict[str, float]:
    out = {"QRL": [], "QNN": [], "CNN": []}
    sys = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB, seed=seed)

    for _ in range(eval_geometries):
        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        for _ in range(eval_steps):
            N = sys.generate_channel()

            theta_qrl = qrl.select(N, N_SCHED, mimo_sys=sys, F=F)
            _, sr_qrl = _sum_rate(sys, N, F, theta_qrl)
            out["QRL"].append(sr_qrl)

            theta_qnn = qnn.select(N, N_SCHED)
            _, sr_qnn = _sum_rate(sys, N, F, theta_qnn)
            out["QNN"].append(sr_qnn)

            theta_cnn = cnn.select(N, N_SCHED)
            _, sr_cnn = _sum_rate(sys, N, F, theta_cnn)
            out["CNN"].append(sr_cnn)

    return {k: float(np.mean(v)) for k, v in out.items()}


def run(n_epochs: int = N_EPOCHS, ant_range: list[int] | None = None,
        tau: float = TAU, seed: int = SEED,
        eval_geometries: int = EVAL_GEOMETRIES,
        eval_steps: int = EVAL_STEPS):
    if ant_range is None:
        ant_range = ANT_RANGE

    np.random.seed(seed)
    print(f"Fig. 5 — Sum-rate vs antennas | T={T}, SNR={SNR_DB} dB, epochs={n_epochs}")
    print(f"N_SCHED={N_SCHED}, tau={tau:.2f}, evaluation={eval_geometries}×{eval_steps}")

    results = {m: [] for m in ["QRL", "QNN", "CNN"]}
    t0 = time.time()

    for A in ant_range:
        if A < T:
            raise ValueError(f"A={A} must be >= T={T} for this experiment")

        print(f"\nA={A} | approx QNode calls: {n_epochs * 2 * 2 * N_LAYERS * T * 2:,}")
        qrl = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
        qnn = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
        cnn = CNNScheduler(A=A, T=T, lr=LR_CNN)
        sys = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB, seed=seed + 100 * A)

        states = {
            "QRL": TrainState(ProportionalFairnessScheduler(T)),
            "QNN": TrainState(ProportionalFairnessScheduler(T)),
            "CNN": TrainState(ProportionalFairnessScheduler(T)),
        }

        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        marked_history = []

        for epoch in range(1, n_epochs + 1):
            # Fig. 5 tests antenna scalability. Keep one geometry for blocks but
            # still refresh occasionally to reduce overfitting.
            if epoch == 1 or epoch % 10 == 1:
                sys.reset_channel_conditions()
                F = sys.beamforming_vector()

            N = sys.generate_channel()

            theta_qrl = qrl.select(N, N_SCHED, mimo_sys=sys, F=F)
            rates_qrl, sr_qrl = _sum_rate(sys, N, F, theta_qrl)
            _, adv_qrl = _pf_advantage(states["QRL"], theta_qrl, rates_qrl, epoch)
            qrl.update(N, adv_qrl, theta_qrl)
            marked_history.append(qrl._last_info.get("marked_count", 0))

            theta_qnn = qnn.select(N, N_SCHED)
            rates_qnn, sr_qnn = _sum_rate(sys, N, F, theta_qnn)
            _, adv_qnn = _pf_advantage(states["QNN"], theta_qnn, rates_qnn, epoch)
            qnn.update(N, adv_qnn, theta_qnn)

            theta_cnn = cnn.select(N, N_SCHED)
            rates_cnn, sr_cnn = _sum_rate(sys, N, F, theta_cnn)
            _, adv_cnn = _pf_advantage(states["CNN"], theta_cnn, rates_cnn, epoch)
            cnn.update(N, adv_cnn, theta_cnn)

            if epoch % 100 == 0:
                print(
                    f"  epoch {epoch:4d}/{n_epochs} | "
                    f"sum-rate QRL/QNN/CNN = {sr_qrl:.2f}/{sr_qnn:.2f}/{sr_cnn:.2f} | "
                    f"marked(avg100)={np.mean(marked_history[-100:]):.1f} | "
                    f"elapsed={time.time() - t0:.0f}s"
                )

        eval_scores = evaluate(qrl, qnn, cnn, A, seed + 1000 * A,
                               eval_geometries=eval_geometries,
                               eval_steps=eval_steps)
        for method in results:
            results[method].append(eval_scores[method])

        print(
            f"  Eval A={A}: QRL={eval_scores['QRL']:.3f}, "
            f"QNN={eval_scores['QNN']:.3f}, CNN={eval_scores['CNN']:.3f} bps/Hz"
        )

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    return ant_range, results


def plot(ant_range, results):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    for method in ["CNN", "QNN", "QRL"]:
        ax.plot(ant_range, results[method], color=COLORS[method], linestyle=STYLES[method],
                marker=MARKS[method], markersize=6, linewidth=1.8, label=method)

    ax.set_xlabel("Number of BS antennas (A)", fontsize=11)
    ax.set_ylabel("Average sum-rate (bps/Hz)", fontsize=11)
    ax.set_title(f"Fig. 5 — Sum-rate vs antennas (T={T}, SNR={SNR_DB} dB)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xticks(ant_range)
    plt.tight_layout()
    plt.savefig("fig5_antennas.png", dpi=150)
    print("Saved → fig5_antennas.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--antennas", type=int, nargs="+", default=ANT_RANGE)
    parser.add_argument("--tau", type=float, default=TAU, help="Oracle sum-rate threshold in bps/Hz")
    parser.add_argument("--eval-geometries", type=int, default=EVAL_GEOMETRIES)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    ant_range, results = run(
        n_epochs=args.epochs,
        ant_range=sorted(set(args.antennas)),
        tau=args.tau,
        seed=args.seed,
        eval_geometries=args.eval_geometries,
        eval_steps=args.eval_steps,
    )
    plot(ant_range, results)
