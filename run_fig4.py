"""
run_fig4.py — Fig. 4: average sum-rate versus number of users.

This version is aligned with the corrected QRL / mMIMO implementation:
  - QRL selection receives mimo_sys and F, so the Grover oracle is actually used;
  - all schedulers select exactly n_schedule users;
  - training reward is PF-inspired, while the plotted metric is sum-rate;
  - QRL, QNN, and CNN keep separate PF reward states / advantage baselines;
  - evaluation uses the same channel realisations for all methods and averages
    over multiple geometries and fading samples.

Example:
    python run_fig4.py --epochs 50 --users 2 4 6
    python run_fig4.py --epochs 500
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


A = 32
SNR_DB = 20.0
N_EPOCHS = 500
N_LAYERS = 2
LR_QRL = 0.02
LR_QNN = 0.02
LR_CNN = 1e-3
G_ITERS = 1
USER_RANGE = [2, 4, 6, 8, 10]
BASELINE_DECAY = 0.9
EVAL_GEOMETRIES = 5
EVAL_STEPS = 30
SEED = 1234
TAU = 80.0

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


def evaluate(qrl, qnn, cnn, T: int, n_sched: int, seed: int,
             eval_geometries: int = EVAL_GEOMETRIES,
             eval_steps: int = EVAL_STEPS) -> dict[str, float]:
    """Evaluate all methods on identical channels; no PF updates during evaluation."""
    out = {"QRL": [], "QNN": [], "CNN": []}
    sys = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB, seed=seed)

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


def run(n_epochs: int = N_EPOCHS, user_range: list[int] | None = None,
        tau_scale: float = 2.0, seed: int = SEED,
        eval_geometries: int = EVAL_GEOMETRIES,
        eval_steps: int = EVAL_STEPS):
    if user_range is None:
        user_range = USER_RANGE

    np.random.seed(seed)
    print(f"Fig. 4 — Sum-rate vs users | A={A}, SNR={SNR_DB} dB, epochs={n_epochs}")
    print(f"Evaluation: {eval_geometries} geometries × {eval_steps} fading samples")

    results = {m: [] for m in ["QRL", "QNN", "CNN"]}
    t0 = time.time()

    for T in user_range:
        n_sched = max(1, T // 2)
        tau = tau_scale * T  # paper-like dynamic oracle threshold in bps/Hz
        print(f"\nT={T}, n_schedule={n_sched}, tau={tau:.2f}")

        qrl = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
        qnn = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
        cnn = CNNScheduler(A=A, T=T, lr=LR_CNN)
        sys = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB, seed=seed + 100 * T)

        states = {
            "QRL": TrainState(ProportionalFairnessScheduler(T)),
            "QNN": TrainState(ProportionalFairnessScheduler(T)),
            "CNN": TrainState(ProportionalFairnessScheduler(T)),
        }

        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        marked_history = []

        for epoch in range(1, n_epochs + 1):
            # Keep the large-scale geometry stable for a short block, then resample
            # so the policies do not overfit to a single fixed geometry.
            if epoch == 1 or epoch % 10 == 1:
                sys.reset_channel_conditions()
                F = sys.beamforming_vector()

            N = sys.generate_channel()

            theta_qrl = qrl.select(N, n_sched, mimo_sys=sys, F=F)
            rates_qrl, sr_qrl = _sum_rate(sys, N, F, theta_qrl)
            _, adv_qrl = _pf_advantage(states["QRL"], theta_qrl, rates_qrl, epoch)
            qrl.update(N, adv_qrl, theta_qrl)
            marked_history.append(qrl._last_info.get("marked_count", 0))

            theta_qnn = qnn.select(N, n_sched)
            rates_qnn, sr_qnn = _sum_rate(sys, N, F, theta_qnn)
            _, adv_qnn = _pf_advantage(states["QNN"], theta_qnn, rates_qnn, epoch)
            qnn.update(N, adv_qnn, theta_qnn)

            theta_cnn = cnn.select(N, n_sched)
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

        eval_scores = evaluate(qrl, qnn, cnn, T, n_sched, seed + 1000 * T,
                               eval_geometries=eval_geometries,
                               eval_steps=eval_steps)
        for method in results:
            results[method].append(eval_scores[method])

        print(
            f"  Eval T={T}: QRL={eval_scores['QRL']:.3f}, "
            f"QNN={eval_scores['QNN']:.3f}, CNN={eval_scores['CNN']:.3f} bps/Hz"
        )

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    return user_range, results


def plot(user_range, results):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    for method in ["CNN", "QNN", "QRL"]:
        ax.plot(user_range, results[method], color=COLORS[method], linestyle=STYLES[method],
                marker=MARKS[method], markersize=7, linewidth=2.0, label=method)

    ax.set_xlabel("Number of users (T)", fontsize=12)
    ax.set_ylabel("Average sum-rate (bps/Hz)", fontsize=12)
    ax.set_title(f"Fig. 4 — Sum-rate vs users (A={A}, SNR={SNR_DB} dB)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(user_range)
    plt.tight_layout()
    plt.savefig("fig4_users.png", dpi=150)
    print("Saved → fig4_users.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--users", type=int, nargs="+", default=USER_RANGE)
    parser.add_argument("--tau-scale", type=float, default=2.0,
                        help="Oracle threshold tau = tau_scale * T. Default: 2.0")
    parser.add_argument("--eval-geometries", type=int, default=EVAL_GEOMETRIES)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    user_range, results = run(
        n_epochs=args.epochs,
        user_range=sorted(set(args.users)),
        tau_scale=args.tau_scale,
        seed=args.seed,
        eval_geometries=args.eval_geometries,
        eval_steps=args.eval_steps,
    )
    plot(user_range, results)
