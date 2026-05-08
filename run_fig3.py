import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL import QRLAgent
from mMIMO_sys import MassiveMIMOSystem


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
A = 32
T = 6
SNR_DB = 20.0
N_EPOCHS = 500
N_LAYERS = 3
LR = 0.02
G_ITERS = 1
TAU = 20.0
N_SCHED = T // 2
BASELINE_DECAY = 0.9
EMA_ALPHA = 0.05
SEED = 7
TRAIN_REWARD = "pf"  # "pf" follows Eq. (10); "sum_rate" follows pure throughput


def ema(values: list[float], alpha: float = EMA_ALPHA) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


def run(n_epochs: int = N_EPOCHS,
        seed: int = SEED,
        train_reward: str = TRAIN_REWARD):
    np.random.seed(seed)
    qnode_calls_per_epoch = 2 * N_LAYERS * T * 2
    print("=" * 70)
    print("Fig. 3-style training convergence")
    print(f"  A={A}, T={T}, SNR={SNR_DB} dB, epochs={n_epochs}, seed={seed}")
    print(f"  n_schedule={N_SCHED}, layers={N_LAYERS}, lr={LR}, G={G_ITERS}, tau={TAU}")
    print(f"  training reward={train_reward}; plotted metric=sum-rate")
    print(f"  ~{qnode_calls_per_epoch * n_epochs:,} VQC QNode calls for parameter-shift")
    print("=" * 70)

    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR, G=G_ITERS, tau=TAU)
    env = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB, seed=seed)
    F = env.beamforming_vector()

    baseline = 0.0
    sum_rate_history: list[float] = []
    train_reward_history: list[float] = []
    marked_history: list[int] = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        N = env.generate_channel()
        theta = agent.select(N, N_SCHED, mimo_sys=env, F=F)
        sinr, rates, sum_rate, pf_reward = env.evaluate_theta(N, F, theta)

        if train_reward == "pf":
            reward = pf_reward
        elif train_reward == "sum_rate":
            reward = sum_rate
        else:
            raise ValueError("train_reward must be 'pf' or 'sum_rate'")

        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward
        advantage = reward - baseline
        agent.update(N, advantage, theta)
        env.update_avg_rate(rates, theta)

        sum_rate_history.append(sum_rate)
        train_reward_history.append(float(reward))
        marked_history.append(agent._last_info.get("marked_count", 0))

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (n_epochs - epoch)
            print(f"  epoch {epoch:4d}/{n_epochs} | "
                  f"sum_rate={sum_rate:7.3f} | "
                  f"mean50={np.mean(sum_rate_history[-50:]):7.3f} | "
                  f"reward={reward:8.3f} | marked={marked_history[-1]:3d} | "
                  f"elapsed={elapsed:5.0f}s ETA~{eta:5.0f}s")

    smooth = ema(sum_rate_history)
    total = time.time() - t0
    print(f"\nDone. Final EMA sum-rate = {smooth[-1]:.3f} bps/Hz (total {total:.0f}s)")
    return {
        "sum_rate": sum_rate_history,
        "smooth_sum_rate": smooth,
        "train_reward": train_reward_history,
        "marked_count": marked_history,
    }


def plot(results: dict, output_path: str = "fig3_training.png"):
    history = results["sum_rate"]
    smooth = results["smooth_sum_rate"]
    epochs = np.arange(1, len(smooth) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(epochs, history, linewidth=0.6, alpha=0.55, label="Raw sum-rate")
    ax.plot(epochs, smooth, linewidth=2.2, label=f"EMA sum-rate α={EMA_ALPHA}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title(f"Grover-inspired QRL training\nA={A}, T={T}, SNR={SNR_DB} dB, n_schedule={N_SCHED}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xlim(1, len(smooth))
    ax.set_ylim(bottom=max(0, min(history) - 2))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved → {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fig. 3-style QRL convergence experiment")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-reward", choices=["pf", "sum_rate"], default=TRAIN_REWARD)
    parser.add_argument("--output", type=str, default="fig3_training.png")
    args = parser.parse_args()

    results = run(args.epochs, seed=args.seed, train_reward=args.train_reward)
    plot(results, output_path=args.output)
