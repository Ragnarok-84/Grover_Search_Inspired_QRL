import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL       import QRLAgent
from mMIMO_sys import MassiveMIMOSystem


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
A         = 32       # BS antennas
T         = 6        # candidate users
SNR_DB    = 20.0     # dB
N_EPOCHS  = 500
N_LAYERS  = 3        # VQC depth
LR        = 0.02     # Adam learning rate
G_ITERS   = 1        # Grover iterations per step
TAU       = 20.0     # oracle threshold (bps/Hz); marks ~top-50% of realisations
# Schedule T//2 users per slot so the scheduling choice is non-trivial
N_SCHED   = T // 2   # = 3
# REINFORCE advantage baseline decay
BASELINE_DECAY = 0.9
# EMA for the convergence plot
EMA_ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ema(values: list, alpha: float = EMA_ALPHA) -> np.ndarray:
    out    = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run(n_epochs: int = N_EPOCHS):
    qnode_calls_per_epoch = 2 * N_LAYERS * T * 2   # parameter-shift
    print("=" * 62)
    print(f"Fig 3 — Training convergence")
    print(f"  A={A}, T={T}, SNR={SNR_DB}dB, epochs={n_epochs}")
    print(f"  N_SCHED={N_SCHED}, N_LAYERS={N_LAYERS}, LR={LR}")
    print(f"  G={G_ITERS}, TAU={TAU}, EMA_alpha={EMA_ALPHA}")
    print(f"  ~{qnode_calls_per_epoch * n_epochs:,} QNode calls total")
    print("=" * 62)

    # ── Initialise agent and environment ──────────────────────────────────
    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR, G=G_ITERS, tau=TAU)
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

    # FIX 3: fix geometry ONCE — do not reset inside the loop
    env.reset_channel_conditions()

    # FIX 4: F is constant for fixed geometry (deterministic DFT beamforming)
    F = env.beamforming_vector()

    # FIX 5: REINFORCE advantage baseline (EMA)
    baseline = 0.0

    history = []
    t0      = time.time()

    for epoch in range(1, n_epochs + 1):

        # Sample one channel realisation (NLoS varies; LoS geometry frozen)
        N = env.generate_channel()

        # QRL selects N_SCHED users via Grover-amplified stochastic policy
        theta = agent.select(N, N_SCHED, mimo_sys=env, F=F)

        # Compute instantaneous SINR and per-user rates (eq. 3, 6)
        sinr  = env.compute_sinr(N, F, theta)
        rates = env.instantaneous_rate(sinr)

        # Reward = instantaneous sum rate (eq. 2)
        reward = float(np.sum(rates))

        # FIX 5: advantage baseline — centre the gradient signal
        baseline   = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward
        advantage  = reward - baseline

        # Update PF historical average rates (eq. 10)
        env.update_avg_rate(rates, theta)

        # REINFORCE update with advantage (variance-reduced)
        agent.update(N, advantage, theta)

        history.append(reward)

        if epoch % 50 == 0:
            elapsed     = time.time() - t0
            eta         = elapsed / epoch * (n_epochs - epoch)
            recent_mean = np.mean(history[-50:])
            print(f"  epoch {epoch:4d}/{n_epochs}  |  "
                  f"sum_rate={reward:.2f}  mean50={recent_mean:.2f}  "
                  f"baseline={baseline:.2f}  |  "
                  f"elapsed={elapsed:.0f}s  ETA~{eta:.0f}s")

    smooth = ema(history)
    total  = time.time() - t0
    print(f"\nDone.  Final EMA sum rate = {smooth[-1]:.2f} bps/Hz  "
          f"(total {total:.0f}s)")
    return history, smooth


# ─────────────────────────────────────────────────────────────────────────────
# Plot — reproduces Fig. 3
# ─────────────────────────────────────────────────────────────────────────────

def plot(history: list, smooth: np.ndarray):
    epochs = np.arange(1, len(smooth) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    ax.plot(epochs, history,
            color='#aac4e0', linewidth=0.6, alpha=0.55,
            label='Grover QRL Avg Reward (raw)')

    ax.plot(epochs, smooth,
            color='#1a6fb5', linewidth=2.2,
            label=f'Smoothed (EMA α={EMA_ALPHA})')

    ax.axhline(22, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(32, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(len(smooth) * 1.01, 22.4, '~22 bps/Hz (start)',
            fontsize=8, color='gray')
    ax.text(len(smooth) * 1.01, 32.4, '~32 bps/Hz (end)',
            fontsize=8, color='gray')

    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=11)
    ax.set_title(
        f'Grover QRL Agent — Training Convergence\n'
        f'A={A}, T={T}, SNR={SNR_DB} dB, N_SCHED={N_SCHED}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, len(smooth))
    ax.set_ylim(bottom=max(0, min(history) - 2))

    plt.tight_layout()
    out = 'fig3_training.png'
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce Fig. 3 of arXiv:2601.20688v1")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Training epochs (default {N_EPOCHS})")
    args = parser.parse_args()

    history, smooth = run(args.epochs)
    plot(history, smooth)