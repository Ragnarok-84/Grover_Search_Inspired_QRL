"""
run_fig3.py  —  Fig 3: Training Convergence Curve
==================================================
Paper: arXiv:2601.20688v1  (Section IV, Fig. 3)

Config : A=32, T=6, SNR=20 dB, 500 epochs
Result : sum rate rises from ~22 → ~32 bps/Hz

Runtime: ~15–20 min on CPU
  QNode calls = 500 epochs × 48 calls/epoch = 24,000

Usage:
    !python run_fig3.py
    !python run_fig3.py --epochs 100
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL       import QRLAgent
from mMIMO_sys import MassiveMIMOSystem

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
A         = 32
T         = 6
SNR_DB    = 20.0
N_EPOCHS  = 500
N_LAYERS  = 2
LR        = 0.02
G_ITERS   = 1
TAU       = 1.0 / (2 ** T)
N_SCHED   = (T + 1) // 2       # = 3 users scheduled per slot
EMA_ALPHA = 0.1


def ema(values, alpha=EMA_ALPHA):
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training  —  1 channel sample per epoch  (Algorithm 1)
#
# Why STEPS=1 (not 10):
#   Paper Algorithm 1 trains on one batch of candidate policies per epoch.
#   STEPS=10 multiplies QNode calls by 10x with no benefit — the channel
#   geometry is reset each epoch anyway so extra steps see the same geometry.
#   QNode calls per epoch : 2 × N_LAYERS × T × 2 = 2×2×6×2 = 48
#   Total for 500 epochs  : 48 × 500 = 24,000  (vs 240,000 with STEPS=10)
# ─────────────────────────────────────────────────────────────────────────────
def run(n_epochs=N_EPOCHS):
    qnode_calls = n_epochs * 2 * N_LAYERS * T * 2
    print(f"Fig 3 — Training convergence  (A={A}, T={T}, SNR={SNR_DB}dB)")
    print(f"        {n_epochs} epochs  |  ~{qnode_calls:,} QNode calls")

    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR, G=G_ITERS, tau=TAU)
    sys   = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

    history = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        sys.reset_channel_conditions()
        F     = sys.beamforming_vector()
        N     = sys.generate_channel()

        theta  = agent.select(N, N_SCHED)
        sinr   = sys.compute_sinr(N, F, theta)
        rates  = sys.instantaneous_rate(sinr)
        reward = sys.compute_pf_reward(rates, theta)
        sys.update_avg_rate(rates, theta)
        agent.update(N, reward, theta)

        history.append(float(np.sum(rates)))

        if epoch % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (n_epochs - epoch)
            print(f"  epoch {epoch:4d}/{n_epochs}  |  "
                  f"sum_rate = {history[-1]:.3f} bps/Hz  |  "
                  f"elapsed = {elapsed:.0f}s  ETA ≈ {eta:.0f}s")

    smooth = ema(history)
    print(f"\nFinal smoothed rate = {smooth[-1]:.3f} bps/Hz  "
          f"(total {time.time()-t0:.0f}s)")
    return history, smooth


def plot(history, smooth):
    epochs = np.arange(1, len(smooth) + 1)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    ax.plot(epochs, history, color='#aac4e0', linewidth=0.6, alpha=0.5,
            label='Raw sum rate')
    ax.plot(epochs, smooth,  color='#1a6fb5', linewidth=2.0,
            label=f'Smoothed (EMA α={EMA_ALPHA})')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title(f'Fig 3 — Training convergence  (A={A}, T={T}, SNR={SNR_DB}dB)',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('fig3_training.png', dpi=150)
    print("Saved → fig3_training.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    args = parser.parse_args()
    history, smooth = run(args.epochs)
    plot(history, smooth)