"""
run_fig3.py  —  Fig 3: Training Convergence Curve
==================================================
Paper: arXiv:2601.20688v1  (Section IV, Fig. 3)

Config : A=32, T=6, SNR=20 dB, 500 epochs
Target : sum rate rises from ~22 → ~32 bps/Hz

FIXES vs original:
  FIX 1 (critical): beamforming_vector now uses n_LoS directly.
    Old code averaged noisy channel samples → all users got beam 0
    → sum rate stuck at ~3-4 bps/Hz instead of ~8-10 bps/Hz.

  FIX 2: agent.select() now receives mimo_sys so QRL oracle can
    evaluate actual reward to identify high-reward states M (Algorithm 1).
    Without this, Grover amplification was bypassed entirely.

  FIX 3: TAU tuned to a meaningful reward-domain threshold.
    Old TAU = 1/64 = 0.0156 made sense as probability but not as reward.
    New TAU = 4.0 bps/Hz ≈ half of typical single-user rate.

  FIX 4: F is computed once per epoch (fixed channel geometry within epoch)
    not re-computed inside select(). This is consistent with paper assumption
    that statistical beamforming F is stable within an episode.

Usage:
    python run_fig3.py
    python run_fig3.py --epochs 100
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL  import QRLAgent
from mMIMO_sys import MassiveMIMOSystem

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (match paper Section IV)
# ─────────────────────────────────────────────────────────────────────────────
A         = 32
T         = 6
SNR_DB    = 20.0
N_EPOCHS  = 500
N_LAYERS  = 3        # increased from 2 for better QRL expressibility
LR        = 0.02
G_ITERS   = 1
TAU       = 4.0      # FIX 3: reward threshold in bps/Hz units
N_SCHED   = (T + 1) // 2   # = 3 users scheduled per slot
EMA_ALPHA = 0.1


def ema(values, alpha=EMA_ALPHA):
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


def run(n_epochs=N_EPOCHS):
    qnode_calls = n_epochs * 2 * N_LAYERS * T * 2
    print(f"Fig 3 — Training convergence  (A={A}, T={T}, SNR={SNR_DB}dB)")
    print(f"        {n_epochs} epochs  |  ~{qnode_calls:,} QNode calls")
    print(f"        N_LAYERS={N_LAYERS}, LR={LR}, TAU={TAU}, N_SCHED={N_SCHED}")

    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR, G=G_ITERS, tau=TAU)
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

    history = []
    t0 = time.time()

    env.reset_channel_conditions()

    for epoch in range(1, n_epochs + 1):
        # FIX 4: beamforming computed once per epoch from n_LoS
        F = env.beamforming_vector()   # now uses n_LoS directly — fast, correct
        N = env.generate_channel()

        # FIX 2: pass env so oracle can evaluate reward for Grover marking
        theta  = agent.select(N, N_SCHED, mimo_sys=env)

        sinr   = env.compute_sinr(N, F, theta)
        rates  = env.instantaneous_rate(sinr)
        reward = float(np.sum(rates))   # instantaneous sum rate as reward signal

        env.update_avg_rate(rates, theta)
        agent.update(N, reward, theta)

        history.append(reward)

        if epoch % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (n_epochs - epoch)
            smooth_recent = np.mean(history[-50:])
            print(f"  epoch {epoch:4d}/{n_epochs}  |  "
                  f"sum_rate = {history[-1]:.3f}  smooth50 = {smooth_recent:.3f}  |  "
                  f"elapsed={elapsed:.0f}s  ETA≈{eta:.0f}s")

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