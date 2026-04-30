"""
run_fig3.py  —  Fig 3: Training Convergence Curve
==================================================
Paper: arXiv:2601.20688v1  (Section IV, Fig. 3)

Target configuration:
  A = 32 antennas, T = 6 users, SNR = 20 dB, 500 epochs
  Sum rate rises from ~22 bps/Hz (random policy) to ~32 bps/Hz (trained)

BUGS FIXED vs previous version:

  FIX 1 — n_LoS normalisation in mMIMO_sys.py (root cause of ~7 vs ~22 start).
    Old: n_LoS = exp(...) / sqrt(A)  — channel power O(1), SINR too low
    New: n_LoS = exp(...)            — channel power O(A), matches paper scale
    Impact: sum rate floor rises from ~7 to ~22 bps/Hz, matching paper Fig. 3.
    NOTE: mMIMO_sys.py must also be updated — see that file.

  FIX 2 — N_SCHED = T = 6, not T//2 = 3.
    Paper schedules all T candidate users each slot; the scheduling vector
    theta selects which users transmit (all ones = all T users active).
    Halving N_SCHED halves the sum rate floor.

  FIX 3 — F passed explicitly to agent.select() and oracle.
    Old: agent.select(N, N_SCHED, mimo_sys=env) — oracle regenerated channel
    New: F computed once, passed via select(N, N_SCHED, mimo_sys=env, F=F)
    Ensures oracle scores candidates on the SAME channel used for training.

  FIX 4 — reward = instantaneous sum rate = sum(log2(1 + SINR_t)).
    This is the direct optimisation target of Algorithm 1 (eq. 2-3).
    Old code mixed this with PF reward in some paths; now fully consistent.

  FIX 5 — TAU calibrated to instantaneous sum rate domain.
    At A=32, T=6, SNR=20dB the mean sum rate is ~22 bps/Hz.
    TAU = 20.0 marks roughly the top half of realisations as high-reward,
    giving Grover a reasonable marked set M to amplify each step.

  FIX 6 — reset_channel_conditions() called once before training, then
    channel geometry (n_LoS, L) is held fixed for all 500 epochs.
    This matches the paper's single-cell episodic setup.

Usage:
    python run_fig3.py
    python run_fig3.py --epochs 200
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL       import QRLAgent
from mMIMO_sys import MassiveMIMOSystem


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters matching paper Section IV / Fig. 3
# ─────────────────────────────────────────────────────────────────────────────
A         = 32       # BS antennas
T         = 6        # users
SNR_DB    = 20.0     # dB
N_EPOCHS  = 500      # training epochs (x-axis in Fig. 3)
N_LAYERS  = 3        # VQC depth
LR        = 0.02     # Adam learning rate
G_ITERS   = 1        # Grover iterations per step
# FIX 5: TAU in instantaneous sum rate units (~22 bps/Hz mean at these params)
TAU       = 20.0
# FIX 2: schedule all T=6 users each slot (paper eq. 8: theta in {0,1}^T)
N_SCHED   = T
# EMA smoothing for the convergence plot (matches Fig. 3 visual style)
EMA_ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def ema(values: list, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Exponential moving average for plotting the convergence curve."""
    out    = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * values[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run(n_epochs: int = N_EPOCHS):
    qnode_calls = n_epochs * 2 * N_LAYERS * T * 2
    print("=" * 62)
    print(f"Fig 3 — Training convergence")
    print(f"  A={A}, T={T}, SNR={SNR_DB}dB, epochs={n_epochs}")
    print(f"  N_SCHED={N_SCHED}, N_LAYERS={N_LAYERS}, LR={LR}")
    print(f"  G={G_ITERS}, TAU={TAU}, EMA_alpha={EMA_ALPHA}")
    print(f"  ~{qnode_calls:,} QNode calls total")
    print("=" * 62)

    # Initialise agent and environment
    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR, G=G_ITERS, tau=TAU)
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

    # FIX 6: fix channel geometry for the entire training run
    env.reset_channel_conditions()

    # Statistical beamforming matrix F is constant for fixed geometry (eq. 7)
    # FIX 3: compute F once and reuse every epoch
    F = env.beamforming_vector()

    history = []   # raw instantaneous sum rates
    t0      = time.time()

    for epoch in range(1, n_epochs + 1):

        # Sample one channel realisation (NLoS component varies, LoS fixed)
        N = env.generate_channel()

        # QRL agent selects scheduling vector via Grover-amplified policy
        # FIX 3: pass F so oracle evaluates candidates on the SAME channel
        theta = agent.select(N, N_SCHED, mimo_sys=env, F=F)

        # Compute instantaneous SINR and per-user rates (eq. 3, 6)
        sinr  = env.compute_sinr(N, F, theta)
        rates = env.instantaneous_rate(sinr)

        # FIX 4: reward = instantaneous sum rate (eq. 2), consistent throughout
        reward = float(np.sum(rates))

        # Update historical average rates for PF tracking (eq. 10)
        env.update_avg_rate(rates, theta)

        # REINFORCE update on VQC policy weights
        agent.update(N, reward, theta)

        history.append(reward)

        if epoch % 50 == 0:
            elapsed     = time.time() - t0
            eta         = elapsed / epoch * (n_epochs - epoch)
            recent_mean = np.mean(history[-50:])
            print(f"  epoch {epoch:4d}/{n_epochs}  |  "
                  f"sum_rate={reward:.2f}  mean50={recent_mean:.2f}  |  "
                  f"elapsed={elapsed:.0f}s  ETA~{eta:.0f}s")

    smooth = ema(history)
    total  = time.time() - t0
    print(f"\nDone. Final EMA sum rate = {smooth[-1]:.2f} bps/Hz  "
          f"(total {total:.0f}s)")
    return history, smooth


# ─────────────────────────────────────────────────────────────────────────────
# Plot — reproduces Fig. 3
# ─────────────────────────────────────────────────────────────────────────────

def plot(history: list, smooth: np.ndarray):
    epochs = np.arange(1, len(smooth) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    # Raw reward (light blue, thin) — mirrors the oscillating raw curve in paper
    ax.plot(epochs, history,
            color='#aac4e0', linewidth=0.6, alpha=0.55,
            label='Grover Agent Avg Reward (raw)')

    # EMA smoothed (dark blue, thick) — the main curve in paper Fig. 3
    ax.plot(epochs, smooth,
            color='#1a6fb5', linewidth=2.2,
            label=f'Smoothed (EMA α={EMA_ALPHA})')

    # Paper Fig. 3 approximate reference points
    ax.axhline(22, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(32, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(510, 22.4, '~22 bps/Hz (start)', fontsize=8, color='gray')
    ax.text(510, 32.4, '~32 bps/Hz (end)',   fontsize=8, color='gray')

    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel('Average Sum Rate (bps/Hz)', fontsize=11)
    ax.set_title(
        f'Grover QRL Agent Training — Rewards\n'
        f'A={A}, T={T}, SNR={SNR_DB} dB',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, len(smooth))
    ax.set_ylim(bottom=max(0, min(history) - 2))

    plt.tight_layout()
    out = 'fig3_training.png'
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
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