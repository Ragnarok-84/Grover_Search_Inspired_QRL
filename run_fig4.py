"""
run_fig4.py  —  Fig 4: Sum Rate vs Number of Users
====================================================
Paper: arXiv:2601.20688v1  (Section IV, Fig. 4)

Config : A=32 fixed, T ∈ {2,4,6,8,10}, SNR=20 dB
Result : QRL ~20, QNN ~17.2, CNN ~15.8 bps/Hz at T=10

Runtime: ~20–30 min on CPU
  QNode calls per T point ≈ 500 epochs × (48 QRL + 48 QNN) ≈ 48,000
  Total 5 T values ≈ 240,000 QNode calls

Usage:
    !python run_fig4.py
    !python run_fig4.py --epochs 200
    !python run_fig4.py --users 2 4 6
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL       import QRLAgent
from benchmark import CNNScheduler, QNNScheduler
from mMIMO_sys import MassiveMIMOSystem

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
A          = 32
SNR_DB     = 20.0
N_EPOCHS   = 500
N_LAYERS   = 2
LR_QRL     = 0.02
LR_QNN     = 0.02
LR_CNN     = 1e-3
G_ITERS    = 1
USER_RANGE = [2, 4, 6, 8, 10]

N_EVAL_EP  = 5
EVAL_STEPS = 50

COLORS = {'QRL': '#1a6fb5', 'QNN': '#1a8c5a', 'CNN': '#c0392b'}
STYLES = {'QRL': '-',       'QNN': '--',       'CNN': ':'}
MARKS  = {'QRL': 'o',       'QNN': 's',        'CNN': '^'}


def n_sched(T):
    return max(1, (T + 1) // 2)


def evaluate(agent, sys, n_sc):
    """No-update eval over N_EVAL_EP episodes × EVAL_STEPS steps."""
    rates_all = []
    for _ in range(N_EVAL_EP):
        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        for _ in range(EVAL_STEPS):
            N     = sys.generate_channel()
            theta = agent.select(N, n_sc)
            sinr  = sys.compute_sinr(N, F, theta)
            rates = sys.instantaneous_rate(sinr)
            rates_all.append(float(np.sum(rates)))
    return float(np.mean(rates_all))


# ─────────────────────────────────────────────────────────────────────────────
# Training  —  all 3 agents see the SAME (N, F) per epoch for fair comparison
#
# QNode calls per epoch per T:
#   QRL update : 2 × N_LAYERS × T × 2
#   QNN update : 2 × N_LAYERS × T × 2
#   CNN update : classical only (negligible)
# ─────────────────────────────────────────────────────────────────────────────
def run(n_epochs=N_EPOCHS, user_range=USER_RANGE):
    print(f"Fig 4 — Sum rate vs users  (A={A}, SNR={SNR_DB}dB, {n_epochs} epochs)")

    results = {m: [] for m in ['QRL', 'QNN', 'CNN']}
    t0 = time.time()

    for T in user_range:
        n_sc = n_sched(T)
        tau  = 1.0 / (2 ** T)
        qrl  = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
        qnn  = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
        cnn  = CNNScheduler(A=A, T=T, lr=LR_CNN)
        sys  = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

        print(f"\n  T={T}  (n_sched={n_sc})  "
              f"~{n_epochs * 2 * 2 * N_LAYERS * T * 2:,} QNode calls")

        for epoch in range(1, n_epochs + 1):
            # All agents train on the same channel realisation — fair comparison
            sys.reset_channel_conditions()
            F = sys.beamforming_vector()
            N = sys.generate_channel()

            # ── QRL ──────────────────────────────────────────────────────
            theta_qrl  = qrl.select(N, n_sc)
            sinr_qrl   = sys.compute_sinr(N, F, theta_qrl)
            rates_qrl  = sys.instantaneous_rate(sinr_qrl)
            reward_qrl = sys.compute_pf_reward(rates_qrl, theta_qrl)
            sys.update_avg_rate(rates_qrl, theta_qrl)
            qrl.update(N, reward_qrl, theta_qrl)

            # ── QNN ──────────────────────────────────────────────────────
            theta_qnn  = qnn.select(N, n_sc)
            sinr_qnn   = sys.compute_sinr(N, F, theta_qnn)
            rates_qnn  = sys.instantaneous_rate(sinr_qnn)
            reward_qnn = sys.compute_pf_reward(rates_qnn, theta_qnn)
            qnn.update(N, reward_qnn, theta_qnn)

            # ── CNN ──────────────────────────────────────────────────────
            theta_cnn  = cnn.select(N, n_sc)
            sinr_cnn   = sys.compute_sinr(N, F, theta_cnn)
            rates_cnn  = sys.instantaneous_rate(sinr_cnn)
            reward_cnn = sys.compute_pf_reward(rates_cnn, theta_cnn)
            cnn.update(N, reward_cnn, theta_cnn)

            if epoch % 100 == 0:
                print(f"    epoch {epoch}/{n_epochs}  "
                      f"({time.time()-t0:.0f}s elapsed)")

        r_qrl = evaluate(qrl, sys, n_sc)
        r_qnn = evaluate(qnn, sys, n_sc)
        r_cnn = evaluate(cnn, sys, n_sc)

        results['QRL'].append(r_qrl)
        results['QNN'].append(r_qnn)
        results['CNN'].append(r_cnn)
        print(f"  T={T}  →  QRL={r_qrl:.3f}  QNN={r_qnn:.3f}  CNN={r_cnn:.3f}  bps/Hz")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    return user_range, results


def plot(user_range, results):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    for m in ['QRL', 'QNN', 'CNN']:
        ax.plot(user_range, results[m],
                color=COLORS[m], linestyle=STYLES[m],
                marker=MARKS[m], markersize=6, linewidth=1.8, label=m)
    ax.set_xlabel('Number of users (T)', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title(f'Fig 4 — Sum rate vs users  (A={A}, SNR={SNR_DB}dB)',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xticks(user_range)
    plt.tight_layout()
    plt.savefig('fig4_users.png', dpi=150)
    print("Saved → fig4_users.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--users",  type=int, nargs="+", default=USER_RANGE)
    args = parser.parse_args()
    user_range, results = run(args.epochs, sorted(set(args.users)))
    plot(user_range, results)