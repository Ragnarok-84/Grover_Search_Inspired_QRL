"""
run_fig4.py  —  Fig 4: Sum Rate vs Number of Users
"""
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from QRL       import QRLAgent
from benchmark import CNNScheduler, QNNScheduler
from mMIMO_sys import MassiveMIMOSystem

A          = 32
SNR_DB     = 20.0
N_EPOCHS   = 500
N_LAYERS   = 2
LR_QRL     = 0.02
LR_QNN     = 0.02
LR_CNN     = 1e-3
G_ITERS    = 1
USER_RANGE = [2, 4, 6, 8, 10]

COLORS = {'QRL': '#1a6fb5', 'QNN': '#1a8c5a', 'CNN': '#c0392b'}
STYLES = {'QRL': '-',       'QNN': '--',       'CNN': ':'}
MARKS  = {'QRL': 'o',       'QNN': 's',        'CNN': '^'}

def evaluate(agent, sys, n_sc, is_qrl=False):
    """Test trên 5 môi trường không gian ngẫu nhiên để lấy kết quả trung bình chuẩn xác."""
    rates_all = []
    for _ in range(5): 
        sys.reset_channel_conditions()
        F = sys.beamforming_vector()
        for _ in range(10): # 10 fading samples per geometry
            N = sys.generate_channel()
            if is_qrl:
                theta = agent.select(N, n_sc, mimo_sys=sys, F=F)
            else:
                theta = agent.select(N, n_sc)
            sinr = sys.compute_sinr(N, F, theta)
            rates_all.append(float(np.sum(sys.instantaneous_rate(sinr))))
    return float(np.mean(rates_all))

def run(n_epochs=N_EPOCHS, user_range=USER_RANGE):
    print(f"Fig 4 — Sum rate vs users (A={A}, SNR={SNR_DB}dB, {n_epochs} epochs)")
    results = {m: [] for m in ['QRL', 'QNN', 'CNN']}
    t0 = time.time()

    for T in user_range:
        n_sc = max(1, T // 2)
        tau  = 2.0 * T  # Ngưỡng động, nhưng Oracle đã có Elitist marking bảo kê
        
        qrl  = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
        qnn  = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
        cnn  = CNNScheduler(A=A, T=T, lr=LR_CNN)
        sys  = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

        base_qrl, base_qnn, base_cnn = 0.0, 0.0, 0.0
        
        for epoch in range(1, n_epochs + 1):
            # Đổi môi trường mỗi 10 epoch để Model học tính tổng quát
            if epoch % 10 == 1:
                sys.reset_channel_conditions()
                F = sys.beamforming_vector()

            N = sys.generate_channel()

            # QRL
            theta_qrl = qrl.select(N, n_sc, mimo_sys=sys, F=F)
            r_qrl = float(np.sum(sys.instantaneous_rate(sys.compute_sinr(N, F, theta_qrl))))
            base_qrl = r_qrl if epoch == 1 else 0.9 * base_qrl + 0.1 * r_qrl
            qrl.update(N, r_qrl - base_qrl, theta_qrl)

            # QNN
            theta_qnn = qnn.select(N, n_sc)
            r_qnn = float(np.sum(sys.instantaneous_rate(sys.compute_sinr(N, F, theta_qnn))))
            base_qnn = r_qnn if epoch == 1 else 0.9 * base_qnn + 0.1 * r_qnn
            qnn.update(N, r_qnn - base_qnn, theta_qnn)

            # CNN
            theta_cnn = cnn.select(N, n_sc)
            r_cnn = float(np.sum(sys.instantaneous_rate(sys.compute_sinr(N, F, theta_cnn))))
            base_cnn = r_cnn if epoch == 1 else 0.9 * base_cnn + 0.1 * r_cnn
            cnn.update(N, r_cnn - base_cnn, theta_cnn)

            if epoch % 100 == 0:
                print(f"    epoch {epoch}/{n_epochs} ({time.time()-t0:.0f}s elapsed)")

        # Evaluate
        r_qrl_eval = evaluate(qrl, sys, n_sc, is_qrl=True)
        r_qnn_eval = evaluate(qnn, sys, n_sc, is_qrl=False)
        r_cnn_eval = evaluate(cnn, sys, n_sc, is_qrl=False)

        results['QRL'].append(r_qrl_eval)
        results['QNN'].append(r_qnn_eval)
        results['CNN'].append(r_cnn_eval)
        print(f"  T={T} -> QRL={r_qrl_eval:.2f}  QNN={r_qnn_eval:.2f}  CNN={r_cnn_eval:.2f} bps/Hz")

    return user_range, results

def plot(user_range, results):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    for m in ['CNN', 'QNN', 'QRL']: 
        ax.plot(user_range, results[m], color=COLORS[m], linestyle=STYLES[m],
                marker=MARKS[m], markersize=7, linewidth=2.0, label=m)
        
    ax.set_xlabel('Number of Users (T)', fontsize=12)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title(f'Fig 4 — Sum Rate vs Users (A={A}, SNR={SNR_DB}dB)', fontsize=12, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(user_range)
    
    plt.tight_layout()
    plt.savefig('fig4_users.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--users",  type=int, nargs="+", default=USER_RANGE)
    args = parser.parse_args()
    user_range, results = run(args.epochs, sorted(set(args.users)))
    plot(user_range, results)