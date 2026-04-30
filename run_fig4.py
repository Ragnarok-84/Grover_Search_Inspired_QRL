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

EVAL_STEPS = 50

COLORS = {'QRL': '#1a6fb5', 'QNN': '#1a8c5a', 'CNN': '#c0392b'}
STYLES = {'QRL': '-',       'QNN': '--',       'CNN': ':'}
MARKS  = {'QRL': 'o',       'QNN': 's',        'CNN': '^'}


def n_sched(T):
    # Số user cho CNN/QNN baseline lấy mẫu
    return max(1, T // 2)


def evaluate(agent, sys, F, n_sc, is_qrl=False):
    """Đánh giá trên cấu hình môi trường hiện tại (không reset)"""
    rates_all = []
    for _ in range(EVAL_STEPS):
        N = sys.generate_channel()
        if is_qrl:
            theta = agent.select(N, n_sc, mimo_sys=sys, F=F)
        else:
            theta = agent.select(N, n_sc)
            
        sinr  = sys.compute_sinr(N, F, theta)
        rates = sys.instantaneous_rate(sinr)
        rates_all.append(float(np.sum(rates)))
        
    return float(np.mean(rates_all))


# ─────────────────────────────────────────────────────────────────────────────
# Training 
# ─────────────────────────────────────────────────────────────────────────────
def run(n_epochs=N_EPOCHS, user_range=USER_RANGE):
    print(f"Fig 4 — Sum rate vs users  (A={A}, SNR={SNR_DB}dB, {n_epochs} epochs)")

    results = {m: [] for m in ['QRL', 'QNN', 'CNN']}
    t0 = time.time()

    for T in user_range:
        n_sc = n_sched(T)
        
        # Scale TAU động theo T vì hệ thống càng nhiều user thì sum rate trung bình càng lớn
        tau  = 2.5 * T  
        
        qrl  = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=G_ITERS, tau=tau)
        qnn  = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
        cnn  = CNNScheduler(A=A, T=T, lr=LR_CNN)
        sys  = MassiveMIMOSystem(A=A, T=T, snr_db=SNR_DB)

        # Cố định hình học kênh truyền cho vòng đời epoch của T này
        sys.reset_channel_conditions()
        F = sys.beamforming_vector()

        print(f"\n  T={T}  (n_sched={n_sc})  "
              f"~{n_epochs * 2 * 2 * N_LAYERS * T * 2:,} QNode calls")

        # Khởi tạo baseline tracking cho cả 3 agent
        base_qrl, base_qnn, base_cnn = 0.0, 0.0, 0.0

        for epoch in range(1, n_epochs + 1):
            N = sys.generate_channel()

            # ── QRL ──────────────────────────────────────────────────────
            theta_qrl  = qrl.select(N, n_sc, mimo_sys=sys, F=F)
            sinr_qrl   = sys.compute_sinr(N, F, theta_qrl)
            rates_qrl  = sys.instantaneous_rate(sinr_qrl)
            
            r_qrl      = float(np.sum(rates_qrl))
            base_qrl   = r_qrl if epoch == 1 else 0.9 * base_qrl + 0.1 * r_qrl
            adv_qrl    = r_qrl - base_qrl
            
            sys.update_avg_rate(rates_qrl, theta_qrl)
            qrl.update(N, adv_qrl, theta_qrl)

            # ── QNN ──────────────────────────────────────────────────────
            theta_qnn  = qnn.select(N, n_sc)
            sinr_qnn   = sys.compute_sinr(N, F, theta_qnn)
            rates_qnn  = sys.instantaneous_rate(sinr_qnn)
            
            r_qnn      = float(np.sum(rates_qnn))
            base_qnn   = r_qnn if epoch == 1 else 0.9 * base_qnn + 0.1 * r_qnn
            adv_qnn    = r_qnn - base_qnn
            
            qnn.update(N, adv_qnn, theta_qnn)

            # ── CNN ──────────────────────────────────────────────────────
            theta_cnn  = cnn.select(N, n_sc)
            sinr_cnn   = sys.compute_sinr(N, F, theta_cnn)
            rates_cnn  = sys.instantaneous_rate(sinr_cnn)
            
            r_cnn      = float(np.sum(rates_cnn))
            base_cnn   = r_cnn if epoch == 1 else 0.9 * base_cnn + 0.1 * r_cnn
            adv_cnn    = r_cnn - base_cnn
            
            cnn.update(N, adv_cnn, theta_cnn)

            if epoch % 100 == 0:
                print(f"    epoch {epoch}/{n_epochs}  "
                      f"({time.time()-t0:.0f}s elapsed)")

        # Đánh giá chung sau khi hội tụ
        r_qrl = evaluate(qrl, sys, F, n_sc, is_qrl=True)
        r_qnn = evaluate(qnn, sys, F, n_sc, is_qrl=False)
        r_cnn = evaluate(cnn, sys, F, n_sc, is_qrl=False)

        results['QRL'].append(r_qrl)
        results['QNN'].append(r_qnn)
        results['CNN'].append(r_cnn)
        print(f"  T={T}  →  QRL={r_qrl:.3f}  QNN={r_qnn:.3f}  CNN={r_cnn:.3f}  bps/Hz")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    return user_range, results


def plot(user_range, results):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    for m in ['CNN', 'QNN', 'QRL']: # Vẽ từ thấp lên cao để QRL không bị đè line
        ax.plot(user_range, results[m],
                color=COLORS[m], linestyle=STYLES[m],
                marker=MARKS[m], markersize=7, linewidth=2.0, label=m)
        
    ax.set_xlabel('Number of Users (T)', fontsize=12)
    ax.set_ylabel('Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title(f'Fig 4 — QNN vs. CNN vs. QRL vs. Users (A={A}, SNR={SNR_DB}dB)',
                 fontsize=12, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
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