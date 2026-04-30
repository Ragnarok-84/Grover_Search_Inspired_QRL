import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

COLORS = {'QRL': '#1a6fb5', 'QNN': '#1a8c5a', 'CNN': '#c0392b'}
STYLES = {'QRL': '-',       'QNN': '--',       'CNN': ':'}
MARKS  = {'QRL': 'o',       'QNN': 's',        'CNN': '^'}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Training convergence
# train_data = (rewards_history, smooth_rewards)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training(train_data):
    rewards_history, smooth_rewards = train_data

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    epochs = np.arange(1, len(smooth_rewards) + 1)

    ax.plot(epochs, rewards_history,
            color='#aac4e0', linewidth=0.6, alpha=0.5,
            label='Raw sum rate')

    ax.plot(epochs, smooth_rewards,
            color=COLORS['QRL'], linewidth=2.0,
            label='Smoothed (EMA α=0.1)')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title('Fig 3 — Training convergence (A=32, T=6, SNR=20dB)',
                 fontsize=11, fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Sum rate vs users
# users_data = (user_range, user_results)
# ─────────────────────────────────────────────────────────────────────────────
def plot_users(users_data):
    user_range, user_results = users_data

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    for method in ['QRL', 'QNN', 'CNN']:
        ax.plot(user_range, user_results[method],
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKS[method],
                markersize=6,
                linewidth=1.8,
                label=method)

    ax.set_xlabel('Number of users (T)', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title('Fig 4 — Sum rate vs users (A=32, SNR=20dB)',
                 fontsize=11, fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xticks(user_range)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Sum rate vs antennas
# ant_data = (ant_range, ant_results)
# ─────────────────────────────────────────────────────────────────────────────
def plot_antennas(ant_data):
    ant_range, ant_results = ant_data

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    for method in ['QRL', 'QNN', 'CNN']:
        ax.plot(ant_range, ant_results[method],
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKS[method],
                markersize=6,
                linewidth=1.8,
                label=method)

    ax.set_xlabel('Number of BS antennas (A)', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title('Fig 5 — Sum rate vs antennas (T=6, SNR=20dB)',
                 fontsize=11, fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xticks(ant_range)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Sum rate vs SNR
# snr_data = (snr_range, snr_results, configs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_snr(snr_data):
    snr_range, snr_results, configs = snr_data

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    line_styles_cfg = ['-', '--']

    for i, (T, A) in enumerate(configs):
        for method in ['QRL', 'QNN', 'CNN']:
            lbl = f"{method} T={T},A={A}"

            ax.plot(snr_range,
                    snr_results[(T, A)][method],
                    color=COLORS[method],
                    linestyle=line_styles_cfg[i],
                    marker=MARKS[method] if i == 0 else None,
                    markersize=5,
                    linewidth=1.6,
                    label=lbl)

    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Sum rate (bps/Hz)', fontsize=11)
    ax.set_title('Fig 6 — Sum rate vs SNR (two configurations)',
                 fontsize=11, fontweight='bold')

    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(snr_range)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Nếu vẫn muốn plot tất cả
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(train_data, users_data, ant_data, snr_data):
    plot_training(train_data)
    plot_users(users_data)
    plot_antennas(ant_data)
    plot_snr(snr_data)