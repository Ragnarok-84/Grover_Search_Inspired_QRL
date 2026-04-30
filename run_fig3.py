import numpy as np

from config_run import *
np.random.seed(42)
 
N_EPOCHS_FIG3 = 500
SLOTS_PER_E   = 5
N_SCHEDULE    = 4
TAU_FRAC      = 0.6
GROVER_G      = 1
N_LAYERS      = 1
BF_KAVG       = 2
LR_QRL        = 0.02
LR_QNN        = 0.02
LR_CNN        = 1e-3
EVAL_EPOCHS   = 100
SMOOTH_W      = 15
print("=" * 55)
print("Fig 3: QRL convergence (A=32, T=6, SNR=20dB, 500ep)")
print("=" * 55)
h3   = run_qrl(A=32, T=6, snr_db=20.0, n_epochs=N_EPOCHS_FIG3)
ep_x = np.arange(1, N_EPOCHS_FIG3 + 1)
s_h3 = smooth(h3, SMOOTH_W)
s_x  = ep_x[SMOOTH_W - 1:]
 
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(s_x, s_h3, color='steelblue', linewidth=1.5,
        label='Grover Agent Avg Reward')
ax.set_xlabel('Epochs')
ax.set_ylabel('Average Sum Rate (bps/Hz)')
ax.set_title('Grover QRL Agent Training - Rewards')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_training_convergence.png', dpi=150)
plt.close()
print(f"  Start≈{h3[0]:.2f}  End≈{last_mean(h3):.2f} bps/Hz")
print("  Saved: fig3_training_convergence.png\n")