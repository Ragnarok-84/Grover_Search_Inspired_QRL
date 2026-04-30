import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from mMIMO_sys  import MassiveMIMOSystem
from QRL        import QRLAgent
from benchmark  import CNNScheduler, QNNScheduler
 
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
 
 
def run_qrl(A, T, snr_db, n_epochs):
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=snr_db)
    agent = QRLAgent(A=A, T=T, n_layers=N_LAYERS, lr=LR_QRL, G=GROVER_G)
    n_sch = min(N_SCHEDULE, T)
    history = []
 
    for ep in range(n_epochs):
        env.reset_channel_conditions()
        F       = env.beamforming_vector(K_avg=BF_KAVG)
        ep_sum  = []
        marked  = []
        rew_buf = []
 
        for _ in range(SLOTS_PER_E):
            N      = env.generate_channel()
            theta  = agent.select(N, marked, n_sch)
            sinr   = env.compute_sinr(N, F, theta)
            rates  = env.instantaneous_rate(sinr)
            reward = float(rates[theta == 1].sum())
 
            tau = TAU_FRAC * (np.mean(rew_buf) if rew_buf else reward)
            if reward >= tau:
                pad   = (8 - T % 8) % 8
                bits  = np.pad(theta.astype(np.uint8), (0, pad))
                idx   = int(np.packbits(bits, bitorder='big')[0]) >> pad
                idx   = int(idx) % (2 ** T)
                if idx not in marked:
                    marked.append(idx)
                if len(marked) > (2 ** T) // 3:
                    marked = marked[-(2 ** T) // 4:]
 
            agent.update(N, reward, theta)
            env.update_avg_rate(rates, theta)
            ep_sum.append(rates[theta == 1].sum())
            rew_buf.append(reward)
 
        history.append(float(np.mean(ep_sum)))
        if (ep + 1) % 50 == 0:
            print(f"    [QRL A={A} T={T}] ep {ep+1}/{n_epochs}  "
                  f"rate={history[-1]:.3f}", flush=True)
    return history
 
 
def run_qnn(A, T, snr_db, n_epochs):
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=snr_db)
    agent = QNNScheduler(A=A, T=T, n_layers=N_LAYERS, lr=LR_QNN)
    n_sch = min(N_SCHEDULE, T)
    history = []
 
    for ep in range(n_epochs):
        env.reset_channel_conditions()
        F      = env.beamforming_vector(K_avg=BF_KAVG)
        ep_sum = []
        for _ in range(SLOTS_PER_E):
            N      = env.generate_channel()
            theta  = agent.select(N, n_sch)
            sinr   = env.compute_sinr(N, F, theta)
            rates  = env.instantaneous_rate(sinr)
            reward = float(rates[theta == 1].sum())
            agent.update(N, reward, theta)
            env.update_avg_rate(rates, theta)
            ep_sum.append(rates[theta == 1].sum())
        history.append(float(np.mean(ep_sum)))
    return history
 
 
def run_cnn(A, T, snr_db, n_epochs):
    env   = MassiveMIMOSystem(A=A, T=T, snr_db=snr_db)
    agent = CNNScheduler(A=A, T=T, lr=LR_CNN)
    n_sch = min(N_SCHEDULE, T)
    history = []
 
    for ep in range(n_epochs):
        env.reset_channel_conditions()
        F      = env.beamforming_vector(K_avg=BF_KAVG)
        ep_sum = []
        for _ in range(SLOTS_PER_E):
            N      = env.generate_channel()
            theta  = agent.select(N, n_sch)
            sinr   = env.compute_sinr(N, F, theta)
            rates  = env.instantaneous_rate(sinr)
            reward = float(rates[theta == 1].sum())
            agent.update(N, reward, theta)
            env.update_avg_rate(rates, theta)
            ep_sum.append(rates[theta == 1].sum())
        history.append(float(np.mean(ep_sum)))
    return history
 
 
def smooth(x, w=SMOOTH_W):
    return np.convolve(x, np.ones(w) / w, mode='valid')
 
def last_mean(h, n=20):
    return float(np.mean(h[-n:]))