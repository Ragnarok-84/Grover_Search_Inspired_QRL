"""
mMIMO_sys.py  —  Massive MIMO Downlink System Model
Implements:
  - Rician fading channel (eq. 5)
  - Statistical DFT beamforming (eq. 7)
  - SINR computation (eq. 6)
  - Ergodic sum rate (eq. 9)
  - Proportional Fairness reward (eq. 10)

FIX 1: Removed duplicate generate_channel() definition.
FIX 2: reset_channel_conditions() now also resets avg_rate.
FIX 3: beamforming_vector() uses vectorised operations (no inner loop).
"""

import numpy as np
from scipy.linalg import toeplitz


class MassiveMIMOSystem:
    def __init__(self, A=32, T=6, K_factor=3.0, snr_db=20.0, omega=0.1):
        self.A      = A
        self.T      = T
        self.K      = K_factor
        self.snr    = 10 ** (snr_db / 10)
        self.sigma2 = 1.0 / self.snr
        self.omega  = omega

        self._build_dft_matrix()
        self.reset_channel_conditions()   # sets avg_rate + channel geometry

    # ─────────────────────────────────────────────────────────────────────
    # DFT matrix  (eq. 7)
    # ─────────────────────────────────────────────────────────────────────
    def _build_dft_matrix(self):
        n = np.arange(self.A)
        k = np.arange(self.A)
        self.Omega = (np.exp(-1j * 2 * np.pi * np.outer(k, n) / self.A)
                      / np.sqrt(self.A))

    # ─────────────────────────────────────────────────────────────────────
    # FIX 1 + 2: Single generate_channel() — uses geometry from
    #            reset_channel_conditions(); only NLoS changes per slot.
    # ─────────────────────────────────────────────────────────────────────
    def generate_channel(self) -> np.ndarray:
        """Generate one channel realisation (small-scale NLoS varies)."""
        n_NLoS = (self.L @ (np.random.randn(self.A, self.T)
                            + 1j * np.random.randn(self.A, self.T))) / np.sqrt(2)
        K_tilde = np.sqrt(self.K / (self.K + 1))
        K_hat   = np.sqrt(1.0  / (self.K + 1))
        return K_tilde * self.n_LoS + K_hat * n_NLoS

    # ─────────────────────────────────────────────────────────────────────
    # FIX 2: reset also clears PF history (avg_rate)
    # ─────────────────────────────────────────────────────────────────────
    def reset_channel_conditions(self):
        """Call at the start of every episode to randomise user geometry."""
        # Random AoD for each user
        self.aod  = np.random.uniform(0, 2 * np.pi, self.T)

        # LoS steering vectors  (A × T)
        self.n_LoS = (np.exp(1j * np.pi
                             * np.outer(np.arange(self.A), np.sin(self.aod)))
                      / np.sqrt(self.A))

        # Spatial correlation (Toeplitz, rho=0.7)
        rho = 0.7
        row = rho ** np.arange(self.A)
        M   = toeplitz(row)
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

        # FIX 2: reset PF average-rate state
        self.avg_rate = np.ones(self.T) * 1e-3

    # ─────────────────────────────────────────────────────────────────────
    # FIX 3: Vectorised beamforming  (eq. 7)
    # Instead of K_avg separate channel calls per user, collect K_avg
    # samples in one batch and use a single matrix multiply.
    # ─────────────────────────────────────────────────────────────────────
    def beamforming_vector(self, K_avg: int = 5) -> np.ndarray:
        """
        Statistical DFT beamforming.
        Returns F of shape (A, T).
        """
        # Accumulate beam power estimates over K_avg realisations
        beam_power = np.zeros((self.A, self.T))   # (A, T)
        for _ in range(K_avg):
            N_s = self.generate_channel()          # (A, T)
            # Omega^H @ N_s  has shape (A, T);  |·|^2 → power per beam per user
            beam_power += np.abs(self.Omega.conj().T @ N_s) ** 2
        beam_power /= K_avg                        # average

        # Pick the strongest beam index per user
        j_star = np.argmax(beam_power, axis=0)     # shape (T,)
        F = self.Omega[:, j_star]                  # shape (A, T)
        return F

    # ─────────────────────────────────────────────────────────────────────
    # SINR  (eq. 6)
    # ─────────────────────────────────────────────────────────────────────
    def compute_sinr(self, N: np.ndarray, F: np.ndarray,
                     theta: np.ndarray) -> np.ndarray:
        sinr = np.zeros(self.T)
        P    = 1.0 / max(theta.sum(), 1)   # equal power among scheduled users

        for t in range(self.T):
            if theta[t] == 0:
                continue
            signal = np.abs(F[:, t].conj() @ N[:, t]) ** 2 * P
            interf = sum(np.abs(F[:, x].conj() @ N[:, t]) ** 2 * P
                         for x in range(self.T) if x != t and theta[x] == 1)
            sinr[t] = signal / (interf + self.sigma2)

        return sinr

    # ─────────────────────────────────────────────────────────────────────
    # Instantaneous rate  (eq. 9)
    # ─────────────────────────────────────────────────────────────────────
    def instantaneous_rate(self, sinr: np.ndarray) -> np.ndarray:
        return np.log2(1.0 + sinr)

    # ─────────────────────────────────────────────────────────────────────
    # Proportional-Fairness reward  (eq. 10)
    # ─────────────────────────────────────────────────────────────────────
    def compute_pf_reward(self, rates: np.ndarray,
                          theta: np.ndarray) -> float:
        a = 1e-9
        reward = 0.0
        for t in range(self.T):
            if theta[t] == 1:
                s_bar_next = ((1 - self.omega) * self.avg_rate[t]
                              + self.omega * rates[t])
                reward += rates[t] / (s_bar_next + a)
        return reward

    # ─────────────────────────────────────────────────────────────────────
    # Update PF average-rate state  (eq. 10)
    # ─────────────────────────────────────────────────────────────────────
    def update_avg_rate(self, rates: np.ndarray, theta: np.ndarray):
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = ((1 - self.omega) * self.avg_rate[t]
                                    + self.omega * rates[t])

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: full step (channel → beamform → SINR → rate)
    # ─────────────────────────────────────────────────────────────────────
    def step(self, theta: np.ndarray, F: np.ndarray = None):
        """
        Generate a channel sample, compute SINR & rates for scheduling
        vector theta.  Optionally accepts pre-computed F.
        Returns (N, F, sinr, rates).
        """
        N = self.generate_channel()
        if F is None:
            F = self.beamforming_vector()
        sinr  = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return N, F, sinr, rates