import numpy as np
from scipy.linalg import toeplitz


class MassiveMIMOSystem:
    def __init__(self, A=32, T=6, K_factor=3.0, snr_db=20.0, omega=0.1):
        self.A = A
        self.T = T
        self.K = K_factor
        self.snr = 10 ** (snr_db / 10)
        self.sigma2 = 1.0 / self.snr
        self.omega = omega

        # PF state
        self.avg_rate = np.ones(T) * 1e-3

        self._build_dft_matrix()
        self.reset_channel_conditions()

    # ─────────────────────────────────────
    # DFT matrix (eq. 7)
    # ─────────────────────────────────────
    def _build_dft_matrix(self):
        n = np.arange(self.A)
        k = np.arange(self.A)
        self.Omega = np.exp(-1j * 2 * np.pi * np.outer(k, n) / self.A) / np.sqrt(self.A)

    # ─────────────────────────────────────
    # Generate Rician channel (eq. 5)
    # ─────────────────────────────────────
    def generate_channel(self):
        # Random AoD mỗi lần (IMPORTANT)
        aod = np.random.uniform(0, 2 * np.pi, self.T)

        # LoS (normalized)
        n_LoS = np.exp(1j * np.pi * np.outer(np.arange(self.A), np.sin(aod)))
        n_LoS /= np.sqrt(self.A)

        # Spatial correlation matrix
        rho = 0.7
        row = rho ** np.arange(self.A)
        M = toeplitz(row)
        L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

        # NLoS
        n_NLoS = (L @ (np.random.randn(self.A, self.T)
                      + 1j * np.random.randn(self.A, self.T))) / np.sqrt(2)

        K_tilde = np.sqrt(self.K / (self.K + 1))
        K_hat = np.sqrt(1.0 / (self.K + 1))

        N = K_tilde * n_LoS + K_hat * n_NLoS
        return N

    # ─────────────────────────────────────
    # Beamforming (eq. 7 — statistical)
    # ─────────────────────────────────────
    def beamforming_vector(self, N):
        """
        Approximate E[||N||^2] bằng averaging nhiều sample
        """
        F = np.zeros((self.A, self.T), dtype=complex)

        K_avg = 5  # số sample để estimate expectation

        for t in range(self.T):
            beam_power = np.zeros(self.A)

            for _ in range(K_avg):
                N_sample = self.generate_channel()
                beam_power += np.abs(self.Omega.conj().T @ N_sample[:, t]) ** 2

            beam_power /= K_avg

            j_star = np.argmax(beam_power)
            F[:, t] = self.Omega[:, j_star]

        return F

    # ─────────────────────────────────────
    # SINR (eq. 6)
    # ─────────────────────────────────────
    def compute_sinr(self, N, F, theta):
        sinr = np.zeros(self.T)
        P = 1.0 / self.T   # fixed theo paper

        for t in range(self.T):
            if theta[t] == 0:
                continue

            # signal
            signal = np.abs(F[:, t].conj() @ N[:, t]) ** 2 * P

            # interference
            interf = 0.0
            for x in range(self.T):
                if x != t and theta[x] == 1:
                    interf += np.abs(F[:, x].conj() @ N[:, t]) ** 2 * P

            sinr[t] = signal / (interf + self.sigma2)

        return sinr

    # ─────────────────────────────────────
    # Rate (eq. 9)
    # ─────────────────────────────────────
    def instantaneous_rate(self, sinr):
        return np.log2(1.0 + sinr)

    # ─────────────────────────────────────
    # PF reward (eq. 10)
    # ─────────────────────────────────────
    def compute_pf_reward(self, rates, theta):
        a = 1e-9
        reward = 0.0

        for t in range(self.T):
            if theta[t] == 1:
                s_bar_next = (1 - self.omega) * self.avg_rate[t] + self.omega * rates[t]
                reward += rates[t] / (s_bar_next + a)

        return reward

    # ─────────────────────────────────────
    # Update PF state (eq. 10)
    # ─────────────────────────────────────
    def update_avg_rate(self, rates, theta):
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = (
                    (1 - self.omega) * self.avg_rate[t]
                    + self.omega * rates[t]
                )
                
    def reset_channel_conditions(self):
        """Được gọi ở đầu mỗi episode để random lại vị trí/góc của user"""
        self.aod = np.random.uniform(0, 2 * np.pi, self.T)
        
        self.n_LoS = np.exp(1j * np.pi * np.outer(np.arange(self.A), np.sin(self.aod)))
        self.n_LoS /= np.sqrt(self.A)
        
        rho = 0.7
        row = rho ** np.arange(self.A)
        M = toeplitz(row)
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

    def generate_channel(self):
        """Chỉ tạo ra mẫu NLoS mới (small-scale fading) cho mỗi time slot"""
        n_NLoS = (self.L @ (np.random.randn(self.A, self.T)
                      + 1j * np.random.randn(self.A, self.T))) / np.sqrt(2)

        K_tilde = np.sqrt(self.K / (self.K + 1))
        K_hat = np.sqrt(1.0 / (self.K + 1))

        N = K_tilde * self.n_LoS + K_hat * n_NLoS
        return N