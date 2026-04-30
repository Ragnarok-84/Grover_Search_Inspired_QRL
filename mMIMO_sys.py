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
        self.reset_channel_conditions()

    def _build_dft_matrix(self):
        n = np.arange(self.A)
        k = np.arange(self.A)
        self.Omega = (np.exp(-1j * 2 * np.pi * np.outer(k, n) / self.A)
                      / np.sqrt(self.A))

    def generate_channel(self) -> np.ndarray:
        """One Rician channel realisation (NLoS varies, LoS fixed per episode)."""
        n_NLoS = (self.L @ (np.random.randn(self.A, self.T)
                            + 1j * np.random.randn(self.A, self.T))) / np.sqrt(2)
        K_tilde = np.sqrt(self.K / (self.K + 1))
        K_hat   = np.sqrt(1.0  / (self.K + 1))
        return K_tilde * self.n_LoS + K_hat * n_NLoS

    def reset_channel_conditions(self):
        """Randomise user geometry at start of every episode."""
        self.aod  = np.random.uniform(0, 2 * np.pi, self.T)
        self.n_LoS = (np.exp(1j * np.pi
                             * np.outer(np.arange(self.A), np.sin(self.aod)))
                      / np.sqrt(self.A))

        rho = 0.7
        row = rho ** np.arange(self.A)
        M   = toeplitz(row)
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

        self.avg_rate = np.ones(self.T) * 1e-3

    def beamforming_vector(self, K_avg: int = 5) -> np.ndarray:
        """
        Statistical DFT beamforming (eq. 7).

        FIX: Project n_LoS onto beam domain instead of averaging noisy samples.

        BUG ROOT CAUSE (old code):
          Averaging |Omega^H @ N_sample|^2 over K_avg realisations converges to
          E[|Omega^H @ N|^2] = LoS_term + NLoS_term.
          With rho=0.7 Toeplitz, NLoS beam-domain power peaks sharply at beam 0
          (diagonal of Omega^H M Omega has max at k=0, value ~5.18 vs ~0.15/user
          for LoS). So averaged beam selection → beam 0 for all users → WRONG.

        FIX: Use n_LoS directly (the deterministic LoS component).
          beam_power[:,t] = |Omega^H n_LoS[:,t]|^2
          This correctly identifies the user's spatial direction.
          K_avg parameter kept for API compatibility but no longer used.
        """
        beam_power = np.abs(self.Omega.conj().T @ self.n_LoS) ** 2  # (A, T)
        j_star = np.argmax(beam_power, axis=0)                       # (T,)
        return self.Omega[:, j_star]                                  # (A, T)

    def compute_sinr(self, N: np.ndarray, F: np.ndarray,
                     theta: np.ndarray) -> np.ndarray:
        sinr = np.zeros(self.T)
        P    = 1.0 / max(theta.sum(), 1)

        for t in range(self.T):
            if theta[t] == 0:
                continue
            signal = np.abs(F[:, t].conj() @ N[:, t]) ** 2 * P
            interf = sum(np.abs(F[:, x].conj() @ N[:, t]) ** 2 * P
                         for x in range(self.T) if x != t and theta[x] == 1)
            sinr[t] = signal / (interf + self.sigma2)

        return sinr

    def instantaneous_rate(self, sinr: np.ndarray) -> np.ndarray:
        return np.log2(1.0 + sinr)

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

    def update_avg_rate(self, rates: np.ndarray, theta: np.ndarray):
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = ((1 - self.omega) * self.avg_rate[t]
                                    + self.omega * rates[t])

    def step(self, theta: np.ndarray, F: np.ndarray = None):
        N = self.generate_channel()
        if F is None:
            F = self.beamforming_vector()
        sinr  = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return N, F, sinr, rates