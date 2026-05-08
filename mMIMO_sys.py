import numpy as np
from scipy.linalg import toeplitz


class MassiveMIMOSystem:
    """
    Single-cell massive-MIMO downlink environment with correlated Rician fading,
    DFT/statistical beamforming, Shannon rate, sum-rate, and PF reward.
    """

    def __init__(self, A: int = 32, T: int = 6,
                 K_factor: float = 3.0, snr_db: float = 20.0,
                 omega: float = 0.1, total_power: float = 1.0,
                 pf_epsilon: float = 1e-9, rho: float = 0.7,
                 seed: int | None = None):
        self.A = A
        self.T = T
        self.K = K_factor
        self.snr_db = snr_db
        self.snr = 10 ** (snr_db / 10.0)
        self.sigma2 = 1.0 / self.snr
        self.omega = omega
        self.total_power = total_power
        self.pf_epsilon = pf_epsilon
        self.rho = rho
        self.rng = np.random.default_rng(seed)

        self.X = int(np.floor(np.sqrt(A)))
        while A % self.X != 0:
            self.X -= 1
        self.Y = A // self.X

        self._build_dft_matrix()
        self.reset_channel_conditions()

    # ──────────────────────────────────────────────────────────────────────
    # DFT / geometry
    # ──────────────────────────────────────────────────────────────────────
    def _build_dft_matrix(self):
        nx = np.arange(self.X); kx = np.arange(self.X)
        Omega_X = np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / self.X) / np.sqrt(self.X)
        ny = np.arange(self.Y); ky = np.arange(self.Y)
        Omega_Y = np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / self.Y) / np.sqrt(self.Y)
        self.Omega = np.kron(Omega_X, Omega_Y)

    def reset_channel_conditions(self):
        """Fix large-scale user geometry. Call once before a training run."""
        azimuth = self.rng.uniform(0.0, 2 * np.pi, self.T)
        elevation = self.rng.uniform(0.0, np.pi, self.T)

        self.n_LoS = np.zeros((self.A, self.T), dtype=complex)
        for t in range(self.T):
            sin_el = np.sin(elevation[t])
            ax = np.exp(1j * np.pi * np.arange(self.X) * sin_el * np.cos(azimuth[t]))
            ay = np.exp(1j * np.pi * np.arange(self.Y) * sin_el * np.sin(azimuth[t]))
            # Unnormalised steering vector. Power is O(A), matching the scale
            # used in the previous implementation and the paper-like plots.
            self.n_LoS[:, t] = np.kron(ax, ay)

        R_X = toeplitz(self.rho ** np.arange(self.X))
        R_Y = toeplitz(self.rho ** np.arange(self.Y))
        M = np.kron(R_X, R_Y)
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

        self._K_tilde = np.sqrt(self.K / (self.K + 1.0))
        self._K_hat = np.sqrt(1.0 / (self.K + 1.0))
        self.reset_pf_state()

    def reset_pf_state(self):
        self.avg_rate = np.ones(self.T, dtype=float) * 1e-3

    # ──────────────────────────────────────────────────────────────────────
    # Channel generation / beamforming
    # ──────────────────────────────────────────────────────────────────────
    def generate_channel(self) -> np.ndarray:
        H_iid = (self.rng.standard_normal((self.A, self.T))
                 + 1j * self.rng.standard_normal((self.A, self.T))) / np.sqrt(2.0)
        n_NLoS = self.L @ H_iid
        return self._K_tilde * self.n_LoS + self._K_hat * n_NLoS

    def beamforming_vector(self, N_stat: np.ndarray | None = None) -> np.ndarray:
        """
        DFT beamforming: choose the strongest beam for each user.
        If N_stat is provided, use that channel; otherwise use fixed LoS geometry.
        """
        channel_for_beams = self.n_LoS if N_stat is None else N_stat
        beam_power = np.abs(self.Omega.conj().T @ channel_for_beams) ** 2
        j_star = np.argmax(beam_power, axis=0)
        return self.Omega[:, j_star]

    # ──────────────────────────────────────────────────────────────────────
    # SINR / rate / reward
    # ──────────────────────────────────────────────────────────────────────
    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=int).reshape(-1)
        if theta.size != self.T:
            raise ValueError(f"theta length must be T={self.T}, got {theta.size}")
        if not np.all((theta == 0) | (theta == 1)):
            raise ValueError("theta must be binary")
        return theta

    def compute_sinr(self, N: np.ndarray, F: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta = self._validate_theta(theta)
        sinr = np.zeros(self.T, dtype=float)
        active = np.where(theta == 1)[0]
        if active.size == 0:
            return sinr

        # Paper assumes equal power per user. For fixed-size scheduling this is
        # equivalent to total_power / active.size; for paper Eq. notation it is
        # also close to total_power / T when active.size is fixed.
        P_user = self.total_power / active.size

        for t in active:
            signal = np.abs(F[:, t].conj() @ N[:, t]) ** 2 * P_user
            interf = 0.0
            for x in active:
                if x != t:
                    interf += np.abs(F[:, x].conj() @ N[:, t]) ** 2 * P_user
            sinr[t] = signal / (interf + self.sigma2)
        return sinr

    def instantaneous_rate(self, sinr: np.ndarray) -> np.ndarray:
        return np.log2(1.0 + np.asarray(sinr, dtype=float))

    def compute_sum_rate(self, rates: np.ndarray, theta: np.ndarray | None = None) -> float:
        rates = np.asarray(rates, dtype=float)
        if theta is None:
            return float(np.sum(rates))
        theta = self._validate_theta(theta)
        return float(np.sum(rates * theta))

    def compute_pf_reward(self, rates: np.ndarray, theta: np.ndarray) -> float:
        theta = self._validate_theta(theta)
        rates = np.asarray(rates, dtype=float)
        reward = 0.0
        for t in range(self.T):
            if theta[t] == 1:
                s_bar_next = (1 - self.omega) * self.avg_rate[t] + self.omega * rates[t]
                reward += rates[t] / (s_bar_next + self.pf_epsilon)
        return float(reward)

    def update_avg_rate(self, rates: np.ndarray, theta: np.ndarray):
        theta = self._validate_theta(theta)
        rates = np.asarray(rates, dtype=float)
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = (1 - self.omega) * self.avg_rate[t] + self.omega * rates[t]

    def evaluate_theta(self, N: np.ndarray, F: np.ndarray, theta: np.ndarray):
        sinr = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return sinr, rates, self.compute_sum_rate(rates, theta), self.compute_pf_reward(rates, theta)

    def step(self, theta: np.ndarray, F: np.ndarray | None = None):
        N = self.generate_channel()
        if F is None:
            F = self.beamforming_vector()
        sinr = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return N, F, sinr, rates
