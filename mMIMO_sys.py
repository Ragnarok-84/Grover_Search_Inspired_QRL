import numpy as np
from scipy.linalg import toeplitz


class MassiveMIMOSystem:

    def __init__(self, A: int = 32, T: int = 6,
                 K_factor: float = 3.0, snr_db: float = 20.0,
                 omega: float = 0.1):
        """
        Parameters
        ----------
        A        : total BS antennas  (must be factorable as X*Y)
        T        : number of candidate users
        K_factor : Rician K-factor (LoS-to-NLoS power ratio)
        snr_db   : downlink SNR in dB
        omega    : PF forgetting factor
        """
        self.A      = A
        self.T      = T
        self.K      = K_factor
        self.snr    = 10 ** (snr_db / 10.0)
        self.sigma2 = 1.0 / self.snr
        self.omega  = omega

        # ── UPA dimensions: X rows × Y cols, X*Y = A ──────────────────────
        self.X = int(np.floor(np.sqrt(A)))
        while A % self.X != 0:
            self.X -= 1
        self.Y = A // self.X   # e.g. A=32 → X=4, Y=8

        self._build_dft_matrix()
        self.reset_channel_conditions()

    # ──────────────────────────────────────────────────────────────────────
    # DFT / geometry
    # ──────────────────────────────────────────────────────────────────────

    def _build_dft_matrix(self):
        """
        UPA DFT matrix = kron(Omega_X, Omega_Y)  shape: (A, A)
        Each axis uses an unitary DFT normalised by 1/sqrt(dim).
        """
        nx = np.arange(self.X)
        kx = np.arange(self.X)
        Omega_X = (np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / self.X)
                   / np.sqrt(self.X))

        ny = np.arange(self.Y)
        ky = np.arange(self.Y)
        Omega_Y = (np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / self.Y)
                   / np.sqrt(self.Y))

        self.Omega = np.kron(Omega_X, Omega_Y)   # (A, A)

    # ──────────────────────────────────────────────────────────────────────
    # Channel geometry initialisation
    # ──────────────────────────────────────────────────────────────────────

    def reset_channel_conditions(self):
        """
        Randomise the large-scale geometry (angles, correlation) for all T
        users.  Call ONCE before the training loop; geometry is then fixed.
        """
        azimuth   = np.random.uniform(0.0, 2 * np.pi, self.T)
        elevation = np.random.uniform(0.0,     np.pi, self.T)

        # ── LoS steering vectors  (UPA Kronecker structure) ───────────────
        # Each vector has power = A (no 1/sqrt(A) normalisation),
        # which keeps channel power O(A) and sum rate in the 20–32 bps/Hz
        # range expected by the paper at SNR=20 dB.
        self.n_LoS = np.zeros((self.A, self.T), dtype=complex)
        for t in range(self.T):
            sin_el = np.sin(elevation[t])
            cos_az = np.cos(azimuth[t])
            sin_az = np.sin(azimuth[t])
            ax = np.exp(1j * np.pi * np.arange(self.X) * sin_el * cos_az)
            ay = np.exp(1j * np.pi * np.arange(self.Y) * sin_el * sin_az)
            self.n_LoS[:, t] = np.kron(ax, ay)   # length A, power = A

        # ── Spatial correlation  M ∈ C^{A×A}  (paper eq. 5, typo fixed) ──
        # M = kron(R_X, R_Y)  with exponential Toeplitz, rho = 0.7
        rho = 0.7
        R_X = toeplitz(rho ** np.arange(self.X))   # (X, X)
        R_Y = toeplitz(rho ** np.arange(self.Y))   # (Y, Y)
        M   = np.kron(R_X, R_Y)                   # (A, A)

        # Cholesky factor for spatially correlated NLoS generation
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))  # (A, A)

        # Rician mixture coefficients
        self._K_tilde = np.sqrt(self.K / (self.K + 1.0))   # LoS weight
        self._K_hat   = np.sqrt(1.0   / (self.K + 1.0))    # NLoS weight

        # PF historical rate tracker
        self.avg_rate = np.ones(self.T) * 1e-3

    # ──────────────────────────────────────────────────────────────────────
    # Channel generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_channel(self) -> np.ndarray:
        """
        Sample one realisation of the correlated Rician channel (eq. 4-5).

        H = K_tilde * n_LoS  +  K_hat * L @ H_iid

        Returns N : (A, T) complex array
        """
        H_iid = (np.random.randn(self.A, self.T)
                 + 1j * np.random.randn(self.A, self.T)) / np.sqrt(2.0)
        n_NLoS = self.L @ H_iid                               # (A, T)
        return self._K_tilde * self.n_LoS + self._K_hat * n_NLoS

    # ──────────────────────────────────────────────────────────────────────
    # Beamforming
    # ──────────────────────────────────────────────────────────────────────

    def beamforming_vector(self) -> np.ndarray:
        """
        Statistical DFT beamforming (eq. 7): for each user t find the DFT
        column j* that maximises |Omega^H n_LoS_t|^2, then use Omega[:, j*]
        as the beamforming vector.

        Returns F : (A, T) complex array  (one beam column per user)
        """
        # Project LoS steering vectors onto DFT beam space
        beam_power = np.abs(self.Omega.conj().T @ self.n_LoS) ** 2  # (A, T)
        j_star     = np.argmax(beam_power, axis=0)                   # (T,)
        return self.Omega[:, j_star]                                 # (A, T)

    # ──────────────────────────────────────────────────────────────────────
    # SINR / rate / reward
    # ──────────────────────────────────────────────────────────────────────

    def compute_sinr(self, N: np.ndarray, F: np.ndarray,
                     theta: np.ndarray) -> np.ndarray:
        """
        Compute per-user SINR for scheduled users (eq. 6).
        Equal power allocation P = 1 / |{t: theta_t=1}|.
        """
        sinr  = np.zeros(self.T)
        n_act = max(int(theta.sum()), 1)
        P     = 1.0 / n_act

        for t in range(self.T):
            if theta[t] == 0:
                continue
            signal = np.abs(F[:, t].conj() @ N[:, t]) ** 2 * P
            interf = sum(
                np.abs(F[:, x].conj() @ N[:, t]) ** 2 * P
                for x in range(self.T) if x != t and theta[x] == 1
            )
            sinr[t] = signal / (interf + self.sigma2)

        return sinr

    def instantaneous_rate(self, sinr: np.ndarray) -> np.ndarray:
        """Per-user Shannon rate  R_t = log2(1 + SINR_t)  (eq. 3)."""
        return np.log2(1.0 + sinr)

    def compute_pf_reward(self, rates: np.ndarray,
                          theta: np.ndarray) -> float:
        """Proportional Fairness reward (eq. 10)."""
        a      = 1e-9
        reward = 0.0
        for t in range(self.T):
            if theta[t] == 1:
                s_bar_next = ((1 - self.omega) * self.avg_rate[t]
                              + self.omega * rates[t])
                reward += rates[t] / (s_bar_next + a)
        return reward

    def update_avg_rate(self, rates: np.ndarray, theta: np.ndarray):
        """EMA update of historical average rates (eq. 10, second line)."""
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = ((1 - self.omega) * self.avg_rate[t]
                                    + self.omega * rates[t])

    def step(self, theta: np.ndarray, F: np.ndarray = None):
        """
        Convenience wrapper: sample channel, compute SINR/rates.
        Returns (N, F, sinr, rates).
        """
        N = self.generate_channel()
        if F is None:
            F = self.beamforming_vector()
        sinr  = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return N, F, sinr, rates