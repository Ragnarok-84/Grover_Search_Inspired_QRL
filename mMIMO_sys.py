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

        # Cấu hình UPA (Uniform Planar Array): Tính số hàng (X) và cột (Y)
        # Ví dụ A = 32 -> X = 4, Y = 8
        self.X = int(np.sqrt(self.A))
        while self.A % self.X != 0:
            self.X -= 1
        self.Y = self.A // self.X

        self._build_dft_matrix()
        self.reset_channel_conditions()

    def _build_dft_matrix(self):
        """
        Ma trận DFT cho UPA là Tích Kronecker của ma trận DFT trục X và trục Y.
        """
        nx = np.arange(self.X)
        kx = np.arange(self.X)
        Omega_X = np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / self.X) / np.sqrt(self.X)

        ny = np.arange(self.Y)
        ky = np.arange(self.Y)
        Omega_Y = np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / self.Y) / np.sqrt(self.Y)

        self.Omega = np.kron(Omega_X, Omega_Y)

    def generate_channel(self) -> np.ndarray:
        """
        Sinh kênh Rician: Kết hợp LoS cố định và NLoS biến đổi.
        """
        # Sinh nhiễu i.i.d Rayleigh cho thành phần NLoS
        H_iid = (np.random.randn(self.A, self.T) + 
                 1j * np.random.randn(self.A, self.T)) / np.sqrt(2)
        
        # Áp dụng độ tương quan không gian cho NLoS
        n_NLoS = self.L @ H_iid
        
        K_tilde = np.sqrt(self.K / (self.K + 1))
        K_hat   = np.sqrt(1.0  / (self.K + 1))
        return K_tilde * self.n_LoS + K_hat * n_NLoS

    def reset_channel_conditions(self):
        """
        Khởi tạo hình học môi trường 3D (UPA) cho các user.
        """
        # Góc phương vị (azimuth) và góc tà (elevation)
        azimuth = np.random.uniform(0, 2 * np.pi, self.T)
        elevation = np.random.uniform(0, np.pi, self.T)

        # 1. Tính toán vector Steering LoS cho UPA (A x T)
        self.n_LoS = np.zeros((self.A, self.T), dtype=complex)
        for t in range(self.T):
            ax = np.exp(1j * np.pi * np.arange(self.X) * np.sin(elevation[t]) * np.cos(azimuth[t]))
            ay = np.exp(1j * np.pi * np.arange(self.Y) * np.sin(elevation[t]) * np.sin(azimuth[t]))
            # Kronecker product gộp 2 trục lại thành vector kích thước A = X*Y
            self.n_LoS[:, t] = np.kron(ax, ay) / np.sqrt(self.A)

        # 2. Ma trận tương quan không gian Kronecker Model (R = R_X kron R_Y)
        rho = 0.7
        R_X = toeplitz(rho ** np.arange(self.X))
        R_Y = toeplitz(rho ** np.arange(self.Y))
        M   = np.kron(R_X, R_Y) # Kích thước chuẩn A x A
        
        # Phân rã Cholesky để lấy ma trận lọc không gian L
        self.L = np.linalg.cholesky(M + 1e-9 * np.eye(self.A))

        self.avg_rate = np.ones(self.T) * 1e-3

    def beamforming_vector(self, K_avg: int = 5) -> np.ndarray:
        """
        Định dạng chùm tia DFT (Eq 7)
        """
        # Ánh xạ thành phần LoS lên không gian DFT để tìm hướng tối ưu nhất
        beam_power = np.abs(self.Omega.conj().T @ self.n_LoS) ** 2  # (A, T)
        j_star = np.argmax(beam_power, axis=0)                       # (T,)
        return self.Omega[:, j_star]                                  # (A, T)

    def compute_sinr(self, N: np.ndarray, F: np.ndarray, theta: np.ndarray) -> np.ndarray:
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

    def compute_pf_reward(self, rates: np.ndarray, theta: np.ndarray) -> float:
        a = 1e-9
        reward = 0.0
        for t in range(self.T):
            if theta[t] == 1:
                s_bar_next = ((1 - self.omega) * self.avg_rate[t] + self.omega * rates[t])
                reward += rates[t] / (s_bar_next + a)
        return reward

    def update_avg_rate(self, rates: np.ndarray, theta: np.ndarray):
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = ((1 - self.omega) * self.avg_rate[t] + self.omega * rates[t])

    def step(self, theta: np.ndarray, F: np.ndarray = None):
        N = self.generate_channel()
        if F is None:
            F = self.beamforming_vector()
        sinr  = self.compute_sinr(N, F, theta)
        rates = self.instantaneous_rate(sinr)
        return N, F, sinr, rates