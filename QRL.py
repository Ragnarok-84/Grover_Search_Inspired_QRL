"""
QRLAgent.py - Grover's Search-Inspired Quantum Reinforcement Learning
Implement thuật toán QRL lai ghép VQC và Amplitude Amplification.

Kiến trúc:
  1. State Preparation (Toán tử A): Variational Quantum Circuit (VQC) với trọng số học được.
  2. Oracle (U_w): Lật pha (Phase flip) các chiến lược lập lịch thỏa mãn điều kiện reward.
  3. Diffusion (D_AA): A -> S_0 -> A_dagger (Khuếch đại biên độ chuẩn mực).
  4. Cập nhật (REINFORCE): Dùng đạo hàm Parameter-shift trên base VQC để học lại phân bố 
     đã được khuếch đại bởi Grover.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QRLAgent:
    def __init__(self, A: int, T: int, n_layers: int = 2, lr: float = 0.02, G: int = 1):
        self.A = A
        self.T = T
        self.n_layers = n_layers
        self.lr = lr
        self.G = G  # Số vòng lặp Grover
        self.N_states = 2 ** T

        # Quantum device
        self.dev = qml.device("default.qubit", wires=T)

        # Trọng số có thể huấn luyện của VQC: shape (n_layers, T, 2)
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Precompute bit-mask ma trận (Tăng tốc độ tính marginals)
        indices = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1).astype(np.float32)

        # Adam Optimizer State
        self._adam_t = 0
        self._adam_m = np.zeros((n_layers, T, 2))
        self._adam_v = np.zeros((n_layers, T, 2))
        self._adam_b1, self._adam_b2, self._adam_eps = 0.9, 0.999, 1e-8

        self._build_circuits()

    # ──────────────────────────────────────────────────────────────────────
    # 1. TOÁN TỬ A (Variational Quantum Circuit)
    # ──────────────────────────────────────────────────────────────────────
    def _vqc_ansatz(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.T), rotation='Y')

        for l in range(self.n_layers):
            for i in range(self.T):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(self.T - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.T - 1, 0])

    # ──────────────────────────────────────────────────────────────────────
    # 2. ORACLE U_w (Phase Flip các trạng thái mục tiêu)
    # ──────────────────────────────────────────────────────────────────────
    def _oracle(self, marked_list):
        for idx in marked_list:
            bitstring = format(idx, f'0{self.T}b')
            
            # Đảo các qubit có giá trị '0' thành '1' để chuẩn bị cho Multi-controlled Z
            for i, b in enumerate(bitstring):
                if b == '0':
                    qml.PauliX(wires=i)

            # Multi-controlled Z lên qubit cuối cùng
            qml.ctrl(qml.PauliZ, control=list(range(self.T - 1)))(wires=self.T - 1)

            # Undo PauliX
            for i, b in enumerate(bitstring):
                if b == '0':
                    qml.PauliX(wires=i)

    # ──────────────────────────────────────────────────────────────────────
    # 3. TOÁN TỬ KHUẾCH TÁN (Diffusion Operator D_AA = A * S_0 * A_dagger)
    # ──────────────────────────────────────────────────────────────────────
    def _diffusion(self, inputs, weights):
        # 3.1. Nghịch đảo của VQC (A_dagger)
        qml.adjoint(self._vqc_ansatz)(inputs, weights)

        # 3.2. Lật pha trạng thái |00...0> (S_0)
        for i in range(self.T):
            qml.PauliX(wires=i)
            
        qml.ctrl(qml.PauliZ, control=list(range(self.T - 1)))(wires=self.T - 1)
        
        for i in range(self.T):
            qml.PauliX(wires=i)

        # 3.3. Áp dụng lại VQC (A)
        self._vqc_ansatz(inputs, weights)

    # ──────────────────────────────────────────────────────────────────────
    # XÂY DỰNG QNODES
    # ──────────────────────────────────────────────────────────────────────
    def _build_circuits(self):
        # Mạch VQC cơ sở (Dùng để tính Gradient Update)
        @qml.qnode(self.dev)
        def base_vqc(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        
        self.base_vqc = base_vqc

        # Mạch Grover đầy đủ (Dùng để chọn action tốt nhất)
        @qml.qnode(self.dev)
        def grover_circ(inputs, weights, marked_list):
            # Khởi tạo trạng thái A|0>
            self._vqc_ansatz(inputs, weights)
            
            # Lặp Grover G lần
            for _ in range(self.G):
                self._oracle(marked_list)
                self._diffusion(inputs, weights)
                
            return qml.probs(wires=range(self.T))
            
        self.grover_circ = grover_circ

    # ──────────────────────────────────────────────────────────────────────
    # UTILS: Xử lý Kênh truyền & Marginals (Giống QNNScheduler)
    # ──────────────────────────────────────────────────────────────────────
    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        magnitudes = np.mean(np.abs(N), axis=0)
        magnitudes = magnitudes / (magnitudes.max() + 1e-9)
        return magnitudes * np.pi

    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        return self._bit_mask.T @ probs_all

    def _log_prob_from_marginals(self, marginals: np.ndarray, theta: np.ndarray) -> float:
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    # ──────────────────────────────────────────────────────────────────────
    # CHỌN ACTION (Kết hợp tính năng Khuếch đại của Grover)
    # ──────────────────────────────────────────────────────────────────────
    def select(self, N: np.ndarray, marked_list: list, n_schedule: int) -> np.ndarray:
        """
        Nếu có marked_list (các state có reward >= tau), mạch Grover sẽ khuếch đại 
        xác suất của chúng trước khi đo lường (measurement).
        """
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)

        if len(marked_list) > 0:
            probs_all = np.array(self.grover_circ(inputs, self.weights, marked_list), dtype=float)
        else:
            probs_all = np.array(self.base_vqc(inputs, self.weights), dtype=float)

        marginals = self._probs_to_marginals(probs_all)
        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    # ──────────────────────────────────────────────────────────────────────
    # CẬP NHẬT TRỌNG SỐ (Học lại phân bố từ hành động đã chọn)
    # ──────────────────────────────────────────────────────────────────────
    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        Dùng Parameter-shift rule trên mạch base_vqc để tiết kiệm tính toán.
        Mạch base_vqc sẽ dần dần được 'nắn' trọng số để tự động sinh ra các 
        hành động tốt mà không cần đến Grover trong tương lai.
        """
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift = np.pi / 2
        w_raw = np.array(self.weights, dtype=float)
        grad = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    # Forward shift
                    w_raw[l, i, k] += shift
                    p_plus = np.array(self.base_vqc(inputs, w_raw), dtype=float)
                    lp_plus = self._log_prob_from_marginals(self._probs_to_marginals(p_plus), theta)

                    # Backward shift
                    w_raw[l, i, k] -= 2 * shift
                    p_minus = np.array(self.base_vqc(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob_from_marginals(self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift  # Restore
                    
                    # Tính Gradient REINFORCE
                    grad[l, i, k] = reward * (lp_plus - lp_minus) / 2.0

        # Cập nhật Adam
        self._adam_t += 1
        self._adam_m = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

        self.weights = pnp.array(new_w, requires_grad=True)