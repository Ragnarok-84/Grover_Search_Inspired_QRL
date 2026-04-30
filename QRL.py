import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:

    def __init__(self, A: int, T: int, n_layers: int = 3,
                 lr: float = 0.02, G: int = 1, tau: float = 4.0):
        """
        Parameters
        ----------
        A        : BS antenna count
        T        : number of candidate users (= number of qubits)
        n_layers : VQC depth
        lr       : Adam learning rate for REINFORCE
        G        : Grover iterations per inference step
        tau      : oracle threshold (instantaneous sum rate, bps/Hz)
        """
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.G        = G
        self.tau      = tau
        self.N_states = 2 ** T

        # ── UPA dimensions (must match mMIMO_sys.py) ──────────────────────
        self.X = int(np.floor(np.sqrt(A)))
        while A % self.X != 0:
            self.X -= 1
        self.Y = A // self.X

        self._build_upa_dft()

        # ── PennyLane device ──────────────────────────────────────────────
        self.dev = qml.device("default.qubit", wires=T)

        # ── Trainable VQC weights: (n_layers, T, 2)  [RY + RZ per qubit] ─
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # ── Bit-mask for marginal computation: shape (N_states, T) ────────
        indices        = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                          ).astype(np.float32)   # (N_states, T)

        # ── Adam optimiser state ──────────────────────────────────────────
        self._adam_t   = 0
        self._adam_m   = np.zeros((n_layers, T, 2))
        self._adam_v   = np.zeros((n_layers, T, 2))
        self._adam_b1  = 0.9
        self._adam_b2  = 0.999
        self._adam_eps = 1e-8

        # ── Grover circuit cache (keyed by marked_tuple) ──────────────────
        self._grover_cache: dict = {}

        self._build_vqc_circuit()

    # ──────────────────────────────────────────────────────────────────────
    # DFT construction (UPA Kronecker structure)
    # ──────────────────────────────────────────────────────────────────────

    def _build_upa_dft(self):
        """Build the same UPA DFT matrix as MassiveMIMOSystem._build_dft_matrix."""
        nx = np.arange(self.X); kx = np.arange(self.X)
        Omega_X = (np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / self.X)
                   / np.sqrt(self.X))
        ny = np.arange(self.Y); ky = np.arange(self.Y)
        Omega_Y = (np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / self.Y)
                   / np.sqrt(self.Y))
        self.Omega = np.kron(Omega_X, Omega_Y)   # (A, A)

    # ──────────────────────────────────────────────────────────────────────
    # Quantum circuit primitives
    # ──────────────────────────────────────────────────────────────────────

    def _vqc_ansatz(self, inputs, weights):
        """
        Variational ansatz: AngleEmbedding → (RY + RZ + circular CNOT) × L.
        """
        qml.AngleEmbedding(inputs, wires=range(self.T), rotation='Y')
        for l in range(self.n_layers):
            for i in range(self.T):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(self.T - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.T - 1, 0])

    def _hadamard_layer(self):
        """H^⊗T uniform superposition (eq. 11-12)."""
        for i in range(self.T):
            qml.Hadamard(wires=i)

    def _oracle(self, marked_tuple: tuple):
        """
        Phase-flip oracle O_M (eq. 13).
        For each marked index: X on zero-bits → multi-ctrl-Z → X back.
        """
        for idx in marked_tuple:
            bits    = format(int(idx), f'0{self.T}b')
            x_wires = [i for i, b in enumerate(bits) if b == '0']
            for w in x_wires:
                qml.PauliX(wires=w)
            qml.ctrl(qml.PauliZ,
                     control=list(range(self.T - 1)))(wires=self.T - 1)
            for w in x_wires:
                qml.PauliX(wires=w)

    def _diffusion(self):
        """
        Grover diffusion operator (eq. 14):  Diff = 2|U><U| - I
        where |U> = H^⊗T |0>.  Implementation: H → S_0 → H.
        """
        self._hadamard_layer()
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ,
                 control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        self._hadamard_layer()

    # ──────────────────────────────────────────────────────────────────────
    # Circuit builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_vqc_circuit(self):
        @qml.qnode(self.dev)
        def vqc_circuit(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        self.vqc_circuit = vqc_circuit

    def _get_grover_circuit(self, marked_tuple: tuple):
        """Compile (or retrieve) the pure Grover amplification circuit."""
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        G_iters = self.G

        @qml.qnode(self.dev)
        def grover_circuit():
            self._hadamard_layer()
            for _ in range(G_iters):
                self._oracle(marked_tuple)
                self._diffusion()
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = grover_circuit
        return grover_circuit

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        """
        Beam-domain CSI feature extraction (eq. 7).

        Steps:
          1. Project N onto DFT beam space: dft_proj = |Omega^H @ N|  (A, T)
          2. Extract peak beam power per user:  best = max over beams  (T,)
          3. Normalise to [0, pi] for AngleEmbedding.

        N : (A, T) complex channel matrix
        Returns angles : (T,) real array in [0, pi]
        """
        dft_proj = np.abs(self.Omega.conj().T @ N)   # (A, T)
        best     = np.max(dft_proj, axis=0)           # (T,) dominant beam/user
        return (best / (best.max() + 1e-9)) * np.pi

    def _probs_to_marginals(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute per-user scheduling marginals from joint 2^T distribution.
        marginals[t] = P(user t is scheduled = 1).
        Uses precomputed bit-mask: (N_states, T) @ (N_states,) → (T,).
        """
        return self._bit_mask.T @ probs   # (T,)

    def _log_prob(self, marginals: np.ndarray, theta: np.ndarray) -> float:
        """
        REINFORCE log-probability under factored Bernoulli policy.
        log pi(theta | N) = Σ_t [ theta_t * log(m_t) + (1-theta_t)*log(1-m_t) ]
        """
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    def _identify_marked_states(self, vqc_probs: np.ndarray,
                                 n_schedule: int,
                                 mimo_sys,
                                 N: np.ndarray,
                                 F: np.ndarray) -> tuple:
        k           = min(16, max(4, self.N_states // 4))
        top_indices = np.argsort(vqc_probs)[-k:][::-1]

        marked      = []
        best_reward = -np.inf
        best_theta  = None

        # Vòng 1: Tìm ra reward cao nhất trong các ứng viên
        for idx in top_indices:
            bits       = format(int(idx), f'0{self.T}b')
            theta_cand = np.array([int(b) for b in bits], dtype=int)
            if theta_cand.sum() == 0: continue
            
            sinr  = mimo_sys.compute_sinr(N, F, theta_cand)
            r     = float(np.sum(mimo_sys.instantaneous_rate(sinr)))
            if r > best_reward:
                best_reward = r
                best_theta  = theta_cand

        # Vòng 2: Đánh dấu (Mark) nếu vượt tau HOẶC là ứng viên giỏi nhất
        for idx in top_indices:
            bits       = format(int(idx), f'0{self.T}b')
            theta_cand = np.array([int(b) for b in bits], dtype=int)
            if theta_cand.sum() == 0: continue
            
            sinr  = mimo_sys.compute_sinr(N, F, theta_cand)
            r     = float(np.sum(mimo_sys.instantaneous_rate(sinr)))
            
            # ELITIST MARKING: Luôn đảm bảo Grover hoạt động
            if r >= self.tau or r == best_reward:
                marked.append(int(idx))

        return tuple(sorted(set(marked))), best_theta, best_reward

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def select(self, N: np.ndarray, n_schedule: int,
               mimo_sys=None, F: np.ndarray = None) -> np.ndarray:
        """
        Select a scheduling vector theta ∈ {0,1}^T (Algorithm 1, inference).

        Steps:
          1. VQC encodes CSI → policy distribution over 2^T candidate schedules.
          2. Oracle marks states with reward >= tau on CURRENT channel (N, F).
          3. Grover circuit amplifies probability mass of marked states.
          4. Combine VQC and Grover distributions → per-user marginals.
          5. STOCHASTIC SAMPLING (not argmax): draw n_schedule users without
             replacement proportional to normalised marginals.

        The stochastic step is critical — deterministic argmax collapses the
        policy to a fixed point and prevents exploration.
        """
        inputs    = pnp.array(self._channel_to_angles(N), requires_grad=False)
        vqc_probs = np.array(self.vqc_circuit(inputs, self.weights), dtype=float)

        if F is None and mimo_sys is not None:
            F = mimo_sys.beamforming_vector()

        # ── Oracle: identify marked set M ─────────────────────────────────
        if mimo_sys is not None and F is not None:
            marked_tuple, _, _ = self._identify_marked_states(
                vqc_probs, n_schedule, mimo_sys, N=N, F=F)
        else:
            k = max(1, self.N_states // 4)
            marked_tuple = tuple(sorted(
                int(s) for s in np.argsort(vqc_probs)[-k:]
            ))

        # ── Grover amplitude amplification ────────────────────────────────
        use_grover = (len(marked_tuple) > 0
                      and len(marked_tuple) < self.N_states // 2)

        if use_grover:
            grover_probs = np.array(
                self._get_grover_circuit(marked_tuple)(), dtype=float)
            combined  = vqc_probs * grover_probs
            combined /= combined.sum() + 1e-9
            marginals  = self._probs_to_marginals(combined)
        else:
            marginals = self._probs_to_marginals(vqc_probs)

        # ── Stochastic selection via normalised marginal probabilities ─────
        # This is the quantum "measurement" step — sampling, not argmax.
        marginals_clipped = np.clip(marginals, 1e-9, 1.0 - 1e-9)
        
        # Sampling Bernoulli độc lập (Agent tự quyết định chọn bao nhiêu user)
        theta = np.random.binomial(1, marginals_clipped)
        
        # Fallback: Nếu Agent lỡ tắt hết users, bật user có xác suất cao nhất
        if theta.sum() == 0:
            theta[np.argmax(marginals_clipped)] = 1
            
        return theta

    def update(self, N: np.ndarray, advantage: float, theta: np.ndarray):
        """
        REINFORCE policy-gradient update on VQC weights (Algorithm 1, training).

        Uses the parameter-shift rule for exact quantum gradients:
            d log pi / d w_k ≈ [ log pi(w+π/2) − log pi(w−π/2) ] / 2

        The caller is responsible for computing advantage = reward − baseline
        so that the gradient signal is centred (variance-reduced REINFORCE).

        Total QNode calls per update = 2 × n_layers × T × 2.
        """
        assert theta is not None, "theta must be provided for REINFORCE update."

        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift  = np.pi / 2
        w_raw  = np.array(self.weights, dtype=float)
        grad   = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    w_raw[l, i, k] += shift
                    p_plus  = np.array(
                        self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_plus = self._log_prob(
                        self._probs_to_marginals(p_plus), theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(
                        self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob(
                        self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift
                    # advantage-weighted gradient (variance-reduced REINFORCE)
                    grad[l, i, k]   = advantage * (lp_plus - lp_minus) / 2.0

        # ── Adam update ───────────────────────────────────────────────────
        self._adam_t += 1
        b1, b2, eps   = self._adam_b1, self._adam_b2, self._adam_eps
        self._adam_m   = b1 * self._adam_m + (1 - b1) * grad
        self._adam_v   = b2 * self._adam_v + (1 - b2) * grad ** 2
        m_hat = self._adam_m / (1 - b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.weights = pnp.array(new_w, requires_grad=True)