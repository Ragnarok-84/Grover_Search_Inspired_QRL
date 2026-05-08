import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:
    """
    Grover-inspired QRL scheduler.

    Important implementation choices in this corrected version:
      - the oracle evaluates the full 2^T scheduling space for small T, not only
        the top-k states from the VQC;
      - if n_schedule is provided, both oracle marking and final measurement are
        restricted to bitstrings with exactly n_schedule scheduled users;
      - the oracle marks only states whose instantaneous sum-rate >= tau;
        no elitist fallback is used;
      - the final action is sampled from the joint distribution over bitstrings,
        not independent Bernoulli marginals;
      - REINFORCE uses the joint VQC probability of the measured bitstring.
    """

    def __init__(self, A: int, T: int, n_layers: int = 3,
                 lr: float = 0.02, G: int = 1, tau: float = 4.0,
                 max_oracle_states: int | None = None):
        self.A = A
        self.T = T
        self.n_layers = n_layers
        self.lr = lr
        self.G = G
        self.tau = tau
        self.N_states = 2 ** T
        self.max_oracle_states = max_oracle_states

        self.X = int(np.floor(np.sqrt(A)))
        while A % self.X != 0:
            self.X -= 1
        self.Y = A // self.X
        self._build_upa_dft()

        self.dev = qml.device("default.qubit", wires=T)
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        indices = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1).astype(np.int8)
        self._valid_cache: dict[int | None, np.ndarray] = {}

        self._adam_t = 0
        self._adam_m = np.zeros((n_layers, T, 2))
        self._adam_v = np.zeros((n_layers, T, 2))
        self._adam_b1 = 0.9
        self._adam_b2 = 0.999
        self._adam_eps = 1e-8

        self._grover_cache: dict = {}
        self._last_info: dict = {}
        self._build_vqc_circuit()

    # ──────────────────────────────────────────────────────────────────────
    # DFT construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_upa_dft(self):
        nx = np.arange(self.X); kx = np.arange(self.X)
        Omega_X = np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / self.X) / np.sqrt(self.X)
        ny = np.arange(self.Y); ky = np.arange(self.Y)
        Omega_Y = np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / self.Y) / np.sqrt(self.Y)
        self.Omega = np.kron(Omega_X, Omega_Y)

    # ──────────────────────────────────────────────────────────────────────
    # Quantum primitives
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

    def _hadamard_layer(self):
        for i in range(self.T):
            qml.Hadamard(wires=i)

    def _phase_flip_index(self, idx: int):
        bits = format(int(idx), f'0{self.T}b')
        x_wires = [i for i, b in enumerate(bits) if b == '0']
        for w in x_wires:
            qml.PauliX(wires=w)
        qml.ctrl(qml.PauliZ, control=list(range(self.T - 1)))(wires=self.T - 1)
        for w in x_wires:
            qml.PauliX(wires=w)

    def _oracle(self, marked_tuple: tuple[int, ...]):
        for idx in marked_tuple:
            self._phase_flip_index(idx)

    def _diffusion(self):
        self._hadamard_layer()
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ, control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        self._hadamard_layer()

    def _build_vqc_circuit(self):
        @qml.qnode(self.dev)
        def vqc_circuit(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        self.vqc_circuit = vqc_circuit

    def _get_grover_circuit(self, marked_tuple: tuple[int, ...]):
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        @qml.qnode(self.dev)
        def grover_circuit():
            self._hadamard_layer()
            for _ in range(self.G):
                self._oracle(marked_tuple)
                self._diffusion()
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = grover_circuit
        return grover_circuit

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        dft_proj = np.abs(self.Omega.conj().T @ N)
        best = np.max(dft_proj, axis=0)
        return (best / (best.max() + 1e-9)) * np.pi

    def _theta_to_index(self, theta: np.ndarray) -> int:
        bits = ''.join(str(int(b)) for b in theta)
        return int(bits, 2)

    def _index_to_theta(self, idx: int) -> np.ndarray:
        return np.array([int(b) for b in format(int(idx), f'0{self.T}b')], dtype=int)

    def _valid_indices(self, n_schedule: int | None) -> np.ndarray:
        if n_schedule is None:
            key = None
        else:
            key = int(n_schedule)
        if key not in self._valid_cache:
            if key is None:
                valid = np.arange(1, self.N_states, dtype=int)  # exclude all-zero schedule
            else:
                valid = np.where(self._bit_mask.sum(axis=1) == key)[0].astype(int)
                if valid.size == 0:
                    raise ValueError(f"No bitstrings with n_schedule={n_schedule} for T={self.T}")
            self._valid_cache[key] = valid
        return self._valid_cache[key]

    def _normalise_over_valid(self, probs: np.ndarray, valid: np.ndarray) -> np.ndarray:
        masked = np.zeros_like(probs, dtype=float)
        masked[valid] = np.clip(probs[valid], 0.0, None)
        total = masked.sum()
        if not np.isfinite(total) or total <= 0:
            masked[valid] = 1.0 / len(valid)
        else:
            masked /= total
        return masked

    def _sample_bitstring(self, probs: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
        masked = self._normalise_over_valid(probs, valid)
        idx = int(np.random.choice(np.arange(self.N_states), p=masked))
        return self._index_to_theta(idx), idx

    def _log_prob_joint(self, probs_all: np.ndarray, theta: np.ndarray) -> float:
        idx = self._theta_to_index(theta)
        return float(np.log(np.clip(probs_all[idx], 1e-12, 1.0)))

    def _candidate_indices_for_oracle(self, vqc_probs: np.ndarray, n_schedule: int | None) -> np.ndarray:
        valid = self._valid_indices(n_schedule)
        if self.max_oracle_states is None or self.max_oracle_states >= len(valid):
            return valid
        # For large T only: deterministic pruning by VQC probability.
        order = np.argsort(vqc_probs[valid])[-self.max_oracle_states:]
        return valid[order]

    def _identify_marked_states(
        self,
        vqc_probs: np.ndarray,
        n_schedule: int | None,
        mimo_sys,
        N: np.ndarray,
        F: np.ndarray,
    ) -> tuple[tuple[int, ...], np.ndarray | None, float]:
        """
        Dynamic-percentile oracle marking.

        self.tau is interpreted as a percentile, not an absolute sum-rate.

        Example:
            self.tau = 80.0  -> mark candidates with reward >= 80th percentile
                            -> roughly top 20% candidates are marked.

            self.tau = 90.0  -> mark top 10%.
            self.tau = 75.0  -> mark top 25%.

        Returns:
            marked_tuple : sorted tuple of marked state indices
            best_theta   : best scheduling vector among candidates
            best_reward  : best sum-rate among candidates
        """
        candidates = self._candidate_indices_for_oracle(vqc_probs, n_schedule)

        candidate_rewards: list[tuple[int, float, np.ndarray]] = []
        best_reward = -np.inf
        best_theta = None

        for idx in candidates:
            idx = int(idx)
            theta_cand = self._index_to_theta(idx)

            sinr = mimo_sys.compute_sinr(N, F, theta_cand)
            rates = mimo_sys.instantaneous_rate(sinr)
            sum_rate = float(mimo_sys.compute_sum_rate(rates, theta_cand))

            candidate_rewards.append((idx, sum_rate, theta_cand))

            if sum_rate > best_reward:
                best_reward = sum_rate
                best_theta = theta_cand.copy()

        if len(candidate_rewards) == 0:
            return tuple(), None, float("-inf")

        rewards = np.array([reward for _, reward, _ in candidate_rewards], dtype=float)

        # Interpret self.tau as percentile.
        # Clamp to a safe range so np.percentile does not crash.
        percentile = float(np.clip(self.tau, 0.0, 100.0))
        threshold = float(np.percentile(rewards, percentile))

        marked = [
            int(idx)
            for idx, reward, _ in candidate_rewards
            if reward >= threshold
        ]

        # Safety fallback: ensure Grover has at least one marked state.
        if len(marked) == 0:
            best_idx = max(candidate_rewards, key=lambda x: x[1])[0]
            marked = [int(best_idx)]

        return tuple(sorted(set(marked))), best_theta, float(best_reward)
        
    
    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def select(self, N: np.ndarray, n_schedule: int | None,
               mimo_sys=None, F: np.ndarray | None = None) -> np.ndarray:
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        vqc_probs = np.array(self.vqc_circuit(inputs, self.weights), dtype=float)
        valid = self._valid_indices(n_schedule)

        if F is None and mimo_sys is not None:
            F = mimo_sys.beamforming_vector()

        if mimo_sys is not None and F is not None:
            marked_tuple, best_theta, best_reward = self._identify_marked_states(
                vqc_probs, n_schedule, mimo_sys, N=N, F=F)
        else:
            # No system/oracle available: no reward-based marking.
            marked_tuple, best_theta, best_reward = tuple(), None, np.nan

        use_grover = 0 < len(marked_tuple) < self.N_states // 2
        if use_grover:
            grover_probs = np.array(self._get_grover_circuit(marked_tuple)(), dtype=float)
            combined = vqc_probs * grover_probs
            combined = self._normalise_over_valid(combined, valid)
        else:
            combined = self._normalise_over_valid(vqc_probs, valid)

        theta, idx = self._sample_bitstring(combined, valid)
        self._last_info = {
            "marked_count": len(marked_tuple),
            "used_grover": use_grover,
            "sampled_index": idx,
            "best_oracle_theta": best_theta,
            "best_oracle_sum_rate": best_reward,
        }
        return theta

    def update(self, N: np.ndarray, advantage: float, theta: np.ndarray):
        assert theta is not None, "theta must be provided for REINFORCE update."
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift = np.pi / 2
        w_raw = np.array(self.weights, dtype=float)
        grad = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    w_raw[l, i, k] += shift
                    p_plus = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_plus = self._log_prob_joint(p_plus, theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob_joint(p_minus, theta)

                    w_raw[l, i, k] += shift
                    grad[l, i, k] = advantage * (lp_plus - lp_minus) / 2.0

        self._adam_t += 1
        b1, b2, eps = self._adam_b1, self._adam_b2, self._adam_eps
        self._adam_m = b1 * self._adam_m + (1 - b1) * grad
        self._adam_v = b2 * self._adam_v + (1 - b2) * grad ** 2
        m_hat = self._adam_m / (1 - b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.weights = pnp.array(new_w, requires_grad=True)
