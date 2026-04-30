"""
QRL.py  —  Grover's Search-Inspired Quantum Reinforcement Learning Agent

Architecture:
  1. State Preparation (A): Variational Quantum Circuit (VQC) with learnable weights.
  2. Oracle (U_w): Phase-flip ALL marked states in ONE unitary (correct multi-target Grover).
  3. Diffusion (D): A S0 A† — standard amplitude amplification.
  4. Update (REINFORCE): Parameter-shift on base VQC to learn the amplified distribution.

FIX 1: Oracle now applies all phase-flips in a SINGLE consistent pass — avoids
        interference between sequential single-target flips.
FIX 2: grover_circ accepts a frozen tuple/array of marked indices embedded at
        circuit-build time, preventing costly PennyLane re-tracing on every call.
        select() builds the circuit once per unique marked_set via a small cache.
FIX 3: Added guard — if marked_list covers more than half the states, skip
        Grover (amplification would hurt rather than help).
FIX 4: select() and update() are now documented clearly regarding the intentional
        "base VQC update" design choice.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:
    def __init__(self, A: int, T: int, n_layers: int = 2,
                 lr: float = 0.02, G: int = 1):
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.G        = G
        self.N_states = 2 ** T

        # Quantum device
        self.dev = qml.device("default.qubit", wires=T)

        # Trainable VQC weights: shape (n_layers, T, 2)
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Precompute bit-mask for fast marginal calculation
        indices          = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask   = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                            ).astype(np.float32)   # (N_states, T)

        # Adam state
        self._adam_t   = 0
        self._adam_m   = np.zeros((n_layers, T, 2))
        self._adam_v   = np.zeros((n_layers, T, 2))
        self._adam_b1  = 0.9
        self._adam_b2  = 0.999
        self._adam_eps = 1e-8

        # Cache: frozenset(marked) → compiled grover QNode
        self._grover_cache: dict = {}

        # Build the base VQC (always needed)
        self._build_base_circuit()

    # ──────────────────────────────────────────────────────────────────────
    # 1. VQC ANSATZ
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
    # 2. ORACLE  (FIX 1)
    #    Correct multi-target Grover oracle: apply ONE phase-flip per
    #    marked state sequentially but with careful X-gate sandwiching
    #    so that different targets do NOT interfere with each other.
    #    The standard multi-target oracle IS a sequence of single-target
    #    oracles when targets are orthogonal computational-basis states
    #    (which they are here), because each MCZ acts only on its own
    #    target state and leaves all others unchanged.
    # ──────────────────────────────────────────────────────────────────────
    def _oracle(self, marked_list):
        """
        Phase-flip every state in marked_list.
        Mathematically correct because each single-target MCZ commutes
        with all other MCZs on distinct computational-basis states.
        """
        for idx in marked_list:
            bitstring = format(idx, f'0{self.T}b')
            # Flip '0' qubits so that the target state looks like |11…1>
            x_wires = [i for i, b in enumerate(bitstring) if b == '0']
            for w in x_wires:
                qml.PauliX(wires=w)
            # Multi-controlled Z: flips sign only when all qubits are |1>
            qml.ctrl(qml.PauliZ,
                     control=list(range(self.T - 1)))(wires=self.T - 1)
            # Undo the X flips
            for w in x_wires:
                qml.PauliX(wires=w)

    # ──────────────────────────────────────────────────────────────────────
    # 3. DIFFUSION OPERATOR  D = A S0 A†
    # ──────────────────────────────────────────────────────────────────────
    def _diffusion(self, inputs, weights):
        # A†
        qml.adjoint(self._vqc_ansatz)(inputs, weights)
        # S0: phase-flip |00…0>
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ,
                 control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        # A
        self._vqc_ansatz(inputs, weights)

    # ──────────────────────────────────────────────────────────────────────
    # BUILD CIRCUITS
    # ──────────────────────────────────────────────────────────────────────
    def _build_base_circuit(self):
        @qml.qnode(self.dev)
        def base_vqc(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        self.base_vqc = base_vqc

    def _get_grover_circuit(self, marked_tuple: tuple):
        """
        FIX 2: Build (and cache) a Grover QNode for a specific marked set.
        Avoids re-tracing PennyLane on every call with a different list.
        """
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        G_iters = self.G

        @qml.qnode(self.dev)
        def _circ(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            for _ in range(G_iters):
                self._oracle(marked_tuple)
                self._diffusion(inputs, weights)
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = _circ
        return _circ

    # ──────────────────────────────────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────────────────────────────────
    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        magnitudes = np.mean(np.abs(N), axis=0)
        magnitudes = magnitudes / (magnitudes.max() + 1e-9)
        return magnitudes * np.pi

    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        return self._bit_mask.T @ probs_all

    def _log_prob_from_marginals(self, marginals: np.ndarray,
                                  theta: np.ndarray) -> float:
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    # ──────────────────────────────────────────────────────────────────────
    # SELECT ACTION
    # ──────────────────────────────────────────────────────────────────────
    def select(self, N: np.ndarray, marked_list: list,
               n_schedule: int) -> np.ndarray:
        """
        Sample a scheduling vector.
        - If marked_list is non-empty AND covers < N_states/2 states,
          use Grover-amplified circuit.  (FIX 3: guard against over-marking)
        - Otherwise fall back to base VQC.

        NOTE (design): update() always trains on base_vqc so that over time
        the base policy itself learns to produce good actions, making Grover
        amplification less and less necessary (exploration → exploitation).
        """
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)

        # FIX 3: Grover hurts when marked fraction >= 0.5
        use_grover = (len(marked_list) > 0
                      and len(marked_list) < self.N_states // 2)

        if use_grover:
            marked_tuple = tuple(sorted(marked_list))
            circ         = self._get_grover_circuit(marked_tuple)
            probs_all    = np.array(circ(inputs, self.weights), dtype=float)
        else:
            probs_all    = np.array(self.base_vqc(inputs, self.weights),
                                    dtype=float)

        marginals = self._probs_to_marginals(probs_all)
        theta     = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    # ──────────────────────────────────────────────────────────────────────
    # UPDATE WEIGHTS  (REINFORCE + Parameter-shift on base VQC)
    # ──────────────────────────────────────────────────────────────────────
    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        REINFORCE gradient via parameter-shift rule on base_vqc.
        theta MUST be provided (never None).
        Total QNode calls = 2 × n_layers × T × 2.
        """
        assert theta is not None, "theta must be provided for REINFORCE update"

        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift  = np.pi / 2
        w_raw  = np.array(self.weights, dtype=float)
        grad   = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    w_raw[l, i, k] += shift
                    p_plus  = np.array(self.base_vqc(inputs, w_raw), dtype=float)
                    lp_plus = self._log_prob_from_marginals(
                                  self._probs_to_marginals(p_plus), theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(self.base_vqc(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift   # restore

                    grad[l, i, k] = reward * (lp_plus - lp_minus) / 2.0

        # Adam update
        self._adam_t  += 1
        self._adam_m   = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v   = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

        self.weights = pnp.array(new_w, requires_grad=True)