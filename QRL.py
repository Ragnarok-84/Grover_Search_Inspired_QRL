import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:
    """
    Grover-inspired QRL agent for mMIMO user scheduling.

    FIXES vs original:
      FIX 1: Correct circuit order — H^⊗N first (uniform superposition),
              then Oracle, then Diffusion. VQC is used ONLY for state-prep
              in a separate circuit, not mixed into the Grover loop.
      FIX 2: Oracle marks states by REWARD > tau, not by probability > tau.
              _identify_marked_states() now takes reward as input.
      FIX 3: select() and update() both operate on the VQC circuit (policy).
              Grover amplification is applied on top of VQC output in select().
      FIX 4: Default tau set to a meaningful positive value; G=1 kept.

    Architecture (per paper Fig. 1, Algorithm 1):
      - VQC circuit: AngleEmbed(CSI) + [RY+RZ+CNOT-ring]×n_layers → policy probs
      - Grover circuit: H^⊗N → [Oracle(M) → Diffusion]×G → amplified probs
        where M is determined by evaluating reward on candidate schedules.
    """

    def __init__(self, A: int, T: int, n_layers: int = 3,
                 lr: float = 0.02, G: int = 1, tau: float = 0.5):
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.G        = G
        self.tau      = tau
        self.N_states = 2 ** T

        self.dev = qml.device("default.qubit", wires=T)

        # Trainable VQC weights: (n_layers, T, 2)  [RY, RZ per qubit/layer]
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Precompute bit-mask for fast marginal calculation
        indices        = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                          ).astype(np.float32)          # (N_states, T)

        # Adam optimiser state
        self._adam_t   = 0
        self._adam_m   = np.zeros((n_layers, T, 2))
        self._adam_v   = np.zeros((n_layers, T, 2))
        self._adam_b1  = 0.9
        self._adam_b2  = 0.999
        self._adam_eps = 1e-8

        # Cache: marked_tuple → compiled Grover QNode
        self._grover_cache: dict = {}

        self._build_vqc_circuit()
        self._build_hadamard_only_circuit()

    # ══════════════════════════════════════════════════════════════════════
    # CIRCUIT BUILDING BLOCKS
    # ══════════════════════════════════════════════════════════════════════

    def _vqc_ansatz(self, inputs, weights):
        """
        VQC: AngleEmbedding + [RY, RZ, CNOT-ring] × n_layers.
        Encodes CSI into trainable qubit rotations.
        This is the POLICY network — trained via REINFORCE.
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
        """H^⊗N — creates uniform superposition (eq. 11-12)."""
        for i in range(self.T):
            qml.Hadamard(wires=i)

    def _oracle(self, marked_tuple):
        """
        FIX 1 + 2: Oracle O_M (eq. 13).
        Phase-flip states in marked_tuple.
        marked_tuple contains state INDICES with high REWARD (not high prob).

        The Grover circuit starts from |0⟩^⊗N, so oracle acts on the
        uniform superposition produced by H^⊗N.
        """
        for idx in marked_tuple:
            bitstring = format(idx, f'0{self.T}b')
            x_wires = [i for i, b in enumerate(bitstring) if b == '0']
            for w in x_wires:
                qml.PauliX(wires=w)
            qml.ctrl(qml.PauliZ,
                     control=list(range(self.T - 1)))(wires=self.T - 1)
            for w in x_wires:
                qml.PauliX(wires=w)

    def _diffusion(self):
        """
        Grover diffusion D = H^⊗N · S₀ · H^⊗N (eq. 14).
        |U⟩ = H^⊗N|0⟩ is the FIXED uniform state — never changes.
        """
        self._hadamard_layer()
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ,
                 control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        self._hadamard_layer()

    # ══════════════════════════════════════════════════════════════════════
    # FULL CIRCUITS
    # ══════════════════════════════════════════════════════════════════════

    def _build_vqc_circuit(self):
        """
        FIX 3: VQC circuit = policy network (trained via REINFORCE).
        Circuit: AngleEmbed(CSI) + [RY+RZ+CNOT]×n_layers → probs.
        Used in update() for gradient computation.
        Also used in select() to get initial policy probs.
        """
        @qml.qnode(self.dev)
        def vqc_circuit(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        self.vqc_circuit = vqc_circuit

    def _build_hadamard_only_circuit(self):
        """
        Pure Grover circuit WITHOUT VQC: H^⊗N → [Oracle→Diffusion]×G.
        FIX 1: Hadamard FIRST — creates uniform superposition per eq. 11-12.
        The VQC output is used to DETERMINE which states to mark (M),
        but the Grover amplification itself starts fresh from |0⟩^⊗N.
        """
        pass  # built on demand via _get_grover_circuit()

    def _get_grover_circuit(self, marked_tuple: tuple):
        """
        FIX 1: Correct Grover circuit.
        H^⊗N → [Oracle(M) → Diffusion] × G → probs.

        Note: this circuit takes NO VQC inputs/weights — it is a pure
        Grover search circuit. The marked set M comes from the VQC policy
        evaluation in select(), not from trainable parameters here.
        """
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        G_iters = self.G

        @qml.qnode(self.dev)
        def grover_circuit():
            # FIX 1: Start with uniform superposition (eq. 11-12)
            self._hadamard_layer()
            # Grover iterations: Oracle + Diffusion (eq. 13-14)
            for _ in range(G_iters):
                self._oracle(marked_tuple)
                self._diffusion()
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = grover_circuit
        return grover_circuit

    # ══════════════════════════════════════════════════════════════════════
    # UTILITY HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        """
        DFT beam-domain feature extraction.
        Projects channel onto beam space, picks strongest beam per user.
        Normalised to [0, π] for AngleEmbedding.
        """
        k = np.arange(self.A)
        n = np.arange(self.A)
        Omega = np.exp(-1j * 2 * np.pi * np.outer(k, n) / self.A) / np.sqrt(self.A)
        dft_proj     = np.abs(Omega.conj().T @ N)      # (A, T)
        best_beam    = np.max(dft_proj, axis=0)        # (T,)
        return (best_beam / (best_beam.max() + 1e-9)) * np.pi

    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        """Per-qubit marginal P(qubit t = 1) from full state distribution."""
        return self._bit_mask.T @ probs_all             # (T,)

    def _log_prob_from_marginals(self, marginals: np.ndarray,
                                  theta: np.ndarray) -> float:
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    def _vqc_probs_to_theta(self, probs_all: np.ndarray,
                             n_schedule: int) -> np.ndarray:
        """Select top-n_schedule users by marginal probability."""
        marginals = self._probs_to_marginals(probs_all)
        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    def _evaluate_reward_for_states(self, probs_all: np.ndarray,
                                     n_schedule: int,
                                     mimo_sys) -> tuple:
        """
        FIX 2: Evaluate REWARD for candidate schedules to identify M.

        For efficiency, we evaluate the top-k states by VQC probability
        (k = min(8, N_states//4)) and mark those with reward > tau.

        Returns (marked_tuple, best_theta, best_reward).
        """
        # Sort states by VQC probability — top candidates
        k = min(8, max(2, self.N_states // 8))
        top_indices = np.argsort(probs_all)[-k:][::-1]

        marked = []
        best_reward = -np.inf
        best_theta  = None

        for idx in top_indices:
            # Decode state index to scheduling vector
            bitstring = format(int(idx), f'0{self.T}b')
            theta_cand = np.array([int(b) for b in bitstring], dtype=int)

            # Only consider valid schedules (exactly n_schedule users)
            if theta_cand.sum() != n_schedule:
                continue

            # FIX 2: evaluate actual reward from MIMO system
            if mimo_sys is not None:
                N  = mimo_sys.generate_channel()
                F  = mimo_sys.beamforming_vector(K_avg=3)
                sinr  = mimo_sys.compute_sinr(N, F, theta_cand)
                rates = mimo_sys.instantaneous_rate(sinr)
                r = mimo_sys.compute_pf_reward(rates, theta_cand)
            else:
                # Fallback: use log-probability as proxy (no MIMO system)
                marginals = self._probs_to_marginals(probs_all)
                r = self._log_prob_from_marginals(marginals, theta_cand)

            if r > best_reward:
                best_reward = r
                best_theta  = theta_cand

            if r >= self.tau:
                marked.append(int(idx))

        return tuple(sorted(marked)), best_theta, best_reward

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def select(self, N: np.ndarray, n_schedule: int,
               mimo_sys=None) -> np.ndarray:
        """
        Select a scheduling vector θ ∈ {0,1}^T.

        Algorithm 1 (corrected):
          1. VQC encodes CSI → policy probability distribution.
          2. Evaluate reward for top candidate states → identify M (FIX 2).
          3. If |M| valid: run Grover circuit H^⊗N→[O_M→Diff]×G (FIX 1).
             Combine: argmax over Grover-amplified probs restricted to
             states that also have high VQC probability.
          4. Else: use VQC marginals directly (pure policy).

        Parameters
        ----------
        N         : (A×T) channel matrix
        n_schedule: number of users to schedule
        mimo_sys  : MassiveMIMOSystem instance (for reward evaluation)
                    If None, falls back to probability-based selection.
        """
        inputs    = pnp.array(self._channel_to_angles(N), requires_grad=False)
        vqc_probs = np.array(self.vqc_circuit(inputs, self.weights), dtype=float)

        # FIX 2: identify marked states by reward evaluation
        if mimo_sys is not None:
            marked_tuple, best_theta, _ = self._evaluate_reward_for_states(
                vqc_probs, n_schedule, mimo_sys)
        else:
            # Fallback without MIMO system: mark top-probability states
            k = max(1, self.N_states // 4)
            marked_tuple = tuple(sorted(
                int(s) for s in np.argsort(vqc_probs)[-k:]
            ))
            best_theta = None

        # FIX 1: Grover amplification from uniform superposition
        use_grover = (len(marked_tuple) > 0
                      and len(marked_tuple) < self.N_states // 2)

        if use_grover:
            circ       = self._get_grover_circuit(marked_tuple)
            grover_probs = np.array(circ(), dtype=float)
            # Combine VQC policy with Grover amplification
            combined   = vqc_probs * grover_probs
            combined  /= combined.sum() + 1e-9
            marginals  = self._probs_to_marginals(combined)
        else:
            marginals = self._probs_to_marginals(vqc_probs)

        # Pick top-n_schedule users
        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1

        # If we found a valid best_theta from reward eval, prefer it
        if best_theta is not None and best_theta.sum() == n_schedule:
            return best_theta
        return theta

    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        FIX 3: REINFORCE policy-gradient update on the VQC circuit.
        Parameter-shift rule on vqc_circuit (policy network).

        The VQC is the trainable component; Grover amplification is
        a fixed search procedure applied on top of the trained policy.

        Total QNode evaluations = 2 × n_layers × T × 2.
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
                    p_plus   = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_plus  = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_plus), theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift
                    grad[l, i, k]   = reward * (lp_plus - lp_minus) / 2.0

        # Adam update
        self._adam_t += 1
        self._adam_m  = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v  = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

        self.weights = pnp.array(new_w, requires_grad=True)