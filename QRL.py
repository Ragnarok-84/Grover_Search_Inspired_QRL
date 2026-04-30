import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:
    """
    Grover-inspired QRL agent for mMIMO user scheduling.

    Parameters
    ----------
    A         : int    number of BS antennas (used for channel→angle mapping)
    T         : int    number of users / qubits
    n_layers  : int     VQC depth (RY+RZ+CNOT blocks)
    lr        : float   Adam learning rate
    G         : int    Grover iterations per select() call (Algorithm 1, K)
    tau       : float   oracle reward threshold τ (Algorithm 1, line 7)
    """

    def __init__(self, A: int, T: int, n_layers: int = 2,
                 lr: float = 0.02, G: int = 1, tau: float = 0.0):
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.G        = G
        self.tau      = tau          # oracle threshold  (eq. Algorithm 1, line 7)
        self.N_states = 2 ** T

        # Quantum device — T wires, one per user (Fig. 2)
        self.dev = qml.device("default.qubit", wires=T)

        # Trainable VQC weights: shape (n_layers, T, 2)  [RY, RZ per qubit/layer]
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Precompute bit-mask for fast marginal calculation
        # _bit_mask[s, t] = 1 if qubit t is |1⟩ in state index s
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

        # Build the base circuit used for inference & training
        self._build_base_circuit()

    # ══════════════════════════════════════════════════════════════════════
    # CIRCUIT BUILDING BLOCKS
    # ══════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────
    # A.  VQC ANSATZ  —  channel-conditioned state preparation
    #     AngleEmbedding encodes normalised CSI magnitudes as RY angles,
    #     followed by trainable RY+RZ rotations and circular CNOT entanglers.
    #     This is applied BEFORE the Hadamard layer so that the uniform
    #     superposition is seeded with channel information.
    # ──────────────────────────────────────────────────────────────────────
    def _vqc_ansatz(self, inputs, weights):
        """
        VQC: AngleEmbedding(inputs) + [RY, RZ, CNOT-ring] × n_layers.
        Encodes CSI into qubit rotations before the Hadamard superposition.
        """
        # Channel-state embedding (eq. 5 / Algorithm 1: state S)
        qml.AngleEmbedding(inputs, wires=range(self.T), rotation='Y')
        for l in range(self.n_layers):
            for i in range(self.T):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            # Circular CNOT entanglement ring (Fig. 2)
            for i in range(self.T - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.T - 1, 0])

    # ──────────────────────────────────────────────────────────────────────
    # B.  HADAMARD LAYER  —  uniform superposition  (eq. 11–12, Fig. 2)
    #     |ψ₁⟩ = H^⊗N |ψ₀⟩ = (1/√2^N) Σ_{i=0}^{2^N−1} |θ⟩
    #     This is Layer 1 of the paper's architecture (Fig. 1).
    # ──────────────────────────────────────────────────────────────────────
    def _hadamard_layer(self):
        """Apply H to every qubit → uniform superposition (eq. 12)."""
        for i in range(self.T):
            qml.Hadamard(wires=i)

    # ──────────────────────────────────────────────────────────────────────
    # C.  ORACLE  O_M  (eq. 13, Fig. 2 — Oracle Layer)
    #     O_M|θ⟩ = -|θ⟩  for θ ∈ M  (marked high-reward states)
    #            =  |θ⟩  for θ ∉ M
    #
    #     Implementation: for each marked index, sandwich a multi-controlled-Z
    #     with PauliX on the '0' qubits of that basis state so the MCZ fires
    #     only on |11…1⟩ → phase flip of the desired state.
    #     Sequential single-target MCZs commute on distinct basis states,
    #     so the order is irrelevant (correct multi-target oracle).
    # ──────────────────────────────────────────────────────────────────────
    def _oracle(self, marked_tuple):
        """
        Phase-flip every state index in marked_tuple (eq. 13).

        Parameters
        ----------
        marked_tuple : tuple of int  —  indices of high-reward states in M
        """
        for idx in marked_tuple:
            bitstring = format(idx, f'0{self.T}b')
            # X-gate on '0' qubits so target state appears as |11…1⟩
            x_wires = [i for i, b in enumerate(bitstring) if b == '0']
            for w in x_wires:
                qml.PauliX(wires=w)
            # Multi-controlled-Z: phase flip only |11…1⟩
            qml.ctrl(qml.PauliZ,
                     control=list(range(self.T - 1)))(wires=self.T - 1)
            # Restore X-gate flips
            for w in x_wires:
                qml.PauliX(wires=w)

    # ──────────────────────────────────────────────────────────────────────
    # D.  DIFFUSION OPERATOR  Diff = 2|U⟩⟨U| - I  (eq. 14, Fig. 2)
    #     |U⟩ = H^⊗N|0⟩  (uniform state from eq. 12)
    #
    #     Standard Grover decomposition:
    #       Diff = H^⊗N · S₀ · H^⊗N
    #     where S₀ = 2|0⟩⟨0| - I  (phase-flip of |00…0⟩).
    #
    #     CORRECTION vs. previous code:
    #       Old: A · S₀ · A†  with  A = VQC(inputs, weights)
    #            → reflection centre changes every training step; breaks
    #              Grover's inversion-about-the-mean guarantee.
    #       New: H^⊗N · S₀ · H^⊗N
    #            → reflection centre is always the fixed uniform state |U⟩
    #              exactly as stated in eq. 14 and Fig. 1 of the paper.
    # ──────────────────────────────────────────────────────────────────────
    def _diffusion(self):
        """
        Grover diffusion operator D = H^⊗N · S₀ · H^⊗N  (eq. 14).
        Inversion about the uniform state |U⟩ = H^⊗N|0⟩.
        """
        # H^⊗N
        self._hadamard_layer()
        # S₀: phase-flip |00…0⟩
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ,
                 control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        # H^⊗N
        self._hadamard_layer()

    # ══════════════════════════════════════════════════════════════════════
    # FULL CIRCUITS
    # ══════════════════════════════════════════════════════════════════════

    def _build_base_circuit(self):
        """
        Base circuit (no Grover amplification):
          VQC_ansatz → H^⊗N → measure probs

        Used for:
          • Inference when no marked states exist or |M| ≥ N_states/2.
          • REINFORCE parameter-shift training (update()).

        The Hadamard layer after the VQC maps the channel-conditioned
        rotations into the same Hilbert-space superposition that the
        Grover circuit uses, keeping training and inference consistent.
        """
        @qml.qnode(self.dev)
        def base_circuit(inputs, weights):
            # Layer 0: channel-conditioned state preparation (VQC)
            self._vqc_ansatz(inputs, weights)
            # Layer 1: Hadamard superposition (eq. 11–12)
            self._hadamard_layer()
            return qml.probs(wires=range(self.T))

        self.base_circuit = base_circuit

    def _get_grover_circuit(self, marked_tuple: tuple):
        """
        Build (and cache) a Grover QNode for a specific marked set.

        Full circuit per Algorithm 1:
          VQC_ansatz → H^⊗N → [Oracle → Diffusion] × G → measure probs

        Caching avoids PennyLane re-tracing for the same marked set.
        marked_tuple is a sorted, frozen tuple so it is hashable.
        """
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        G_iters = self.G

        @qml.qnode(self.dev)
        def grover_circuit(inputs, weights):
            # Layer 0: channel-conditioned state preparation (VQC)
            self._vqc_ansatz(inputs, weights)
            # Layer 1: Hadamard — uniform superposition (eq. 12)
            self._hadamard_layer()
            # Layers 2+3: Oracle + Diffusion repeated G times (Algorithm 1)
            for _ in range(G_iters):
                self._oracle(marked_tuple)   # eq. 13
                self._diffusion()            # eq. 14
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = grover_circuit
        return grover_circuit

    # ══════════════════════════════════════════════════════════════════════
    # UTILITY HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        """
        Convert channel matrix N (A×T) to T embedding angles ∈ [0, π].
        Normalise mean absolute magnitude per user to [0,1] then scale by π.
        """
        magnitudes = np.mean(np.abs(N), axis=0)              # (T,)
        magnitudes = magnitudes / (magnitudes.max() + 1e-9)
        return magnitudes * np.pi

    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        """
        Compute per-qubit marginal P(qubit t = 1) from full state probabilities.
        marginals[t] = Σ_{s: bit t of s = 1} probs_all[s]
        """
        return self._bit_mask.T @ probs_all                  # (T,)

    def _log_prob_from_marginals(self, marginals: np.ndarray,
                                  theta: np.ndarray) -> float:
        """
        Log-probability of scheduling vector theta under Bernoulli marginals.
        log P(θ) = Σ_t [ θ_t · log p_t + (1−θ_t) · log(1−p_t) ]
        """
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    def _identify_marked_states(self, probs_all: np.ndarray) -> tuple:
        """
        Oracle threshold logic — Algorithm 1, lines 6–9.

        A state index s is marked (placed in M) if its probability
        exceeds self.tau. This approximates "R ≥ τ": states that the
        current policy already assigns high probability are treated as
        high-reward candidates for amplification.

        Returns a sorted tuple of marked indices (hashable for cache).
        """
        marked = tuple(
            int(s) for s in np.where(probs_all >= self.tau)[0]
        )
        return marked

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        """
        Select a scheduling vector θ ∈ {0,1}^T for channel state N.

        Algorithm 1 (simplified per-step):
          1. Encode channel N as VQC input angles.
          2. Run base_circuit to get initial probability distribution.
          3. Oracle threshold (lines 6–9): identify M = {s : p_s ≥ τ}.
          4. If |M| ∈ (0, N_states/2): run Grover-amplified circuit.
             Else: use base_circuit output directly.
          5. Select top-n_schedule users by marginal probability.

        Parameters
        ----------
        N          : np.ndarray (A×T) — channel matrix for this time slot
        n_schedule : int              — number of users to schedule

        Returns
        -------
        theta : np.ndarray (T,) with exactly n_schedule ones
        """
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)

        # Step 2: base probability distribution (VQC + H^⊗N)
        base_probs = np.array(self.base_circuit(inputs, self.weights),
                              dtype=float)

        # Step 3: oracle threshold — identify high-reward states M (Algorithm 1, line 7)
        marked_tuple = self._identify_marked_states(base_probs)

        # Step 4: Grover amplification guard (|M| < N_states/2 required)
        use_grover = (len(marked_tuple) > 0
                      and len(marked_tuple) < self.N_states // 2)

        if use_grover:
            # Grover circuit: VQC → H^⊗N → [O_M → Diff]×G  (Algorithm 1, step 10)
            circ      = self._get_grover_circuit(marked_tuple)
            probs_all = np.array(circ(inputs, self.weights), dtype=float)
        else:
            probs_all = base_probs

        # Step 5: pick top-n_schedule users by marginal P(qubit t = 1)
        marginals = self._probs_to_marginals(probs_all)
        theta     = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        REINFORCE policy-gradient update via parameter-shift rule.

        Gradient (Algorithm 1, amplitude amplification step):
          ∇_w log P(θ|w) = [ log P(θ|w+π/2·e_k) − log P(θ|w−π/2·e_k) ] / 2
          ∇_w J ≈ reward × ∇_w log P(θ|w)

        Gradients are computed on base_circuit (VQC + H^⊗N) so that
        the underlying policy distribution is trained directly, making
        the Grover amplification progressively less necessary as the
        base policy improves (exploration → exploitation, Fig. 3).

        Total QNode evaluations = 2 × n_layers × T × 2.

        Parameters
        ----------
        N      : np.ndarray (A×T) — channel matrix used when theta was chosen
        reward : float            — PF reward received (R in Algorithm 1)
        theta  : np.ndarray (T,)  — scheduling vector that was executed
        """
        assert theta is not None, "theta must be provided for REINFORCE update."

        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift  = np.pi / 2
        w_raw  = np.array(self.weights, dtype=float)
        grad   = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    # Forward shift: w_k ← w_k + π/2
                    w_raw[l, i, k] += shift
                    p_plus   = np.array(self.base_circuit(inputs, w_raw),
                                        dtype=float)
                    lp_plus  = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_plus), theta)

                    # Backward shift: w_k ← w_k − π/2
                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(self.base_circuit(inputs, w_raw),
                                        dtype=float)
                    lp_minus = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_minus), theta)

                    # Restore weight
                    w_raw[l, i, k] += shift

                    # REINFORCE × parameter-shift gradient
                    grad[l, i, k] = reward * (lp_plus - lp_minus) / 2.0

        # Adam update
        self._adam_t += 1
        self._adam_m  = (self._adam_b1 * self._adam_m
                         + (1 - self._adam_b1) * grad)
        self._adam_v  = (self._adam_b2 * self._adam_v
                         + (1 - self._adam_b2) * grad ** 2)
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

        self.weights = pnp.array(new_w, requires_grad=True)