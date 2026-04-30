"""
QRL.py  —  Grover-inspired Quantum Reinforcement Learning Agent
===============================================================
Implements Algorithm 1 and Fig. 1 from arXiv:2601.20688v1.

Architecture (paper Fig. 1):
  Layer 1 — Hadamard layer:   H^⊗N creates uniform superposition (eq. 11-12)
  Layer 2 — Oracle layer:     O_M phase-flips high-reward states (eq. 13)
  Layer 3 — Diffusion layer:  Diff amplifies marked states (eq. 14)
  Repeated G times per decision step.

The VQC (variational quantum circuit) encodes CSI via AngleEmbedding and
serves as the trainable policy network, updated via REINFORCE + parameter-shift.
Grover amplification is applied on top of the VQC output at inference time.

BUGS FIXED vs previous version:
  FIX 1 — select() no longer returns best_theta prematurely.
           Old code short-circuited after greedy reward eval, bypassing Grover.
           Now always selects from Grover-amplified combined distribution.

  FIX 2 — _identify_marked_states() receives N and F from select()
           instead of calling generate_channel() internally.
           Old code evaluated oracle on a DIFFERENT channel realisation,
           making Grover amplify the wrong states entirely.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QRLAgent:

    def __init__(self, A: int, T: int, n_layers: int = 3,
                 lr: float = 0.02, G: int = 1, tau: float = 4.0):
        """
        Parameters
        ----------
        A        : number of BS antennas
        T        : number of users
        n_layers : VQC depth
        lr       : Adam learning rate
        G        : Grover iterations per decision step
        tau      : reward threshold for oracle marking (bps/Hz units)
        """
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.G        = G
        self.tau      = tau
        self.N_states = 2 ** T

        self.dev = qml.device("default.qubit", wires=T)

        # Trainable VQC weights shape: (n_layers, T, 2) — one RY + one RZ per qubit
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Bit-mask (N_states x T): bit_mask[s, t] = t-th bit of state index s
        indices        = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                          ).astype(np.float32)

        # Adam state
        self._adam_t   = 0
        self._adam_m   = np.zeros((n_layers, T, 2))
        self._adam_v   = np.zeros((n_layers, T, 2))
        self._adam_b1  = 0.9
        self._adam_b2  = 0.999
        self._adam_eps = 1e-8

        # Cache compiled Grover QNodes by marked_tuple
        self._grover_cache: dict = {}

        self._build_vqc_circuit()

    # =====================================================================
    # QUANTUM CIRCUIT PRIMITIVES
    # =====================================================================

    def _vqc_ansatz(self, inputs, weights):
        """
        Variational Quantum Circuit ansatz (policy network).
        AngleEmbedding encodes beam-domain CSI features.
        Entangling layers: RY + RZ rotations + circular CNOT ring.
        """
        qml.AngleEmbedding(inputs, wires=range(self.T), rotation='Y')
        for l in range(self.n_layers):
            for i in range(self.T):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(self.T - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.T - 1, 0])   # closing ring

    def _hadamard_layer(self):
        """H^N — uniform superposition over all 2^T states (eq. 11-12)."""
        for i in range(self.T):
            qml.Hadamard(wires=i)

    def _oracle(self, marked_tuple):
        """
        Oracle O_M (eq. 13): phase-flip each state in marked_tuple.
        Implementation: Pauli-X on zero-bits -> multi-controlled Z -> Pauli-X back.
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
        Grover diffusion operator (eq. 14): Diff = 2|U><U| - I
        where |U> = H^N|0> is the uniform state.
        Implementation: H^N -> S_0 (phase-flip |0>) -> H^N
        """
        self._hadamard_layer()
        for i in range(self.T):
            qml.PauliX(wires=i)
        qml.ctrl(qml.PauliZ,
                 control=list(range(self.T - 1)))(wires=self.T - 1)
        for i in range(self.T):
            qml.PauliX(wires=i)
        self._hadamard_layer()

    # =====================================================================
    # FULL CIRCUITS
    # =====================================================================

    def _build_vqc_circuit(self):
        """Build and store the VQC QNode (trainable policy circuit)."""
        @qml.qnode(self.dev)
        def vqc_circuit(inputs, weights):
            self._vqc_ansatz(inputs, weights)
            return qml.probs(wires=range(self.T))
        self.vqc_circuit = vqc_circuit

    def _get_grover_circuit(self, marked_tuple: tuple):
        """
        Build (or retrieve from cache) the Grover search circuit.

        Circuit (paper Fig. 1, three solid layers):
          H^N  ->  [Oracle(M) -> Diffusion] x G  ->  measure probs

        No trainable parameters — pure Grover amplitude amplification.
        """
        if marked_tuple in self._grover_cache:
            return self._grover_cache[marked_tuple]

        G_iters = self.G

        @qml.qnode(self.dev)
        def grover_circuit():
            self._hadamard_layer()           # eq. 11-12: uniform superposition
            for _ in range(G_iters):
                self._oracle(marked_tuple)   # eq. 13: phase-flip high-reward states
                self._diffusion()            # eq. 14: inversion about the mean
            return qml.probs(wires=range(self.T))

        self._grover_cache[marked_tuple] = grover_circuit
        return grover_circuit

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        """
        Extract beam-domain CSI features for VQC AngleEmbedding input.

        Projects channel matrix N onto DFT beam space (eq. 7),
        picks the dominant beam energy per user, normalises to [0, pi].

        N : (A x T) complex channel matrix
        Returns : (T,) real array in [0, pi]
        """
        k     = np.arange(self.A)
        n     = np.arange(self.A)
        Omega = (np.exp(-1j * 2 * np.pi * np.outer(k, n) / self.A)
                 / np.sqrt(self.A))
        dft_proj = np.abs(Omega.conj().T @ N)   # (A, T)
        best     = np.max(dft_proj, axis=0)     # (T,) dominant beam per user
        return (best / (best.max() + 1e-9)) * np.pi

    def _probs_to_marginals(self, probs: np.ndarray) -> np.ndarray:
        """
        Per-user marginal P(user t scheduled = 1) from the 2^T joint distribution.
        Returns (T,) array via precomputed bit-mask projection.
        """
        return self._bit_mask.T @ probs         # (T,)

    def _log_prob(self, marginals: np.ndarray, theta: np.ndarray) -> float:
        """
        Log-probability of scheduling vector theta under independent Bernoulli
        marginals. Used for the REINFORCE gradient estimate.
        """
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    def _identify_marked_states(self,
                                 vqc_probs:  np.ndarray,
                                 n_schedule: int,
                                 mimo_sys,
                                 N:          np.ndarray,
                                 F:          np.ndarray) -> tuple:
        """
        Algorithm 1, lines 6-9: Oracle evaluation to identify marked set M.

        Evaluates the top-k candidate schedules (ranked by VQC probability)
        using the MIMO reward function on the CURRENT channel N.
        States whose instantaneous sum rate >= tau are marked for amplification.

        FIX 2: N and F come from select() — NOT regenerated here.
               The oracle must score the same channel the agent is acting on.

        Parameters
        ----------
        vqc_probs  : (2^T,) full state distribution from VQC
        n_schedule : required number of scheduled users
        mimo_sys   : MassiveMIMOSystem (compute_sinr + instantaneous_rate)
        N          : (A x T) current channel — same as passed to select()
        F          : (A x T) beamforming matrix — same as passed to select()

        Returns
        -------
        marked_tuple : sorted tuple of high-reward state indices (M)
        best_theta   : theta with highest observed reward among candidates
        best_reward  : corresponding reward value
        """
        k           = min(8, max(2, self.N_states // 8))
        top_indices = np.argsort(vqc_probs)[-k:][::-1]

        marked      = []
        best_reward = -np.inf
        best_theta  = None

        for idx in top_indices:
            bitstring  = format(int(idx), f'0{self.T}b')
            theta_cand = np.array([int(b) for b in bitstring], dtype=int)

            if theta_cand.sum() != n_schedule:
                continue

            # Reward = instantaneous sum rate on the CURRENT channel N, F
            sinr  = mimo_sys.compute_sinr(N, F, theta_cand)
            rates = mimo_sys.instantaneous_rate(sinr)
            r     = float(np.sum(rates))

            if r > best_reward:
                best_reward = r
                best_theta  = theta_cand

            if r >= self.tau:
                marked.append(int(idx))

        return tuple(sorted(marked)), best_theta, best_reward

    # =====================================================================
    # PUBLIC API  (Algorithm 1)
    # =====================================================================

    def select(self, N: np.ndarray, n_schedule: int,
               mimo_sys=None, F: np.ndarray = None) -> np.ndarray:
        """
        Select scheduling vector theta in {0,1}^T  (Algorithm 1, inference).

        Steps (paper Algorithm 1):
          1. VQC encodes CSI -> policy distribution over 2^T candidate schedules.
          2. Oracle evaluates top candidates on channel N -> marks set M.
          3. Grover circuit amplifies probability of states in M.
          4. Combined (VQC x Grover) marginals -> top-n_schedule users selected.

        FIX 1: Selection always comes from Grover-amplified marginals.
               Old code did `return best_theta` here, making the entire
               Grover circuit dead code and blocking policy learning.

        Parameters
        ----------
        N          : (A x T) current channel matrix
        n_schedule : number of users to activate
        mimo_sys   : MassiveMIMOSystem for oracle reward evaluation
        F          : (A x T) beamforming matrix; computed if None
        """
        # Step 1: VQC policy distribution
        inputs    = pnp.array(self._channel_to_angles(N), requires_grad=False)
        vqc_probs = np.array(self.vqc_circuit(inputs, self.weights), dtype=float)

        if F is None and mimo_sys is not None:
            F = mimo_sys.beamforming_vector()

        # Step 2: Oracle — identify marked set M using CURRENT channel N, F
        if mimo_sys is not None and F is not None:
            marked_tuple, _, _ = self._identify_marked_states(
                vqc_probs, n_schedule, mimo_sys, N=N, F=F)  # FIX 2
        else:
            k = max(1, self.N_states // 4)
            marked_tuple = tuple(sorted(
                int(s) for s in np.argsort(vqc_probs)[-k:]
            ))

        # Step 3: Grover amplitude amplification (eq. 11-14)
        use_grover = (len(marked_tuple) > 0
                      and len(marked_tuple) < self.N_states // 2)

        if use_grover:
            grover_probs = np.array(
                self._get_grover_circuit(marked_tuple)(), dtype=float)
            combined  = vqc_probs * grover_probs
            combined /= combined.sum() + 1e-9
            marginals = self._probs_to_marginals(combined)
        else:
            marginals = self._probs_to_marginals(vqc_probs)

        # Step 4: Collapse — pick top-n_schedule users by amplified marginal
        # FIX 1: always from marginals, never short-circuit with best_theta
        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        REINFORCE policy-gradient update on the VQC  (Algorithm 1, training).

        Parameter-shift rule for exact quantum gradients:
            d<O>/dw_k = [ <O>(w_k + pi/2) - <O>(w_k - pi/2) ] / 2

        REINFORCE objective:  J = E[ reward * log pi(theta | N) ]
        Gradient:             grad J ~ reward * grad log pi(theta | N)

        Adam optimiser maintains per-parameter moment estimates.
        Total QNode calls per update = 2 x n_layers x T x 2.
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
                    p_plus  = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_plus = self._log_prob(self._probs_to_marginals(p_plus), theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(self.vqc_circuit(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob(self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift
                    grad[l, i, k]   = reward * (lp_plus - lp_minus) / 2.0

        # Adam update
        self._adam_t += 1
        b1, b2, eps  = self._adam_b1, self._adam_b2, self._adam_eps
        self._adam_m  = b1 * self._adam_m + (1 - b1) * grad
        self._adam_v  = b2 * self._adam_v + (1 - b2) * grad ** 2
        m_hat = self._adam_m / (1 - b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - b2 ** self._adam_t)
        new_w = w_raw + self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.weights = pnp.array(new_w, requires_grad=True)