import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _PENNYLANE_OK = True
except ImportError:
    _PENNYLANE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _relu(x):    return np.maximum(0.0, x)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _build_upa_dft(A: int):
    """Construct UPA DFT matrix identical to MassiveMIMOSystem._build_dft_matrix."""
    X = int(np.floor(np.sqrt(A)))
    while A % X != 0:
        X -= 1
    Y = A // X
    nx = np.arange(X); kx = np.arange(X)
    Omega_X = (np.exp(-1j * 2 * np.pi * np.outer(kx, nx) / X)
               / np.sqrt(X))
    ny = np.arange(Y); ky = np.arange(Y)
    Omega_Y = (np.exp(-1j * 2 * np.pi * np.outer(ky, ny) / Y)
               / np.sqrt(Y))
    return np.kron(Omega_X, Omega_Y)   # (A, A)


def _channel_to_angles(N: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """
    Beam-domain CSI feature extraction (eq. 7, shared by CNN and QNN).

    Projects (A, T) channel N onto DFT beam space, extracts peak beam
    power per user, normalises to [0, pi].
    """
    dft_proj = np.abs(Omega.conj().T @ N)   # (A, T)
    best     = np.max(dft_proj, axis=0)     # (T,)
    return (best / (best.max() + 1e-9)) * np.pi


def _stochastic_select(scores: np.ndarray, n_schedule: int) -> np.ndarray:
    """
    Select n_schedule users by sampling WITHOUT replacement, proportional
    to normalised scores (softmax-like marginals).
    This mirrors the quantum measurement step in QRLAgent.select().
    """
    probs = np.clip(scores, 1e-9, 1.0 - 1e-9)
    theta = np.random.binomial(1, probs)
    
    # Fallback nếu tắt hết user
    if theta.sum() == 0:
        theta[np.argmax(probs)] = 1
    return theta


# ─────────────────────────────────────────────────────────────────────────────
# CNN SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class CNNScheduler:
    """
    MLP baseline: Linear(T→128) → ReLU → Linear(128→64) → ReLU
                  → Linear(64→T) → Sigmoid
    Input: DFT beam-domain features (T,) — same feature extraction as QRL.
    Trained with REINFORCE + advantage baseline + Adam.
    """

    def __init__(self, A: int, T: int, lr: float = 1e-3):
        self.A     = A
        self.T     = T
        self.lr    = lr
        self.Omega = _build_upa_dft(A)

        # Input is (T,) beam-domain angles, not raw flattened channel
        def w(fi, fo): return np.random.randn(fo, fi) * np.sqrt(2.0 / fi)
        self.W1 = w(T, 128); self.b1 = np.zeros(128)
        self.W2 = w(128, 64); self.b2 = np.zeros(64)
        self.W3 = w(64, T);  self.b3 = np.zeros(T)

        self._t  = 0
        self._ms = [np.zeros_like(p) for p in self._params()]
        self._vs = [np.zeros_like(p) for p in self._params()]

        # Cache forward-pass result so update() reuses it
        self._last_cache = None

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _forward(self, x: np.ndarray):
        z1 = self.W1 @ x + self.b1;  a1 = _relu(z1)
        z2 = self.W2 @ a1 + self.b2; a2 = _relu(z2)
        z3 = self.W3 @ a2 + self.b3; a3 = _sigmoid(z3)
        return a3, (x, z1, a1, z2, a2, z3, a3)

    def _backward(self, cache, advantage: float, theta: np.ndarray):
        x, z1, a1, z2, a2, z3, a3 = cache
        # REINFORCE gradient: advantage * d log pi / d params
        # For Bernoulli policy: d log pi / d logit = (theta - sigma(logit))
        dz3 = advantage * (theta - a3)
        dW3 = np.outer(dz3, a2); db3 = dz3
        da2 = self.W3.T @ dz3;   dz2 = da2 * (z2 > 0)
        dW2 = np.outer(dz2, a1); db2 = dz2
        da1 = self.W2.T @ dz2;   dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x);  db1 = dz1
        return [dW1, db1, dW2, db2, dW3, db3]

    def _adam(self, grads):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self._params(); new_p = []
        for i, (p, g, m, v) in enumerate(
                zip(params, grads, self._ms, self._vs)):
            self._ms[i] = b1 * m + (1 - b1) * g
            self._vs[i] = b2 * v + (1 - b2) * g ** 2
            mh = self._ms[i] / (1 - b1 ** self._t)
            vh = self._vs[i] / (1 - b2 ** self._t)
            new_p.append(p + self.lr * mh / (np.sqrt(vh) + eps))
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = new_p

    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        """
        Forward pass on DFT features → stochastic selection of n_schedule users.
        Caches the forward pass for reuse in update().
        """
        x            = _channel_to_angles(N, self.Omega)   # (T,) features
        probs, cache = self._forward(x)
        self._last_cache = cache
        # Stochastic sampling — mirrors quantum measurement
        return _stochastic_select(probs, n_schedule)

    def update(self, N: np.ndarray, advantage: float, theta: np.ndarray):
        """
        REINFORCE update.  advantage = reward − baseline (computed by caller).
        Reuses cached forward pass from the preceding select() call.
        """
        if self._last_cache is not None:
            cache = self._last_cache
            self._last_cache = None
        else:
            x = _channel_to_angles(N, self.Omega)
            _, cache = self._forward(x)
        self._adam(self._backward(cache, advantage, theta))


# ─────────────────────────────────────────────────────────────────────────────
# QNN SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class QNNScheduler:
    """
    Variational Quantum Circuit (VQC) baseline.

    Architecture (paper ref [5]):
      AngleEmbedding → (RY + RZ + circular CNOT) × n_layers → marginals

    Input features: DFT beam-domain angles (T,) — same as QRLAgent.
    select(): stochastic sampling from marginals (quantum measurement).
    update(): parameter-shift REINFORCE with advantage baseline.
    """

    def __init__(self, A: int, T: int, n_layers: int = 2, lr: float = 0.02):
        if not _PENNYLANE_OK:
            raise ImportError("PennyLane is required for QNNScheduler.")
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.N_states = 2 ** T
        self.Omega    = _build_upa_dft(A)

        self.dev     = qml.device("default.qubit", wires=T)
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Bit-mask for marginal computation
        indices          = np.arange(self.N_states, dtype=np.int32)
        self._bit_mask   = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                            ).astype(np.float32)

        # Adam state
        self._adam_t   = 0
        self._adam_m   = np.zeros((n_layers, T, 2))
        self._adam_v   = np.zeros((n_layers, T, 2))
        self._adam_b1  = 0.9
        self._adam_b2  = 0.999
        self._adam_eps = 1e-8

        self._build_circuit()

    def _build_circuit(self):
        T, n_layers = self.T, self.n_layers

        @qml.qnode(self.dev)
        def _vqc(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(T), rotation='Y')
            for l in range(n_layers):
                for i in range(T):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                for i in range(T - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[T - 1, 0])
            return qml.probs(wires=range(T))

        self._vqc = _vqc

    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        return self._bit_mask.T @ probs_all

    def _log_prob_from_marginals(self, marginals: np.ndarray,
                                  theta: np.ndarray) -> float:
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    def _adam_step(self, grad: np.ndarray, w: np.ndarray) -> np.ndarray:
        self._adam_t += 1
        self._adam_m  = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v  = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        return w + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        """
        VQC forward pass → per-user marginals → stochastic selection.
        Mirrors the quantum measurement step of QRLAgent.select().
        """
        inputs    = pnp.array(
            _channel_to_angles(N, self.Omega), requires_grad=False)
        probs_all = np.array(self._vqc(inputs, self.weights), dtype=float)
        marginals = self._probs_to_marginals(probs_all)
        return _stochastic_select(marginals, n_schedule)

    def update(self, N: np.ndarray, advantage: float, theta: np.ndarray):
        """
        Parameter-shift REINFORCE update.
        advantage = reward − baseline (computed by caller).
        Total QNode calls = 2 × n_layers × T × 2.
        """
        assert theta is not None, "theta must be provided for QNN update."

        inputs = pnp.array(
            _channel_to_angles(N, self.Omega), requires_grad=False)
        shift  = np.pi / 2
        w_raw  = (self.weights.numpy()
                  if hasattr(self.weights, 'numpy')
                  else np.array(self.weights, dtype=float))
        grad   = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    w_raw[l, i, k] += shift
                    p_plus   = np.array(self._vqc(inputs, w_raw), dtype=float)
                    lp_plus  = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_plus), theta)

                    w_raw[l, i, k] -= 2 * shift
                    p_minus  = np.array(self._vqc(inputs, w_raw), dtype=float)
                    lp_minus = self._log_prob_from_marginals(
                                   self._probs_to_marginals(p_minus), theta)

                    w_raw[l, i, k] += shift
                    grad[l, i, k]   = advantage * (lp_plus - lp_minus) / 2.0

        new_w        = self._adam_step(grad, w_raw)
        self.weights = pnp.array(new_w, requires_grad=True)