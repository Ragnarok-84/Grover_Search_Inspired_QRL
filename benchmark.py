"""
benchmark.py  —  CNN and QNN Scheduler Baselines

CNN  : Fully-connected MLP (3 layers) with REINFORCE policy gradient (NumPy only).
QNN  : Variational Quantum Circuit scheduler (PennyLane).

FIX 1: PennyLane imports are guarded so CNNScheduler works without PennyLane.
FIX 2: QNNScheduler.update() now requires theta (never None); removed the
       meaningless log(mean(marginals)) fallback.
FIX 3: CNNScheduler caches the forward-pass from select() so update() reuses
       it instead of running a redundant second forward pass.
"""

import numpy as np

# ── Guard PennyLane import so CNNScheduler still works if PL is absent ──────
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _PENNYLANE_OK = True
except ImportError:
    _PENNYLANE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _relu(x):     return np.maximum(0, x)
def _sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ─────────────────────────────────────────────────────────────────────────────
# CNN SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
class CNNScheduler:
    """
    MLP baseline: Linear(A*T→128) → ReLU → Linear(128→64) → ReLU
                  → Linear(64→T) → Sigmoid
    Trained with REINFORCE (policy gradient).
    """

    def __init__(self, A: int, T: int, lr: float = 1e-3):
        self.A  = A
        self.T  = T
        self.lr = lr

        def w(fi, fo): return np.random.randn(fo, fi) * np.sqrt(2.0 / fi)
        self.W1 = w(A * T, 128); self.b1 = np.zeros(128)
        self.W2 = w(128, 64);    self.b2 = np.zeros(64)
        self.W3 = w(64, T);      self.b3 = np.zeros(T)

        self._t  = 0
        self._ms = [np.zeros_like(p) for p in self._params()]
        self._vs = [np.zeros_like(p) for p in self._params()]

        # FIX 3: cache last forward-pass result for reuse in update()
        self._last_cache = None

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _forward(self, x):
        z1 = self.W1 @ x + self.b1;  a1 = _relu(z1)
        z2 = self.W2 @ a1 + self.b2; a2 = _relu(z2)
        z3 = self.W3 @ a2 + self.b3; a3 = _sigmoid(z3)
        return a3, (x, z1, a1, z2, a2, z3, a3)

    def _backward(self, cache, reward, theta):
        x, z1, a1, z2, a2, z3, a3 = cache
        # REINFORCE gradient for Bernoulli policy
        dz3 = reward * (theta - a3)
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
        Forward pass → choose top-n_schedule users.
        FIX 3: cache is stored so update() can reuse it.
        """
        x           = np.abs(N).flatten()
        probs, cache = self._forward(x)
        self._last_cache = cache          # save for update()

        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(probs)[-n_schedule:]] = 1
        return theta

    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        FIX 3: reuse cached forward pass from the preceding select() call
        instead of running a redundant second forward pass.
        """
        if self._last_cache is not None:
            cache = self._last_cache
            self._last_cache = None
        else:
            _, cache = self._forward(np.abs(N).flatten())

        self._adam(self._backward(cache, reward, theta))


# ─────────────────────────────────────────────────────────────────────────────
# QNN SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
class QNNScheduler:
    """
    Variational Quantum Circuit (VQC) baseline.

    Architecture (paper ref [5]):
      AngleEmbedding → (RY + RZ + circular CNOT) × n_layers → probs

    FIX 2: update() requires theta; meaningless None-fallback removed.
    """

    def __init__(self, A: int, T: int, n_layers: int = 2, lr: float = 0.02):
        if not _PENNYLANE_OK:
            raise ImportError("PennyLane is required for QNNScheduler.")
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.N_states = 2 ** T

        self.dev     = qml.device("default.qubit", wires=T)
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # Precompute bit-mask for fast marginals
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

    def _adam_step(self, grad: np.ndarray, w: np.ndarray) -> np.ndarray:
        self._adam_t += 1
        self._adam_m  = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v  = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        return w + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        inputs    = pnp.array(self._channel_to_angles(N), requires_grad=False)
        probs_all = np.array(self._vqc(inputs, self.weights), dtype=float)
        marginals = self._probs_to_marginals(probs_all)
        theta     = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        """
        FIX 2: theta is mandatory.  Removed meaningless None fallback.
        Total QNode calls = 2 × n_layers × T × 2.
        """
        assert theta is not None, "theta must be provided for QNN update."

        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift  = np.pi / 2
        w_raw  = (self.weights.numpy()
                  if hasattr(self.weights, 'numpy')
                  else np.array(self.weights, dtype=float))
        grad = np.zeros_like(w_raw)

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
                    grad[l, i, k]   = reward * (lp_plus - lp_minus) / 2.0

        new_w        = self._adam_step(grad, w_raw)
        self.weights = pnp.array(new_w, requires_grad=True)