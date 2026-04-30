"""
CNNScheduler.py    CNN-based User Scheduler
Fully-connected network (CNN baseline) using pure NumPy — no PyTorch needed.

Architecture:
  Linear(A*T → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → T) → Sigmoid
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

def _relu(x):    return np.maximum(0, x)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class CNNScheduler:
    def __init__(self, A: int, T: int, lr: float = 1e-3):
        self.A = A; self.T = T; self.lr = lr

        def w(fi, fo): return np.random.randn(fo, fi) * np.sqrt(2.0/fi)
        self.W1=w(A*T,128); self.b1=np.zeros(128)
        self.W2=w(128, 64); self.b2=np.zeros(64)
        self.W3=w(64,   T); self.b3=np.zeros(T)

        self._t=0
        self._ms=[np.zeros_like(p) for p in self._params()]
        self._vs=[np.zeros_like(p) for p in self._params()]

    def _params(self): return [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]

    def _forward(self, x):
        z1=self.W1@x+self.b1; a1=_relu(z1)
        z2=self.W2@a1+self.b2; a2=_relu(z2)
        z3=self.W3@a2+self.b3; a3=_sigmoid(z3)
        return a3,(x,z1,a1,z2,a2,z3,a3)
    
    # SỬA 1: Cần truyền action đã chọn (theta) vào cache để tính gradient
    def _backward(self, cache, reward, theta):
        x, z1, a1, z2, a2, z3, a3 = cache
        
        # SỬA 2: Gradient chuẩn của thuật toán REINFORCE cho hàm Sigmoid
        # (Chênh lệch giữa hành động thực tế và xác suất dự đoán)
        dz3 = reward * (theta - a3)
        
        dW3 = np.outer(dz3, a2); db3 = dz3
        da2 = self.W3.T @ dz3;   dz2 = da2 * (z2 > 0)
        dW2 = np.outer(dz2, a1); db2 = dz2
        da1 = self.W2.T @ dz2;   dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x);  db1 = dz1
        return [dW1, db1, dW2, db2, dW3, db3]

    # SỬA 3: Hàm update cần nhận thêm tham số theta
    def update(self, N: np.ndarray, reward: float, theta: np.ndarray):
        _, cache = self._forward(np.abs(N).flatten())
        self._adam(self._backward(cache, reward, theta))


    def _adam(self, grads):
        self._t+=1; b1,b2,eps=0.9,0.999,1e-8
        params=self._params(); new_p=[]
        for i,(p,g,m,v) in enumerate(zip(params,grads,self._ms,self._vs)):
            self._ms[i]=b1*m+(1-b1)*g
            self._vs[i]=b2*v+(1-b2)*g**2
            mh=self._ms[i]/(1-b1**self._t)
            vh=self._vs[i]/(1-b2**self._t)
            new_p.append(p-self.lr*mh/(np.sqrt(vh)+eps))
        self.W1,self.b1,self.W2,self.b2,self.W3,self.b3=new_p

    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        probs,_=self._forward(np.abs(N).flatten())
        theta=np.zeros(self.T,dtype=int)
        theta[np.argsort(probs)[-n_schedule:]]=1
        return theta

   


"""
QNNScheduler.py   Quantum Neural Network Scheduler
Uses a real Variational Quantum Circuit (VQC) implemented in PennyLane,
replacing the MLP approximation in the original file.

Architecture (matching paper reference [5]):
  • AngleEmbedding  : encode channel magnitudes into qubit rotations
  • n_layers of     : RY + RZ rotations  +  circular CNOT entanglement
  • Measurement     : probability of each basis state → top-K scheduling

Performance optimisations vs original:
  1. Bit-mask matrix (precomputed, shape 2^T x T):
       marginals = bit_mask.T @ probs_all   (one matmul, no Python loop)
     Original had a double Python loop O(2^T x T) called once per QNode
     evaluation — i.e. up to 49x per update() call.

  2. log_prob_theta() no longer calls _vqc() internally.
     Instead, update() passes in the already-evaluated probs_all, so the
     parameter-shift loop (2 x n_params evaluations) drives ALL QNode calls.
     Total QNode calls per update = 2 x n_layers x T x 2  (same as before,
     but the marginal computation per call is now a single matmul).

  3. select() reuses the same matmul path, eliminating the separate loop
     that existed in the original select() body.

  4. Adam optimiser replaces vanilla gradient ascent:
     same per-step cost, faster convergence → fewer total update() calls
     needed in practice.
"""


class QNNScheduler:
    """
    Variational Quantum Circuit (VQC) baseline for user scheduling.

    Parameters
    ----------
    A        : number of BS antennas
    T        : number of users  (= number of qubits in the VQC)
    n_layers : depth of the parameterised circuit
    lr       : learning rate for the policy-gradient update
    """

    def __init__(self, A: int, T: int, n_layers: int = 2, lr: float = 0.02):
        self.A        = A
        self.T        = T
        self.n_layers = n_layers
        self.lr       = lr
        self.N_states = 2 ** T

        # ── PennyLane device ──────────────────────────────────────────────
        self.dev = qml.device("default.qubit", wires=T)

        # ── Trainable weights: shape (n_layers, T, 2)  [RY, RZ per qubit] ─
        self.weights = pnp.array(
            np.random.uniform(0, np.pi / 2, (n_layers, T, 2)),
            requires_grad=True,
        )

        # ── Precompute bit-mask matrix (2^T × T, dtype float32) ───────────
        # bit_mask[s, i] = 1 if bit i of state s is '1', else 0.
        # marginals = bit_mask.T @ probs_all   →  shape (T,)
        # This replaces the O(2^T × T) Python double loop with one matmul.
        indices = np.arange(self.N_states, dtype=np.int32)
        # Unpack bits: shape (2^T, T), MSB first (matches format(s,'0Tb'))
        self._bit_mask = ((indices[:, None] >> np.arange(T - 1, -1, -1)) & 1
                          ).astype(np.float32)   # shape (N_states, T)

        # ── Adam state for weight update ──────────────────────────────────
        self._adam_t  = 0
        self._adam_m  = np.zeros((n_layers, T, 2))   # first moment
        self._adam_v  = np.zeros((n_layers, T, 2))   # second moment
        self._adam_b1 = 0.9
        self._adam_b2 = 0.999
        self._adam_eps = 1e-8

        # ── Build QNode ───────────────────────────────────────────────────
        self._build_circuit()

    # ──────────────────────────────────────────────────────────────────────
    def _build_circuit(self):
        T        = self.T
        n_layers = self.n_layers

        @qml.qnode(self.dev)
        def _vqc(inputs, weights):
            """
            inputs  : 1-D array of length T  (angle-embedded channel features)
            weights : shape (n_layers, T, 2)
            Returns : probability vector of length 2^T
            """
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

    # ──────────────────────────────────────────────────────────────────────
    def _channel_to_angles(self, N: np.ndarray) -> np.ndarray:
        """
        Compress A×T channel matrix → T angles in [0, π].
        Uses per-user mean channel magnitude, normalised.
        """
        magnitudes = np.mean(np.abs(N), axis=0)           # shape (T,)
        magnitudes = magnitudes / (magnitudes.max() + 1e-9)
        return magnitudes * np.pi                           # → [0, π]

    # ──────────────────────────────────────────────────────────────────────
    def _probs_to_marginals(self, probs_all: np.ndarray) -> np.ndarray:
        """
        Compute per-qubit marginal probabilities P(qubit i = |1⟩).

        Uses the precomputed bit-mask matrix so this is a single BLAS matmul
        instead of a Python loop over 2^T states.

        Parameters
        ----------
        probs_all : shape (2^T,)  — full state probability vector from VQC

        Returns
        -------
        marginals : shape (T,)
        """
        return self._bit_mask.T @ probs_all   # (T, 2^T) @ (2^T,) → (T,)

    # ──────────────────────────────────────────────────────────────────────
    def _log_prob_from_marginals(self, marginals: np.ndarray,
                                  theta: np.ndarray) -> float:
        """
        log P(theta) = Σ_i log P(qubit_i = theta_i)
        Factored (mean-field) approximation over qubit marginals.
        """
        p = np.where(theta == 1, marginals, 1.0 - marginals)
        return float(np.sum(np.log(np.clip(p, 1e-9, 1.0))))

    # ──────────────────────────────────────────────────────────────────────
    def _adam_step(self, grad: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Apply one Adam gradient-ascent step, return updated weights."""
        self._adam_t += 1
        self._adam_m = self._adam_b1 * self._adam_m + (1 - self._adam_b1) * grad
        self._adam_v = self._adam_b2 * self._adam_v + (1 - self._adam_b2) * grad ** 2
        m_hat = self._adam_m / (1 - self._adam_b1 ** self._adam_t)
        v_hat = self._adam_v / (1 - self._adam_b2 ** self._adam_t)
        return w + self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)

    # ──────────────────────────────────────────────────────────────────────
    def select(self, N: np.ndarray, n_schedule: int) -> np.ndarray:
        """
        Choose n_schedule users via VQC probability measurement.
        One QNode call; marginals via matmul.
        """
        inputs    = pnp.array(self._channel_to_angles(N), requires_grad=False)
        probs_all = np.array(self._vqc(inputs, self.weights), dtype=float)
        marginals = self._probs_to_marginals(probs_all)

        theta = np.zeros(self.T, dtype=int)
        theta[np.argsort(marginals)[-n_schedule:]] = 1
        return theta

    # ──────────────────────────────────────────────────────────────────────
    def update(self, N: np.ndarray, reward: float, theta: np.ndarray = None):
        """
        REINFORCE policy-gradient update via parameter-shift rule.

        Optimisations vs original
        ─────────────────────────
        • _vqc() is called exactly 2 × n_params times (irreducible minimum
          for parameter-shift).  No extra calls: marginals are computed from
          the returned probs_all via matmul, not by calling _vqc() again.
        • log_prob_theta() is inlined and accepts probs_all directly, so
          there is zero redundant QNode evaluation.
        • Adam replaces vanilla SGD for faster convergence.

        Total QNode calls = 2 × n_layers × T × 2
          (e.g. T=6, n_layers=2  →  48 calls, same count as before but
           each call's Python overhead is now O(T) not O(2^T × T))
        """
        inputs = pnp.array(self._channel_to_angles(N), requires_grad=False)
        shift  = np.pi / 2

        w_raw = (self.weights.numpy()
                 if hasattr(self.weights, 'numpy')
                 else np.array(self.weights, dtype=float))

        grad = np.zeros_like(w_raw)

        for l in range(self.n_layers):
            for i in range(self.T):
                for k in range(2):
                    # ── forward shifted ───────────────────────────────────
                    w_raw[l, i, k] += shift
                    p_plus    = np.array(self._vqc(inputs, w_raw), dtype=float)
                    lp_plus   = self._log_prob_from_marginals(
                                    self._probs_to_marginals(p_plus), theta
                                ) if theta is not None else float(
                                    np.log(np.mean(
                                        self._probs_to_marginals(p_plus)) + 1e-9))

                    # ── backward shifted ──────────────────────────────────
                    w_raw[l, i, k] -= 2 * shift
                    p_minus   = np.array(self._vqc(inputs, w_raw), dtype=float)
                    lp_minus  = self._log_prob_from_marginals(
                                    self._probs_to_marginals(p_minus), theta
                                ) if theta is not None else float(
                                    np.log(np.mean(
                                        self._probs_to_marginals(p_minus)) + 1e-9))

                    w_raw[l, i, k] += shift   # restore

                    # REINFORCE gradient: reward × ∇ log P(theta)
                    grad[l, i, k] = reward * (lp_plus - lp_minus) / 2.0

        new_w = self._adam_step(grad, w_raw)
        self.weights = pnp.array(new_w, requires_grad=True)