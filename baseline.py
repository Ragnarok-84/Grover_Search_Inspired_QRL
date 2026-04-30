import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROPORTIONAL FAIRNESS SCHEDULER & REWARD ORACLE
# ─────────────────────────────────────────────────────────────────────────────
class ProportionalFairnessScheduler:
    """
    Proportional Fairness module used both as a classical baseline scheduler 
    and as the Reward Oracle inside QRL (eq. 10).
    """

    def __init__(self, T: int, omega: float = 0.1, a: float = 1e-9):
        self.T = T
        self.omega = omega
        self.a = a # Biến a nhỏ ngăn lỗi chia cho 0 theo bài báo
        self.avg_rate = np.ones(T) * 1e-3

    def calculate_pf_reward(self, theta, inst_rates):
        next_avg_rates = np.copy(self.avg_rate)

        for t in range(self.T):
            if theta[t] == 1:
                next_avg_rates[t] = (
                    (1 - self.omega) * self.avg_rate[t]
                    + self.omega * inst_rates[t]
                )

        # eq. 10: sum only over SCHEDULED users (unscheduled have S_hat=0)
        reward = sum(
            inst_rates[t] / (next_avg_rates[t] + self.a)
            for t in range(self.T) if theta[t] == 1
        )
        return reward

    def schedule(self, inst_rates: np.ndarray, n_schedule: int = None):
        """
        Classical Greedy PF scheduling logic.
        Selects users that maximize inst_rate / avg_rate.
        """
        if n_schedule is None:
            n_schedule = max(1, self.T // 2)
            
        scores = inst_rates / (self.avg_rate + self.a)
        theta  = np.zeros(self.T, dtype=int)
        top    = np.argsort(scores)[-n_schedule:]
        theta[top] = 1
        return theta

    def update(self, theta: np.ndarray, inst_rates: np.ndarray):
        """
        Updates the historical ergodic rate S_bar(t).
        Users not scheduled maintain their previous average.
        """
        for t in range(self.T):
            if theta[t] == 1:
                self.avg_rate[t] = (1 - self.omega) * self.avg_rate[t] + self.omega * inst_rates[t]