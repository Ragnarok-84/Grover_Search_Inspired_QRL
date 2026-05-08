import numpy as np


class ProportionalFairnessScheduler:
    """
    Classical proportional-fairness scheduler and PF reward oracle.
    """

    def __init__(self, T: int, omega: float = 0.1, a: float = 1e-9):
        self.T = T
        self.omega = omega
        self.a = a
        self.avg_rate = np.ones(T, dtype=float) * 1e-3

    def reset(self):
        self.avg_rate[:] = 1e-3

    def calculate_pf_reward(self, theta, inst_rates):
        theta = np.asarray(theta, dtype=int)
        inst_rates = np.asarray(inst_rates, dtype=float)
        next_avg_rates = np.copy(self.avg_rate)
        active = np.where(theta == 1)[0]
        for t in active:
            next_avg_rates[t] = (1 - self.omega) * self.avg_rate[t] + self.omega * inst_rates[t]
        return float(np.sum(inst_rates[active] / (next_avg_rates[active] + self.a)))

    def schedule(self, inst_rates: np.ndarray, n_schedule: int | None = None):
        if n_schedule is None:
            n_schedule = max(1, self.T // 2)
        n_schedule = int(np.clip(n_schedule, 1, self.T))
        scores = np.asarray(inst_rates, dtype=float) / (self.avg_rate + self.a)
        theta = np.zeros(self.T, dtype=int)
        top = np.argsort(scores)[-n_schedule:]
        theta[top] = 1
        return theta

    def update(self, theta: np.ndarray, inst_rates: np.ndarray):
        theta = np.asarray(theta, dtype=int)
        inst_rates = np.asarray(inst_rates, dtype=float)
        for t in np.where(theta == 1)[0]:
            self.avg_rate[t] = (1 - self.omega) * self.avg_rate[t] + self.omega * inst_rates[t]
