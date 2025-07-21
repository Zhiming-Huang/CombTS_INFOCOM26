import numpy as np
from typing import List, Set, Dict, Any

class BGCTS:
    """
    Batch Gaussian Combinatorial Thompson Sampling (BG-CTS) for sleeping arms.
    Adapted for this project: supports available arms, feasible combinations, and project environment.
    """
    def __init__(self, environment, m=None, sigma=1.0, sigma_prior=1.0, lamda=0.0, rnd_generator=None):
        self.environment = environment
        self.K = environment.num_arms
        self.m = m if m is not None else environment.max_combination_size
        self.sigma = sigma
        self.sigma_prior = sigma_prior
        self.lamda = lamda
        self.muhats = np.zeros(self.K)
        self.sigmapost = 1e8 * np.ones(self.K)
        self.rnd_generator = rnd_generator if rnd_generator is not None else np.random

    def reset(self):
        self.muhats = np.zeros(self.K)
        self.sigmapost = 1e8 * np.ones(self.K)

    def fbonuses(self, round_idx, lamda=None):
        lamda = self.lamda if lamda is None else lamda
        t = max(round_idx + 1, np.exp(1))
        cst = 0 if lamda == 0 else np.log(1 + np.exp(1) / lamda)
        f = (1 + lamda) * 2 * (np.log(t) + (self.m + 2) * np.log(np.log(t)) + self.m / 2 * cst)
        return f

    def gbonuses(self, round_idx, lamda=None):
        f = self.fbonuses(round_idx, lamda)
        l = 1 if (round_idx + 1) < 2 else np.log(round_idx + 1)
        return f / l

    def select_combination(self, round_idx: int) -> Set[int]:
        available_arms = list(self.environment.get_available_arms_for_round(round_idx))
        feasible_combinations = self.environment.get_feasible_combinations(set(available_arms))
        if not feasible_combinations or len(available_arms) == 0:
            return set()
        m = min(self.m, len(available_arms))
        if len(available_arms) <= m:
            return set(available_arms)
        std = np.sqrt(self.gbonuses(round_idx)) * self.sigmapost
        estimate_mean = self.rnd_generator.normal(self.muhats, std)
        available_indices = np.array(available_arms)
        available_estimates = estimate_mean[available_indices]
        if m == 0:
            return set()
        top_indices = available_indices[np.argpartition(available_estimates, -m)[-m:]]
        best_comb = None
        best_score = -np.inf
        for comb in feasible_combinations:
            if len(comb) == 0:
                continue
            score = sum(estimate_mean[list(comb)])
            if score > best_score:
                best_score = score
                best_comb = comb
        return best_comb if best_comb is not None else set(top_indices)

    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float], round_idx: int):
        """
        Update posterior for BG-CTS. round_idx is accepted for interface consistency but unused.
        """
        for arm in played_arms:
            r = rewards.get(arm, 0)
            sigmaprev = self.sigmapost[arm]
            self.sigmapost[arm] = np.sqrt(1 / (1 / sigmaprev ** 2 + 1 / self.sigma ** 2))
            self.muhats[arm] = self.sigmapost[arm] ** 2 * (self.muhats[arm] / sigmaprev ** 2 + r / self.sigma_prior ** 2)

    def get_arm_statistics(self) -> Dict[str, Any]:
        return {
            'num_arms': self.K,
            'muhats': self.muhats.copy(),
            'sigmapost': self.sigmapost.copy(),
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            'algorithm_name': 'BG-CTS',
            'description': 'Batch Gaussian Combinatorial Thompson Sampling (BG-CTS)'
        } 