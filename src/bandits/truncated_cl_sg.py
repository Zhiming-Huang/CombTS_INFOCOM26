import numpy as np
from typing import Set, Dict, Any, Optional


class TCLSG:
    """
    Truncated Combinatorial Learning with Single Gaussian (T-CL-SG).

    Each round draws one Gaussian seed conditioned on [-B, B] and uses the
    MOSS-style truncated radius
        sqrt(gamma [log(T / (N(n_{a,t}+1)))]_+ / (n_{a,t}+1)).
    """

    def __init__(
        self,
        environment,
        gamma: float = 0.01,
        horizon: Optional[int] = None,
        truncation_level: Optional[float] = None,
        rng=None,
        optimistic_init: bool = True,
    ):
        self.environment = environment
        self.num_arms = environment.num_arms
        self.gamma = gamma
        self.horizon = int(horizon or getattr(environment, "num_rounds", 1))
        self.truncation_level = (
            float(truncation_level)
            if truncation_level is not None
            else float(np.sqrt(4.0 / gamma) + 1.0)
        )
        self.optimistic_init = optimistic_init
        self.pull_counts = np.zeros(self.num_arms)
        self.total_rewards = np.zeros(self.num_arms)
        self.empirical_means = np.zeros(self.num_arms)

        if optimistic_init:
            self.empirical_means.fill(1.0)

        self.rng = rng if rng is not None else np.random

    def _truncated_gaussian_seed(self) -> float:
        while True:
            seed = self.rng.normal(0, 1)
            if -self.truncation_level <= seed <= self.truncation_level:
                return seed

    def select_combination(self, round_idx: int) -> Set[int]:
        available_arms = self.environment.get_available_arms_for_round(round_idx)
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        if not feasible_combinations:
            return set()

        w_t = self._truncated_gaussian_seed()
        reward_estimates = {}

        for arm in available_arms:
            count = self.pull_counts[arm]
            log_term = np.log(self.horizon / (self.num_arms * (count + 1.0)))
            log_term = max(log_term, 0.0)
            radius = np.sqrt(self.gamma * log_term / (count + 1.0))
            reward_estimates[arm] = self.empirical_means[arm] + w_t * radius

        best_combination = None
        best_score = float("-inf")

        for combination in feasible_combinations:
            if combination.issubset(available_arms):
                score = sum(reward_estimates[arm] for arm in combination)
                if score > best_score:
                    best_score = score
                    best_combination = combination

        return best_combination if best_combination is not None else set()

    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float], round_idx: int):
        for arm in played_arms:
            if arm in rewards:
                reward = rewards[arm]
                self.pull_counts[arm] += 1
                self.total_rewards[arm] += reward
                self.empirical_means[arm] = self.total_rewards[arm] / self.pull_counts[arm]

    def get_arm_statistics(self) -> Dict[str, Any]:
        return {
            "num_arms": self.num_arms,
            "pull_counts": self.pull_counts.copy(),
            "total_rewards": self.total_rewards.copy(),
            "empirical_means": self.empirical_means.copy(),
            "gamma": self.gamma,
            "horizon": self.horizon,
            "truncation_level": self.truncation_level,
            "optimistic_init": self.optimistic_init,
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            "algorithm_name": "T-CL-SG",
            "posterior_formula": (
                r"\bar{r}_{a,t}=\hat{r}_{a,n_{a,t}}+w_t"
                r"\sqrt{\gamma[\log(T/(N(n_{a,t}+1)))]_+/(n_{a,t}+1)}"
            ),
            "gamma_parameter": self.gamma,
            "horizon": self.horizon,
            "truncation_level": self.truncation_level,
            "description": "T-CL-SG: shared truncated Gaussian seed with a MOSS-style radius.",
        }
