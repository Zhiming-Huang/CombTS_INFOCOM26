import numpy as np
from typing import List, Set, Dict, Any


class CLSG:
    """
    Common-Noise Linearized Stochastic Gaussian (CL-SG) algorithm for combinatorial bandits with sleeping arms.
    
    Each round, generate a single standard Gaussian variable w_t, and for each arm compute:
        \bar{r}_{a, t} = \hat{r}_{a, n_{a, t}} + w_t * sqrt(gamma * ln t / (n_{a, t} + 1))
    Then select the feasible combination with the highest sum of reward estimates.
    """
    def __init__(self, environment, gamma: float = 0.1, rng=None):
        """
        Initialize the CL-SG algorithm.
        Args:
            environment: Environment instance
            gamma: Variance scaling parameter (default: 0.1)
        """
        self.environment = environment
        self.num_arms = environment.num_arms
        self.gamma = gamma
        self.pull_counts = np.zeros(self.num_arms)  # n_{a,t}
        self.total_rewards = np.zeros(self.num_arms)
        self.empirical_means = np.zeros(self.num_arms)
        self.rng = rng if rng is not None else np.random

    def select_combination(self, round_idx: int) -> Set[int]:
        available_arms = self.environment.get_available_arms_for_round(round_idx)
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        if not feasible_combinations:
            return set()
        w_t = self.rng.normal(0, 1)
        reward_estimates = {}
        for arm in available_arms:
            mean = self.empirical_means[arm]
            variance = self.gamma * np.log(round_idx + 1) / (self.pull_counts[arm] + 1) if self.pull_counts[arm] > 0 else 1.0
            variance = max(variance, 1e-6)
            reward_estimates[arm] = mean + w_t * np.sqrt(variance)
        best_combination = None
        best_score = float('-inf')
        for combination in feasible_combinations:
            if combination.issubset(available_arms):
                score = sum(reward_estimates[arm] for arm in combination)
                if score > best_score:
                    best_score = score
                    best_combination = combination
        return best_combination if best_combination is not None else set()

    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float], round_idx: int):
        """
        Update empirical means and pull counts based on observed rewards.
        Args:
            played_arms: Set of arms that were played
            rewards: Dictionary mapping arm_id to observed reward
            round_idx: Current round index (unused)
        """
        for arm in played_arms:
            if arm in rewards:
                reward = rewards[arm]
                self.pull_counts[arm] += 1
                self.total_rewards[arm] += reward
                self.empirical_means[arm] = self.total_rewards[arm] / self.pull_counts[arm]

    def get_arm_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_arms': self.num_arms,
            'pull_counts': self.pull_counts.copy(),
            'total_rewards': self.total_rewards.copy(),
            'empirical_means': self.empirical_means.copy(),
            'gamma': self.gamma
        }
        return stats

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            'algorithm_name': 'CL-SG',
            'posterior_formula': r'\bar{r}_{a, t} = \hat{r}_{a, n_{a, t}} + w_t \sqrt{\frac{\gamma \ln t}{n_{a, t}+1}}',
            'gamma_parameter': self.gamma,
            'description': 'CL-SG: Common-noise Gaussian TS with single noise per round.'
        } 