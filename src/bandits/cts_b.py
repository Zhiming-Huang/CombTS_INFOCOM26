import numpy as np
from typing import List, Set, Dict, Any
import random


class CTSB:
    """
    Combinatorial Thompson Sampling with Beta Prior algorithm for sleeping arms.
    
    This class implements a combinatorial bandit algorithm that can handle
    sleeping arms (arms that may not be available at every round).
    """
    
    def __init__(self, environment, alpha: float = 1.0, beta: float = 1.0, rng=None):
        """
        Initialize the CTS-B algorithm.
        
        Args:
            environment: Environment instance that provides available arms and feasible combinations
            alpha: Prior parameter for Beta distribution (default: 1.0)
            beta: Prior parameter for Beta distribution (default: 1.0)
        """
        self.environment = environment
        self.num_arms = environment.num_arms
        self.alpha = alpha
        self.beta = beta
        
        # Initialize posterior parameters for each arm
        # alpha_posterior[i] = alpha + number of successes for arm i
        # beta_posterior[i] = beta + number of failures for arm i
        self.alpha_posterior = np.ones(self.num_arms) * alpha
        self.beta_posterior = np.ones(self.num_arms) * beta
        
        self.rng = rng if rng is not None else np.random
    
    def select_combination(self, round_idx: int) -> Set[int]:
        """
        Select a feasible combination based on Thompson Sampling.
        
        Returns:
            Selected feasible combination (set of arms)
        """
        # Get available arms and feasible combinations from environment
        # Use the consistent method to ensure same arms for algorithm and benchmark
        available_arms = self.environment.get_available_arms_for_round(round_idx)
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return set()
        
        # Draw posterior samples for all available arms
        posterior_samples = {}
        for arm in available_arms:
            # Draw from Beta(alpha_posterior[arm], beta_posterior[arm])
            sample = self.rng.beta(self.alpha_posterior[arm], self.beta_posterior[arm])
            posterior_samples[arm] = sample
        
        # Find the feasible combination with highest sum of posterior samples
        best_combination = None
        best_score = float('-inf')
        
        for combination in feasible_combinations:
            # Only consider combinations that are subsets of available arms
            if combination.issubset(available_arms):
                score = sum(posterior_samples[arm] for arm in combination)
                if score > best_score:
                    best_score = score
                    best_combination = combination
        
        return best_combination if best_combination is not None else set()
    
    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float], round_idx: int):
        """
        Update posterior distributions based on observed rewards.
        Args:
            played_arms: Set of arms that were played
            rewards: Dictionary mapping arm_id to observed reward
            round_idx: Current round index (unused)
        """
        for arm in played_arms:
            if arm in rewards:
                reward = rewards[arm]
                # Treat the observed reward as the mean of a Bernoulli distribution
                # Generate a random number and update based on whether it's > 0.5
                if self.rng.random() < reward:  # Success with probability reward
                    self.alpha_posterior[arm] += 1
                else:  # Failure with probability (1 - reward)
                    self.beta_posterior[arm] += 1
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the arms.
        
        Returns:
            Dictionary containing arm statistics
        """
        stats = {
            'num_arms': self.num_arms,
            'alpha_posterior': self.alpha_posterior.copy(),
            'beta_posterior': self.beta_posterior.copy(),
            'expected_rewards': {}
        }
        
        for arm in range(self.num_arms):
            if self.alpha_posterior[arm] + self.beta_posterior[arm] > 2:  # At least one observation
                expected_reward = self.alpha_posterior[arm] / (self.alpha_posterior[arm] + self.beta_posterior[arm])
                stats['expected_rewards'][arm] = expected_reward
        
        return stats 