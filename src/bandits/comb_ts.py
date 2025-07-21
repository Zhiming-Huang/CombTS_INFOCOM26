import numpy as np
from typing import List, Set, Dict, Any
import random


class CombTS:
    """
    Combinatorial Thompson Sampling algorithm for sleeping arms.
    
    This class implements a combinatorial bandit algorithm that can handle
    sleeping arms (arms that may not be available at every round).
    """
    
    def __init__(self, environment, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the CombTS algorithm.
        
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
        
        # Track which arms have been played at least once
        self.played_arms = set()
    
    def select_combination(self) -> Set[int]:
        """
        Select a feasible combination based on Thompson Sampling.
        
        Returns:
            Selected feasible combination (set of arms)
        """
        # Get available arms and feasible combinations from environment
        # Use the consistent method to ensure same arms for algorithm and benchmark
        available_arms = self.environment.sample_available_arms_once()
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return set()
        
        # Draw posterior samples for all available arms
        posterior_samples = {}
        for arm in available_arms:
            # Draw from Beta(alpha_posterior[arm], beta_posterior[arm])
            sample = np.random.beta(self.alpha_posterior[arm], self.beta_posterior[arm])
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
    
    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float]):
        """
        Update posterior distributions based on observed rewards.
        
        Args:
            played_arms: Set of arms that were played
            rewards: Dictionary mapping arm_id to observed reward
        """
        for arm in played_arms:
            if arm in rewards:
                reward = rewards[arm]
                
                # Update posterior parameters
                if reward > 0:  # Success
                    self.alpha_posterior[arm] += 1
                else:  # Failure
                    self.beta_posterior[arm] += 1
                
                self.played_arms.add(arm)
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the arms.
        
        Returns:
            Dictionary containing arm statistics
        """
        stats = {
            'num_arms': self.num_arms,
            'played_arms': len(self.played_arms),
            'alpha_posterior': self.alpha_posterior.copy(),
            'beta_posterior': self.beta_posterior.copy(),
            'expected_rewards': {}
        }
        
        for arm in range(self.num_arms):
            if self.alpha_posterior[arm] + self.beta_posterior[arm] > 2:  # At least one observation
                expected_reward = self.alpha_posterior[arm] / (self.alpha_posterior[arm] + self.beta_posterior[arm])
                stats['expected_rewards'][arm] = expected_reward
        
        return stats 