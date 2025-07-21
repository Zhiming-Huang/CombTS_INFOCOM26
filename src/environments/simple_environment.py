import numpy as np
from typing import List, Set, Dict, Any
import random


class SimpleEnvironment:
    """
    Simple environment for testing CombTS algorithm.
    
    This environment has 10 arms:
    - 3 optimal arms with Bernoulli(0.9) reward distribution
    - 7 suboptimal arms with Bernoulli(0.1) reward distribution
    - Each arm has 0.5 availability rate
    - Feasible combinations are all subsets of available arms with size <= 3
    """
    
    def __init__(self, num_arms: int = 10, num_optimal: int = 3, 
                 optimal_mean: float = 0.9, suboptimal_mean: float = 0.1,
                 availability_rate: float = 0.5, max_combination_size: int = 3):
        """
        Initialize the simple environment.
        
        Args:
            num_arms: Total number of arms
            num_optimal: Number of optimal arms
            optimal_mean: Mean reward for optimal arms
            suboptimal_mean: Mean reward for suboptimal arms
            availability_rate: Probability that each arm is available
            max_combination_size: Maximum size of feasible combinations
        """
        self.num_arms = num_arms
        self.num_optimal = num_optimal
        self.optimal_mean = optimal_mean
        self.suboptimal_mean = suboptimal_mean
        self.availability_rate = availability_rate
        self.max_combination_size = max_combination_size
        
        # Define which arms are optimal (first num_optimal arms)
        self.optimal_arms = set(range(num_optimal))
        
        # Create reward means for each arm
        self.arm_means = {}
        for arm in range(num_arms):
            if arm in self.optimal_arms:
                self.arm_means[arm] = optimal_mean
            else:
                self.arm_means[arm] = suboptimal_mean
        
        # Calculate optimal expected reward
        self.optimal_expected_reward = num_optimal * optimal_mean
    
    def get_available_arms(self) -> Set[int]:
        """
        Sample available arms based on availability rate.
        
        Returns:
            Set of available arm IDs
        """
        available_arms = set()
        for arm in range(self.num_arms):
            if random.random() < self.availability_rate:
                available_arms.add(arm)
        return available_arms
    
    def get_available_arms_fixed(self, available_arms: Set[int]) -> Set[int]:
        """
        Return the provided available arms (for consistency in regret calculation).
        
        Args:
            available_arms: Set of available arm IDs
            
        Returns:
            The same set of available arms
        """
        return available_arms
    
    def sample_available_arms_once(self) -> Set[int]:
        """
        Sample available arms once and store them for consistent use.
        
        Returns:
            Set of available arm IDs
        """
        if not hasattr(self, '_current_available_arms'):
            self._current_available_arms = self.get_available_arms()
        return self._current_available_arms.copy()
    
    def reset_available_arms(self):
        """
        Reset the current available arms (call this at the start of each round).
        """
        if hasattr(self, '_current_available_arms'):
            delattr(self, '_current_available_arms')
    
    def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]:
        """
        Get all feasible combinations from available arms.
        
        Args:
            available_arms: Set of available arm IDs
            
        Returns:
            List of feasible combinations (each is a set of arms)
        """
        feasible_combinations = []
        
        # Generate all subsets of available arms with size <= max_combination_size
        available_list = list(available_arms)
        for size in range(1, min(self.max_combination_size + 1, len(available_arms) + 1)):
            from itertools import combinations
            for combo in combinations(available_list, size):
                feasible_combinations.append(set(combo))
        
        return feasible_combinations
    
    def generate_reward(self, arm: int) -> float:
        """
        Generate reward for a specific arm based on its Bernoulli distribution.
        
        Args:
            arm: ID of the arm
            
        Returns:
            Reward (0 or 1)
        """
        if arm not in self.arm_means:
            raise ValueError(f"Arm {arm} not found")
        
        mean = self.arm_means[arm]
        return np.random.binomial(1, mean)
    
    def generate_combination_reward(self, combination: Set[int]) -> Dict[int, float]:
        """
        Generate rewards for all arms in a combination.
        
        Args:
            combination: Set of arm IDs representing a combination
            
        Returns:
            Dictionary mapping arm_id to reward
        """
        rewards = {}
        for arm in combination:
            rewards[arm] = self.generate_reward(arm)
        return rewards
    
    def get_optimal_combination(self, available_arms: Set[int]) -> Set[int]:
        """
        Get the optimal combination from available arms.
        
        Args:
            available_arms: Set of available arm IDs
            
        Returns:
            Optimal combination (arms with highest mean rewards, up to max_combination_size)
        """
        if not available_arms:
            return set()
        
        # Sort available arms by their mean rewards (descending order)
        sorted_arms = sorted(available_arms, key=lambda arm: self.arm_means[arm], reverse=True)
        
        # Return the top arms up to max_combination_size
        return set(sorted_arms[:self.max_combination_size])
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Returns:
            Dictionary containing environment information
        """
        return {
            'num_arms': self.num_arms,
            'num_optimal': self.num_optimal,
            'optimal_arms': self.optimal_arms,
            'optimal_mean': self.optimal_mean,
            'suboptimal_mean': self.suboptimal_mean,
            'availability_rate': self.availability_rate,
            'max_combination_size': self.max_combination_size,
            'arm_means': self.arm_means.copy(),
            'optimal_expected_reward': self.optimal_expected_reward
        } 