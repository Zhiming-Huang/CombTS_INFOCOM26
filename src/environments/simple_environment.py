import numpy as np
from typing import List, Set, Dict, Any
import random
import os


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
                 availability_rate: float = 0.5, max_combination_size: int = 3,
                 num_rounds: int = None, pre_generate_rewards: bool = False, seed: int = None):
        """
        Initialize the simple environment.
        
        Args:
            num_arms: Total number of arms
            num_optimal: Number of optimal arms
            optimal_mean: Mean reward for optimal arms
            suboptimal_mean: Mean reward for suboptimal arms
            availability_rate: Probability that each arm is available
            max_combination_size: Maximum size of feasible combinations
            num_rounds: Number of rounds to pre-generate available arms (if None, generate on-demand)
            pre_generate_rewards: Whether to pre-generate all rewards for all rounds
            seed: Random seed for reproducibility
        """
        self.num_arms = num_arms
        self.num_optimal = num_optimal
        self.optimal_mean = optimal_mean
        self.suboptimal_mean = suboptimal_mean
        self.availability_rate = availability_rate
        self.max_combination_size = max_combination_size
        self.num_rounds = num_rounds
        self.pre_generate_rewards = pre_generate_rewards
        
        # Set random seed for reproducibility
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.rng = np.random.default_rng()
        
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
        
        # Generate availability matrix if num_rounds is specified
        if num_rounds is not None:
            self._generate_availability_matrix()
        
        # Generate rewards matrix if pre_generate_rewards is True
        if num_rounds is not None and pre_generate_rewards:
            self._generate_rewards_matrix()
    
    def _generate_availability_matrix(self):
        """
        Generate availability matrix for all rounds.
        Shape: (num_arms, num_rounds), values: 0 or 1
        """
        # Generate random matrix with availability_rate probability
        self.availability_matrix = self.rng.binomial(1, self.availability_rate, 
                                                   size=(self.num_arms, self.num_rounds))
        print(f"Generated availability matrix: {self.availability_matrix.shape}")
    
    def _generate_rewards_matrix(self):
        """
        Generate rewards matrix for all rounds.
        Shape: (num_arms, num_rounds), values: 0 or 1
        """
        self.rewards_matrix = np.zeros((self.num_arms, self.num_rounds), dtype=np.int8)
        
        for arm in range(self.num_arms):
            mean = self.arm_means[arm]
            # Generate rewards for this arm across all rounds
            self.rewards_matrix[arm, :] = self.rng.binomial(1, mean, self.num_rounds)
        
        print(f"Generated rewards matrix: {self.rewards_matrix.shape}")
    
    def get_available_arms_for_round(self, round_idx: int) -> Set[int]:
        """
        Get available arms for a specific round from the pre-generated matrix.
        
        Args:
            round_idx: Round index (0-based)
            
        Returns:
            Set of available arm IDs for the specified round
        """
        if not hasattr(self, 'availability_matrix'):
            raise ValueError("Availability matrix not generated. Set num_rounds in constructor.")
        
        if round_idx >= self.availability_matrix.shape[1]:
            raise ValueError(f"Round {round_idx} exceeds matrix size {self.availability_matrix.shape[1]}")
        
        # Get the column for this round and find available arms (where value == 1)
        available_mask = self.availability_matrix[:, round_idx] == 1
        available_arms = set(np.where(available_mask)[0])
        
        return available_arms
    
    def get_reward_for_round(self, arm: int, round_idx: int) -> float:
        """
        Get reward for a specific arm and round from the pre-generated matrix.
        
        Args:
            arm: ID of the arm
            round_idx: Round index (0-based)
            
        Returns:
            Reward (0 or 1)
        """
        if not hasattr(self, 'rewards_matrix'):
            raise ValueError("Rewards matrix not generated. Set pre_generate_rewards=True in constructor.")
        
        if arm >= self.rewards_matrix.shape[0]:
            raise ValueError(f"Arm {arm} exceeds matrix size {self.rewards_matrix.shape[0]}")
        
        if round_idx >= self.rewards_matrix.shape[1]:
            raise ValueError(f"Round {round_idx} exceeds matrix size {self.rewards_matrix.shape[1]}")
        
        return float(self.rewards_matrix[arm, round_idx])
    
    def get_available_arms(self) -> Set[int]:
        """
        Sample available arms based on availability rate (on-demand generation).
        
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
        Get available arms for the current round (DEPRECATED - use get_available_arms_for_round instead).
        
        Returns:
            Set of available arm IDs
        """
        import warnings
        warnings.warn("sample_available_arms_once is deprecated. Use get_available_arms_for_round(round_idx) instead.", 
                     DeprecationWarning, stacklevel=2)
        
        if hasattr(self, 'availability_matrix'):
            # Use pre-generated matrix with current round
            if not hasattr(self, '_current_round'):
                self._current_round = 0
            available_arms = self.get_available_arms_for_round(self._current_round)
            self._current_round += 1  # Increment for next call
            return available_arms
        else:
            # Fallback to on-demand generation
            if not hasattr(self, '_current_available_arms'):
                self._current_available_arms = self.get_available_arms()
            return self._current_available_arms.copy()
    
    def reset_available_arms(self):
        """
        Move to next round (for backward compatibility).
        """
        if hasattr(self, 'availability_matrix'):
            # Move to next round
            if not hasattr(self, '_current_round'):
                self._current_round = 0
            self._current_round += 1
        else:
            # Fallback to old behavior
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
        
        # If rewards are pre-generated, use them
        if hasattr(self, 'rewards_matrix') and hasattr(self, '_current_round'):
            return self.get_reward_for_round(arm, self._current_round)
        
        # Otherwise, generate on-demand
        mean = self.arm_means[arm]
        return self.rng.binomial(1, mean)
    
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
    
    def reset_round_counter(self):
        """
        Reset the round counter to 0 (call this at the start of a new simulation).
        """
        if hasattr(self, 'availability_matrix'):
            self._current_round = 0
        else:
            if hasattr(self, '_current_available_arms'):
                delattr(self, '_current_available_arms')
    
    def get_current_round(self) -> int:
        """
        Get the current round number.
        
        Returns:
            Current round number
        """
        if hasattr(self, 'availability_matrix'):
            return getattr(self, '_current_round', 0)
        else:
            return 0
    
    def get_availability_matrix_info(self) -> Dict[str, Any]:
        """
        Get information about the availability matrix.
        
        Returns:
            Dictionary containing matrix information
        """
        if not hasattr(self, 'availability_matrix'):
            return {'matrix_generated': False}
        
        matrix = self.availability_matrix
        return {
            'matrix_generated': True,
            'shape': matrix.shape,
            'total_available': np.sum(matrix),
            'availability_rate_actual': np.mean(matrix),
            'memory_usage_mb': matrix.nbytes / (1024 * 1024)
        }
    
    def get_rewards_matrix_info(self) -> Dict[str, Any]:
        """
        Get information about the rewards matrix.
        
        Returns:
            Dictionary containing matrix information
        """
        if not hasattr(self, 'rewards_matrix'):
            return {'rewards_matrix_generated': False}
        
        matrix = self.rewards_matrix
        return {
            'rewards_matrix_generated': True,
            'shape': matrix.shape,
            'total_rewards': np.sum(matrix),
            'reward_rate_actual': np.mean(matrix),
            'memory_usage_mb': matrix.nbytes / (1024 * 1024)
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Returns:
            Dictionary containing environment information
        """
        info = {
            'num_arms': self.num_arms,
            'num_optimal': self.num_optimal,
            'optimal_arms': self.optimal_arms,
            'optimal_mean': self.optimal_mean,
            'suboptimal_mean': self.suboptimal_mean,
            'availability_rate': self.availability_rate,
            'max_combination_size': self.max_combination_size,
            'arm_means': self.arm_means.copy(),
            'optimal_expected_reward': self.optimal_expected_reward,
            'pre_generate_rewards': self.pre_generate_rewards
        }
        
        # Add matrix information
        matrix_info = self.get_availability_matrix_info()
        rewards_info = self.get_rewards_matrix_info()
        info.update(matrix_info)
        info.update(rewards_info)
        
        return info 