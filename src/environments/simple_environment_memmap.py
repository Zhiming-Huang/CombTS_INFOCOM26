import numpy as np
import tempfile
import os
from typing import List, Set, Dict, Any
import random


class SimpleEnvironmentMemmap:
    """
    Memory-mapped version of SimpleEnvironment for large-scale simulations.
    
    This environment supports:
    - Pre-generated availability matrix with memory mapping
    - Pre-generated rewards matrix with memory mapping
    - Chunked generation for very large matrices
    - File persistence for reuse across sessions
    """
    
    def __init__(self, num_arms: int = 10, num_optimal: int = 3, 
                 optimal_mean: float = 0.9, suboptimal_mean: float = 0.1,
                 availability_rate: float = 0.5, max_combination_size: int = 3,
                 num_rounds: int = None, pre_generate_rewards: bool = False,
                 chunk_size: int = 10000, use_persistent_files: bool = False,
                 seed: int = None):
        """
        Initialize the memory-mapped environment.
        
        Args:
            num_arms: Total number of arms
            num_optimal: Number of optimal arms
            optimal_mean: Mean reward for optimal arms
            suboptimal_mean: Mean reward for suboptimal arms
            availability_rate: Probability that each arm is available
            max_combination_size: Maximum size of feasible combinations
            num_rounds: Number of rounds to pre-generate matrices
            pre_generate_rewards: Whether to pre-generate rewards matrix
            chunk_size: Number of rounds to generate per chunk
            use_persistent_files: Whether to save matrices to disk for reuse
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
        self.chunk_size = chunk_size
        self.use_persistent_files = use_persistent_files
        
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
        
        # Generate matrices if num_rounds is specified
        if num_rounds is not None:
            self._generate_matrices()
    
    def _generate_matrices(self):
        """
        Generate availability and rewards matrices using memory mapping.
        """
        if self.use_persistent_files:
            # Use persistent files
            self.availability_file = f"availability_matrix_{self.num_arms}_{self.num_rounds}.dat"
            if self.pre_generate_rewards:
                self.rewards_file = f"rewards_matrix_{self.num_arms}_{self.num_rounds}.dat"
        else:
            # Use temporary files
            self.availability_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat').name
            if self.pre_generate_rewards:
                self.rewards_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat').name
        
        # Generate availability matrix
        self._generate_availability_matrix()
        
        # Generate rewards matrix if requested
        if self.pre_generate_rewards:
            self._generate_rewards_matrix()
    
    def _generate_availability_matrix(self):
        """
        Generate availability matrix using memory mapping with chunked generation.
        """
        print(f"Generating availability matrix: {self.num_arms} x {self.num_rounds}")
        
        # Create memory-mapped file
        self.availability_matrix = np.memmap(
            self.availability_file,
            dtype=np.int8,
            mode='w+',
            shape=(self.num_arms, self.num_rounds)
        )
        
        # Generate in chunks to avoid memory issues
        for start_round in range(0, self.num_rounds, self.chunk_size):
            end_round = min(start_round + self.chunk_size, self.num_rounds)
            chunk_size_actual = end_round - start_round
            
            # Generate chunk
            chunk = self.rng.binomial(1, self.availability_rate, 
                                    size=(self.num_arms, chunk_size_actual))
            
            # Write to memory-mapped file
            self.availability_matrix[:, start_round:end_round] = chunk
            
            print(f"Generated availability chunk: rounds {start_round}-{end_round-1}")
        
        # Flush to disk
        self.availability_matrix.flush()
        print(f"Availability matrix saved to: {self.availability_file}")
    
    def _generate_rewards_matrix(self):
        """
        Generate rewards matrix using memory mapping with chunked generation.
        """
        print(f"Generating rewards matrix: {self.num_arms} x {self.num_rounds}")
        
        # Create memory-mapped file
        self.rewards_matrix = np.memmap(
            self.rewards_file,
            dtype=np.int8,
            mode='w+',
            shape=(self.num_arms, self.num_rounds)
        )
        
        # Generate in chunks to avoid memory issues
        for start_round in range(0, self.num_rounds, self.chunk_size):
            end_round = min(start_round + self.chunk_size, self.num_rounds)
            chunk_size_actual = end_round - start_round
            
            # Generate chunk for each arm
            chunk = np.zeros((self.num_arms, chunk_size_actual), dtype=np.int8)
            for arm in range(self.num_arms):
                mean = self.arm_means[arm]
                chunk[arm, :] = self.rng.binomial(1, mean, chunk_size_actual)
            
            # Write to memory-mapped file
            self.rewards_matrix[:, start_round:end_round] = chunk
            
            print(f"Generated rewards chunk: rounds {start_round}-{end_round-1}")
        
        # Flush to disk
        self.rewards_matrix.flush()
        print(f"Rewards matrix saved to: {self.rewards_file}")
    
    def get_available_arms_for_round(self, round_idx: int) -> Set[int]:
        """
        Get available arms for a specific round from the memory-mapped matrix.
        
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
        Get reward for a specific arm and round from the memory-mapped matrix.
        
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
        Get available arms for the current round (for backward compatibility).
        
        Returns:
            Set of available arm IDs
        """
        if hasattr(self, 'availability_matrix'):
            # Use pre-generated matrix
            if not hasattr(self, '_current_round'):
                self._current_round = 0
            return self.get_available_arms_for_round(self._current_round)
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
            'memory_usage_mb': matrix.nbytes / (1024 * 1024),
            'file_path': self.availability_file
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
            'memory_usage_mb': matrix.nbytes / (1024 * 1024),
            'file_path': self.rewards_file
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
            'pre_generate_rewards': self.pre_generate_rewards,
            'chunk_size': self.chunk_size,
            'use_persistent_files': self.use_persistent_files
        }
        
        # Add matrix information
        matrix_info = self.get_availability_matrix_info()
        rewards_info = self.get_rewards_matrix_info()
        info.update(matrix_info)
        info.update(rewards_info)
        
        return info
    
    def cleanup(self):
        """
        Clean up temporary files (call this when done with the environment).
        """
        if hasattr(self, 'availability_matrix'):
            del self.availability_matrix
        
        if hasattr(self, 'rewards_matrix'):
            del self.rewards_matrix
        
        # Remove temporary files if not using persistent files
        if not self.use_persistent_files:
            if hasattr(self, 'availability_file') and os.path.exists(self.availability_file):
                os.remove(self.availability_file)
            
            if hasattr(self, 'rewards_file') and os.path.exists(self.rewards_file):
                os.remove(self.rewards_file)
    
    def __del__(self):
        """
        Cleanup when object is destroyed.
        """
        try:
            self.cleanup()
        except:
            pass 