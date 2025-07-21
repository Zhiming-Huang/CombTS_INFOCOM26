import numpy as np
from typing import List, Set, Dict, Any
import random
import os
import tempfile


class SimpleEnvironmentMemmap:
    """
    Simple environment with memory-mapped availability matrix for large-scale simulations.
    
    This environment uses memory mapping to handle availability matrices that are too large
    to fit in memory. It provides the same interface as SimpleEnvironment but with
    disk-based storage for scalability.
    """
    
    def __init__(self, num_arms: int = 10, num_optimal: int = 3, 
                 optimal_mean: float = 0.9, suboptimal_mean: float = 0.1,
                 availability_rate: float = 0.5, max_combination_size: int = 3,
                 num_rounds: int = None, seed: int = None, 
                 matrix_file: str = None, chunk_size: int = 1000):
        """
        Initialize the simple environment with memory mapping.
        
        Args:
            num_arms: Total number of arms
            num_optimal: Number of optimal arms
            optimal_mean: Mean reward for optimal arms
            suboptimal_mean: Mean reward for suboptimal arms
            availability_rate: Probability that each arm is available
            max_combination_size: Maximum size of feasible combinations
            num_rounds: Number of rounds to pre-generate available arms
            seed: Random seed for reproducibility
            matrix_file: Path to memory-mapped file (if None, creates temporary file)
            chunk_size: Size of chunks for matrix generation
        """
        self.num_arms = num_arms
        self.num_optimal = num_optimal
        self.optimal_mean = optimal_mean
        self.suboptimal_mean = suboptimal_mean
        self.availability_rate = availability_rate
        self.max_combination_size = max_combination_size
        self.num_rounds = num_rounds
        self.chunk_size = chunk_size
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
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
            self._generate_memory_mapped_matrix(matrix_file)
    
    def _generate_memory_mapped_matrix(self, matrix_file: str = None):
        """
        Generate availability matrix using memory mapping.
        
        Args:
            matrix_file: Path to memory-mapped file (if None, creates temporary file)
        """
        if matrix_file is None:
            # Create temporary file with unique name
            temp_dir = tempfile.gettempdir()
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            matrix_file = os.path.join(temp_dir, f'availability_matrix_{os.getpid()}_{unique_id}.npy')
            self._is_temp_file = True
        else:
            self._is_temp_file = False
        
        self.matrix_file = matrix_file
        
        print(f"Creating memory-mapped matrix: {self.num_arms} arms × {self.num_rounds} rounds")
        print(f"Matrix file: {matrix_file}")
        
        # Create memory-mapped file
        self.availability_matrix = np.memmap(matrix_file, dtype='uint8', mode='w+', 
                                           shape=(self.num_arms, self.num_rounds))
        
        # Generate matrix in chunks to avoid memory issues
        print("Generating availability matrix in chunks...")
        for chunk_start in range(0, self.num_rounds, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.num_rounds)
            chunk_size_actual = chunk_end - chunk_start
            
            # Generate chunk
            chunk = np.random.binomial(1, self.availability_rate, 
                                     size=(self.num_arms, chunk_size_actual))
            
            # Write chunk to memory-mapped file
            self.availability_matrix[:, chunk_start:chunk_end] = chunk
            
            # Progress report
            if chunk_start % (self.chunk_size * 10) == 0:
                progress = (chunk_start / self.num_rounds) * 100
                print(f"  Progress: {progress:.1f}%")
        
        # Flush to disk
        self.availability_matrix.flush()
        print("Matrix generation completed!")
    
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
    
    def get_availability_matrix_info(self) -> Dict[str, Any]:
        """
        Get information about the availability matrix.
        
        Returns:
            Dictionary containing matrix information
        """
        if not hasattr(self, 'availability_matrix'):
            return {'matrix_generated': False}
        
        matrix = self.availability_matrix
        file_size_mb = os.path.getsize(self.matrix_file) / (1024 * 1024)
        
        return {
            'matrix_generated': True,
            'shape': matrix.shape,
            'total_available': np.sum(matrix),
            'availability_rate_actual': np.mean(matrix),
            'file_size_mb': file_size_mb,
            'matrix_file': self.matrix_file,
            'is_temp_file': getattr(self, '_is_temp_file', False)
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
            'storage_type': 'memory_mapped'
        }
        
        # Add matrix information
        matrix_info = self.get_availability_matrix_info()
        info.update(matrix_info)
        
        return info
    
    def __del__(self):
        """
        Cleanup temporary file if it was created.
        """
        if hasattr(self, '_is_temp_file') and self._is_temp_file:
            try:
                if os.path.exists(self.matrix_file):
                    os.remove(self.matrix_file)
            except:
                pass  # Ignore cleanup errors 