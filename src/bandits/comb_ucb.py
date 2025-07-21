import numpy as np
from typing import List, Set, Dict, Any


class CombUCB:
    """
    Combinatorial UCB algorithm for sleeping arms.
    
    This class implements a combinatorial bandit algorithm using UCB (Upper Confidence Bound)
    that can handle sleeping arms (arms that may not be available at every round).
    
    The algorithm computes θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}
    where \hat{r}_{a,n_{a,t}} is the empirical mean reward and n_{a,t} is the number of pulls.
    """
    
    def __init__(self, environment, rng=None):
        """
        Initialize the CombUCB algorithm.
        
        Args:
            environment: Environment instance that provides available arms and feasible combinations
        """
        self.environment = environment
        self.num_arms = environment.num_arms
        
        # Initialize tracking variables for each arm
        self.pull_counts = np.zeros(self.num_arms)  # n_{a,t}: number of pulls for each arm
        self.total_rewards = np.zeros(self.num_arms)  # sum of rewards for each arm
        self.empirical_means = np.zeros(self.num_arms)  # \hat{r}_{a,n_{a,t}}: empirical mean rewards
        
        self.rng = rng if rng is not None else np.random
    
    def select_combination(self, round_idx: int) -> Set[int]:
        """
        Select a feasible combination based on UCB values.
        
        Returns:
            Selected feasible combination (set of arms)
        """
        # Get available arms and feasible combinations from environment
        available_arms = self.environment.get_available_arms_for_round(round_idx)
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return set()
        
        # Compute UCB values for all available arms
        ucb_values = {}
        for arm in available_arms:
            ucb_values[arm] = self._compute_ucb_value(arm, round_idx)
        
        # Find the feasible combination with highest sum of UCB values
        best_combination = None
        best_score = float('-inf')
        
        for combination in feasible_combinations:
            # Only consider combinations that are subsets of available arms
            if combination.issubset(available_arms):
                score = sum(ucb_values[arm] for arm in combination)
                if score > best_score:
                    best_score = score
                    best_combination = combination
        
        return best_combination if best_combination is not None else set()
    
    def _compute_ucb_value(self, arm: int, round_idx: int) -> float:
        """
        Compute the UCB value for a given arm.
        
        Args:
            arm: Arm index
            
        Returns:
            UCB value: θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}
        """
        if self.pull_counts[arm] == 0:
            # If arm has never been pulled, return infinity to encourage exploration
            return float('inf')
        
        # Empirical mean reward: \hat{r}_{a,n_{a,t}}
        empirical_mean = self.empirical_means[arm]
        
        # UCB exploration term: \sqrt{1.5 \ln t / n_{a,t}}
        exploration_term = np.sqrt(1.5 * np.log(round_idx + 1) / self.pull_counts[arm])
        
        # UCB value: θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}
        ucb_value = empirical_mean + exploration_term
        
        return ucb_value
    
    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float], round_idx: int):
        """
        Update empirical means and pull counts based on observed rewards.
        
        Args:
            played_arms: Set of arms that were played
            rewards: Dictionary mapping arm_id to observed reward
        """
        for arm in played_arms:
            if arm in rewards:
                reward = rewards[arm]
                
                # Update pull count
                self.pull_counts[arm] += 1
                
                # Update total rewards
                self.total_rewards[arm] += reward
                
                # Update empirical mean: \hat{r}_{a,n_{a,t}} = total_rewards / pull_count
                self.empirical_means[arm] = self.total_rewards[arm] / self.pull_counts[arm]
        
        # Increment round counter
        # self.current_round += 1 # This line is removed as per the edit hint
    
    def get_arm_statistics(self, round_idx: int) -> Dict[str, Any]:
        """
        Get statistics about the arms.
        
        Returns:
            Dictionary containing arm statistics
        """
        stats = {
            'num_arms': self.num_arms,
            'pull_counts': self.pull_counts.copy(),
            'total_rewards': self.total_rewards.copy(),
            'empirical_means': self.empirical_means.copy(),
            'ucb_values': {}
        }
        
        # Compute current UCB values for all arms
        for arm in range(self.num_arms):
            if self.pull_counts[arm] > 0:
                stats['ucb_values'][arm] = self._compute_ucb_value(arm, round_idx)
        
        return stats
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        return {
            'algorithm_name': 'CombUCB',
            'ucb_formula': 'θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}',
            'exploration_parameter': 1.5,
            'description': 'Combinatorial UCB algorithm with sleeping arms support'
        } 