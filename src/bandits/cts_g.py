import numpy as np
from typing import List, Set, Dict, Any


class CTSG:
    """
    Combinatorial Thompson Sampling with Gaussian Priors algorithm for sleeping arms.
    
    This class implements a combinatorial bandit algorithm using Gaussian priors
    that can handle sleeping arms (arms that may not be available at every round).
    
    The algorithm samples from \mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1})
    where \hat{r}_{a, n_{a, t}} is the empirical mean reward and \gamma is a tunable parameter.
    """
    
    def __init__(self, environment, gamma: float = 1.0, rng=None):
        """
        Initialize the CTS-G algorithm.
        
        Args:
            environment: Environment instance that provides available arms and feasible combinations
            gamma: Tunable parameter for variance scaling (default: 1.0)
        """
        self.environment = environment
        self.num_arms = environment.num_arms
        self.gamma = gamma
        
        # Initialize tracking variables for each arm
        self.pull_counts = np.zeros(self.num_arms)  # n_{a,t}: number of pulls for each arm
        self.total_rewards = np.zeros(self.num_arms)  # sum of rewards for each arm
        self.empirical_means = np.zeros(self.num_arms)  # \hat{r}_{a,n_{a,t}}: empirical mean rewards
        
        # Current round counter
        self.rng = rng if rng is not None else np.random
    
    def select_combination(self, round_idx: int) -> Set[int]:
        """
        Select a feasible combination based on Gaussian prior samples.
        
        Returns:
            Selected feasible combination (set of arms)
        """
        available_arms = self.environment.get_available_arms_for_round(round_idx)
        feasible_combinations = self.environment.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return set()
        
        # Draw posterior samples for all available arms
        posterior_samples = {}
        for arm in available_arms:
            posterior_samples[arm] = self._compute_gaussian_sample(arm, round_idx)
        
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
    
    def _compute_gaussian_sample(self, arm: int, round_idx: int) -> float:
        """
        Compute a sample from the Gaussian prior for a given arm.
        
        Args:
            arm: Arm index
            
        Returns:
            Sample from \mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1})
        """
        
        # Empirical mean reward: \hat{r}_{a, n_{a, t}}
        empirical_mean = self.empirical_means[arm]
        
        # Variance: \frac{\gamma \ln t}{n_{a, t}+1}
        variance = self.gamma * np.log(round_idx + 1) / (self.pull_counts[arm] + 1)
        
        
        # Sample from Gaussian prior: \mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1})
        sample = self.rng.normal(empirical_mean, np.sqrt(variance))
        
        return sample
    
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
                
                # Update empirical mean: \hat{r}_{a, n_{a, t}} = total_rewards / pull_count
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
            'gamma': self.gamma,
            'gaussian_samples': {}
        }
        
        # Compute current Gaussian samples for all arms
        for arm in range(self.num_arms):
            if self.pull_counts[arm] > 0:
                stats['gaussian_samples'][arm] = self._compute_gaussian_sample(arm, round_idx)
        
        return stats
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the algorithm.
        
        Returns:
            Dictionary containing algorithm information
        """
        return {
            'algorithm_name': 'CTSG',
            'posterior_formula': r'\mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1})',
            'gamma_parameter': self.gamma,
            'description': 'Combinatorial Thompson Sampling with Gaussian Priors and sleeping arms support'
        } 