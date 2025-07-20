import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import beta
import logging


class CombTSAgent:
    """
    Combinatorial Thompson Sampling agent for network routing.
    
    This agent maintains Beta distributions for each edge's reward probability
    and uses Thompson Sampling to select paths from the feasible set.
    """
    
    def __init__(self, edges: List[Tuple[int, int]], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize the Thompson Sampling agent.
        
        Args:
            edges: List of all possible edges in the network
            alpha_prior: Alpha parameter for Beta prior (successes + 1)
            beta_prior: Beta parameter for Beta prior (failures + 1)
        """
        self.edges = [tuple(sorted(edge)) for edge in edges]
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Initialize Beta distribution parameters for each edge
        # alpha = successes + alpha_prior, beta = failures + beta_prior
        self.alpha_params = {edge: alpha_prior for edge in self.edges}
        self.beta_params = {edge: beta_prior for edge in self.edges}
        
        # Keep track of total observations for each edge
        self.edge_observations = {edge: 0 for edge in self.edges}
        
        # Logging for debugging
        self.logger = logging.getLogger(__name__)
    
    def sample_edge_means(self) -> Dict[Tuple[int, int], float]:
        """
        Sample reward estimates for each edge from their respective Beta distributions.
        
        Returns:
            Dictionary mapping edges to their sampled reward estimates
        """
        sampled_means = {}
        
        for edge in self.edges:
            alpha = self.alpha_params[edge]
            beta_param = self.beta_params[edge]
            
            # Sample from Beta distribution
            sampled_mean = np.random.beta(alpha, beta_param)
            sampled_means[edge] = sampled_mean
        
        return sampled_means
    
    def get_path_score(self, path: List[int], edge_means: Dict[Tuple[int, int], float]) -> float:
        """
        Calculate the total score for a path based on sampled edge means.
        
        Args:
            path: List of node IDs representing the path
            edge_means: Dictionary of sampled edge reward estimates
            
        Returns:
            Total score for the path (sum of edge scores)
        """
        if len(path) < 2:
            return 0.0
        
        total_score = 0.0
        
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i + 1]]))
            if edge in edge_means:
                total_score += edge_means[edge]
            else:
                # Edge not in our model, assign low score
                total_score += 0.0
        
        return total_score
    
    def select_path(self, feasible_paths: List[List[int]]) -> List[int]:
        """
        Select the best path from feasible paths using Thompson Sampling.
        
        Args:
            feasible_paths: List of feasible paths, each path is a list of node IDs
            
        Returns:
            Selected path (list of node IDs)
        """
        if not feasible_paths:
            return []
        
        if len(feasible_paths) == 1:
            return feasible_paths[0]
        
        # Sample edge means from Beta distributions
        edge_means = self.sample_edge_means()
        
        # Calculate scores for all feasible paths
        path_scores = []
        for path in feasible_paths:
            score = self.get_path_score(path, edge_means)
            path_scores.append(score)
        
        # Select path with highest score
        best_idx = np.argmax(path_scores)
        selected_path = feasible_paths[best_idx]
        
        self.logger.debug(f"Selected path {selected_path} with score {path_scores[best_idx]}")
        return selected_path
    
    def update(self, path: List[int], reward: float):
        """
        Update Beta distribution parameters based on observed reward.
        
        The reward is distributed equally among all edges in the path.
        
        Args:
            path: The selected path (list of node IDs)
            reward: Total reward observed for the path
        """
        if len(path) < 2:
            return
        
        # Get edges in the path
        path_edges = []
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i + 1]]))
            path_edges.append(edge)
        
        if not path_edges:
            return
        
        # Distribute reward equally among edges
        # For Bernoulli rewards, we assume each edge gets reward/num_edges success probability
        edge_reward = reward / len(path_edges)
        
        # Update Beta parameters for each edge in the path
        for edge in path_edges:
            if edge in self.alpha_params:
                # For Bernoulli rewards: reward is either 0 or 1 per edge
                # We use the edge_reward as success probability
                self.alpha_params[edge] += edge_reward
                self.beta_params[edge] += (1 - edge_reward)
                self.edge_observations[edge] += 1
                
                self.logger.debug(f"Updated edge {edge}: alpha={self.alpha_params[edge]:.3f}, "
                                f"beta={self.beta_params[edge]:.3f}")
    
    def update_with_edge_rewards(self, path: List[int], edge_rewards: Dict[Tuple[int, int], float]):
        """
        Update Beta distribution parameters with individual edge rewards.
        
        Args:
            path: The selected path (list of node IDs)
            edge_rewards: Dictionary mapping edges to their individual rewards
        """
        if len(path) < 2:
            return
        
        # Get edges in the path
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i + 1]]))
            
            if edge in edge_rewards and edge in self.alpha_params:
                reward = edge_rewards[edge]
                # For Bernoulli rewards: reward is 0 or 1
                self.alpha_params[edge] += reward
                self.beta_params[edge] += (1 - reward)
                self.edge_observations[edge] += 1
                
                self.logger.debug(f"Updated edge {edge} with reward {reward}: "
                                f"alpha={self.alpha_params[edge]:.3f}, "
                                f"beta={self.beta_params[edge]:.3f}")
    
    def get_edge_confidence_intervals(self, confidence: float = 0.95) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Get confidence intervals for edge reward estimates.
        
        Args:
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary mapping edges to their (lower_bound, upper_bound) confidence intervals
        """
        alpha_level = 1 - confidence
        intervals = {}
        
        for edge in self.edges:
            alpha = self.alpha_params[edge]
            beta_param = self.beta_params[edge]
            
            # Calculate confidence interval using Beta distribution
            lower = beta.ppf(alpha_level / 2, alpha, beta_param)
            upper = beta.ppf(1 - alpha_level / 2, alpha, beta_param)
            
            intervals[edge] = (lower, upper)
        
        return intervals
    
    def get_edge_posterior_means(self) -> Dict[Tuple[int, int], float]:
        """
        Get posterior mean estimates for each edge.
        
        Returns:
            Dictionary mapping edges to their posterior mean estimates
        """
        means = {}
        
        for edge in self.edges:
            alpha = self.alpha_params[edge]
            beta_param = self.beta_params[edge]
            
            # Posterior mean of Beta distribution
            mean = alpha / (alpha + beta_param)
            means[edge] = mean
        
        return means
    
    def reset(self):
        """Reset the agent to initial state."""
        self.alpha_params = {edge: self.alpha_prior for edge in self.edges}
        self.beta_params = {edge: self.beta_prior for edge in self.edges}
        self.edge_observations = {edge: 0 for edge in self.edges}
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics about the agent's state.
        
        Returns:
            Dictionary containing various statistics
        """
        posterior_means = self.get_edge_posterior_means()
        confidence_intervals = self.get_edge_confidence_intervals()
        
        stats = {
            'total_observations': sum(self.edge_observations.values()),
            'edge_observations': self.edge_observations.copy(),
            'posterior_means': posterior_means,
            'confidence_intervals': confidence_intervals,
            'alpha_params': self.alpha_params.copy(),
            'beta_params': self.beta_params.copy()
        }
        
        return stats