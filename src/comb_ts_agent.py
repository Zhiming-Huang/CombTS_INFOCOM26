import numpy as np
from typing import List, Tuple, Dict

import random
from routing_environment import RoutingEnvironment



class CombTSAgent:
    """
    CombTSAgent class implementing combinatorial Thompson Sampling strategy
    for selecting paths and updating posterior distributions.
    """
    
    def __init__(self, environment: RoutingEnvironment, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the combinatorial Thompson Sampling agent.
        
        Args:
            environment: RoutingEnvironment instance
            alpha: Prior alpha parameter for Beta distribution (default: 1.0)
            beta: Prior beta parameter for Beta distribution (default: 1.0)
        """
        if not isinstance(environment, RoutingEnvironment):
            raise TypeError("environment must be a RoutingEnvironment instance")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        
        self.environment = environment
        self.alpha = alpha
        self.beta = beta
        
        # Initialize random number generator for thread safety
        self.rng = np.random.RandomState()
        
        # Initialize Beta distribution parameters for each edge
        self.edge_alpha = {}
        self.edge_beta = {}
        
        # Initialize all edges with prior parameters
        for edge in environment.get_all_edges():
            self.edge_alpha[edge] = alpha
            self.edge_beta[edge] = beta
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng.seed(seed)
    
    def sample_edge_means(self) -> Dict[Tuple[int, int], float]:
        """
        Sample reward estimates for each edge from their Beta distributions.
        
        Returns:
            Dictionary mapping edges to sampled reward estimates
        """
        edge_means = {}
        
        for edge in self.environment.get_all_edges():
            # Sample from Beta distribution using instance RNG
            sampled_mean = self.rng.beta(self.edge_alpha[edge], self.edge_beta[edge])
            edge_means[edge] = sampled_mean
        
        return edge_means
    
    def select_path(self, feasible_paths: List[List[int]]) -> List[int]:
        """
        Select the best path based on sampled reward estimates.
        
        Args:
            feasible_paths: List of feasible paths to choose from
            
        Returns:
            Selected path as list of nodes
        """
        if not feasible_paths:
            return []
        
        # Sample edge means
        edge_means = self.sample_edge_means()
        
        # Calculate path scores
        path_scores = []
        for path in feasible_paths:
            path_score = 0.0
            path_edges = self.environment.get_path_edges(path)
            
            for edge in path_edges:
                if edge in edge_means:
                    path_score += edge_means[edge]
            
            path_scores.append(path_score)
        
        # Select path with highest score
        best_path_idx = np.argmax(path_scores)
        return feasible_paths[best_path_idx]
    
    def update(self, path: List[int], reward: float):
        """
        Update Beta distribution parameters based on observed reward.
        Since we can't observe individual edge rewards, we use the total path reward
        to make probabilistic updates to edges.
        
        Args:
            path: Selected path as list of nodes
            reward: Observed reward for the path
        """
        if not path or len(path) < 2:
            return
        
        path_edges = self.environment.get_path_edges(path)
        num_edges = len(path_edges)
        
        if num_edges == 0:
            return
        
        # Since we observe path reward but need to update edge parameters,
        # we treat the fraction of edges that "succeeded" as reward/num_edges
        # This is a simplification but reasonable for the bandit setting
        success_rate = reward / num_edges
        
        # Update each edge based on the success rate
        for edge in path_edges:
            if edge in self.edge_alpha:
                # Use the success rate to probabilistically update
                # This treats each edge as having success_rate probability of success
                if self.rng.random() < success_rate:
                    self.edge_alpha[edge] += 1
                else:
                    self.edge_beta[edge] += 1
    
    def get_edge_estimates(self) -> Dict[Tuple[int, int], float]:
        """
        Get current mean estimates for all edges.
        
        Returns:
            Dictionary mapping edges to their current mean estimates
        """
        edge_estimates = {}
        
        for edge in self.environment.get_all_edges():
            if edge in self.edge_alpha:
                alpha = self.edge_alpha[edge]
                beta = self.edge_beta[edge]
                total = alpha + beta
                mean_estimate = alpha / total if total > 0 else 0.5
                edge_estimates[edge] = mean_estimate
        
        return edge_estimates
    
    def get_edge_confidence(self) -> Dict[Tuple[int, int], float]:
        """
        Get confidence intervals for edge estimates.
        
        Returns:
            Dictionary mapping edges to their confidence (variance of Beta distribution)
        """
        edge_confidence = {}
        
        for edge in self.environment.get_all_edges():
            if edge in self.edge_alpha:
                alpha = self.edge_alpha[edge]
                beta = self.edge_beta[edge]
                total = alpha + beta
                if total > 0:
                    # Variance of Beta distribution
                    variance = (alpha * beta) / ((total ** 2) * (total + 1))
                    edge_confidence[edge] = variance
                else:
                    edge_confidence[edge] = float('inf')
        
        return edge_confidence
