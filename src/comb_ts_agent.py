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
        self.environment = environment
        self.alpha = alpha
        self.beta = beta
        
        # Initialize Beta distribution parameters for each edge
        self.edge_alpha = {}
        self.edge_beta = {}
        
        # Initialize all edges with prior parameters
        for edge in environment.get_all_edges():
            self.edge_alpha[edge] = alpha
            self.edge_beta[edge] = beta
    
    def sample_edge_means(self) -> Dict[Tuple[int, int], float]:
        """
        Sample reward estimates for each edge from their Beta distributions.
        
        Returns:
            Dictionary mapping edges to sampled reward estimates
        """
        edge_means = {}
        
        for edge in self.environment.get_all_edges():
            # Sample from Beta distribution
            sampled_mean = np.random.beta(self.edge_alpha[edge], self.edge_beta[edge])
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
        Reward is distributed equally among all edges in the path.
        
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
        
        # Distribute reward equally among edges
        edge_reward = reward / num_edges
        
        # Update Beta parameters for each edge in the path
        for edge in path_edges:
            if edge in self.edge_alpha:
                # Convert edge_reward to binary (0 or 1) for Beta distribution
                # We use the probability interpretation: edge_reward represents success probability
                if edge_reward > 0:
                    # Treat as success
                    self.edge_alpha[edge] += 1
                else:
                    # Treat as failure
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
                mean_estimate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
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
