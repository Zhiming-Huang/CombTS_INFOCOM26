import numpy as np
import networkx as nx
from typing import List, Set, Dict, Any, Tuple, Optional
from collections import defaultdict
import random


class RoutingEnvironment:
    """
    Routing Environment for Combinatorial Bandit Algorithms.
    
    This environment models a network routing problem where:
    - Each link (edge) in the network is modeled as an arm
    - Paths (combinations of links) are feasible combinations
    - The goal is to find the optimal path from source to destination
    
    Attributes:
        graph: NetworkX graph representing the network topology
        source: Source node
        destination: Destination node
        link_availability_rates: Dictionary mapping edges to availability rates
        link_reward_means: Dictionary mapping edges to reward means
        num_arms: Number of links (edges) in the network
        num_rounds: Number of simulation rounds
        max_combination_size: Maximum path length (number of links)
        arm_means: Array of expected rewards for each link
        availability_matrix: Pre-generated availability matrix (if enabled)
        rewards_matrix: Pre-generated rewards matrix (if enabled)
        rng: Random number generator
    """
    
    def __init__(self, graph: nx.Graph, source: int, destination: int,
                 link_availability_rates: Dict[Tuple[int, int], float],
                 link_reward_means: Dict[Tuple[int, int], float],
                 num_rounds: int = 10000,
                 max_combination_size: Optional[int] = None,
                 pre_generate_availability: bool = True,
                 pre_generate_rewards: bool = True,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the routing environment.
        
        Args:
            graph: NetworkX graph representing the network topology
            source: Source node ID
            destination: Destination node ID
            link_availability_rates: Dictionary mapping edges to availability rates
            link_reward_means: Dictionary mapping edges to reward means
            num_rounds: Number of simulation rounds
            max_combination_size: Maximum path length (if None, calculated automatically)
            pre_generate_availability: Whether to pre-generate availability matrix
            pre_generate_rewards: Whether to pre-generate rewards matrix
            rng: Random number generator
        """
        self.graph = graph.copy()
        self.source = source
        self.destination = destination
        self.link_availability_rates = link_availability_rates.copy()
        self.link_reward_means = link_reward_means.copy()
        self.num_rounds = num_rounds
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Convert edges to arms (links)
        self.edges = list(self.graph.edges())
        self.num_arms = len(self.edges)
        
        # Create edge to arm index mapping
        self.edge_to_arm = {edge: i for i, edge in enumerate(self.edges)}
        self.arm_to_edge = {i: edge for i, edge in enumerate(self.edges)}
        
        # Set arm means based on link reward means
        self.arm_means = np.zeros(self.num_arms)
        for edge, mean in self.link_reward_means.items():
            if edge in self.edge_to_arm:
                self.arm_means[self.edge_to_arm[edge]] = mean
        
        # Set availability rates for each arm
        self.arm_availability_rates = np.zeros(self.num_arms)
        for edge, rate in self.link_availability_rates.items():
            if edge in self.edge_to_arm:
                self.arm_availability_rates[self.edge_to_arm[edge]] = rate
        
        # Calculate max combination size if not provided
        if max_combination_size is None:
            # Find shortest path length as a reasonable upper bound
            try:
                shortest_path = nx.shortest_path(self.graph, source, destination)
                self.max_combination_size = len(shortest_path) - 1  # Number of edges
            except nx.NetworkXNoPath:
                self.max_combination_size = self.num_arms
        else:
            self.max_combination_size = max_combination_size
        
        # Pre-generate matrices if requested
        self.availability_matrix = None
        self.rewards_matrix = None
        
        if pre_generate_availability:
            self._generate_availability_matrix()
        
        if pre_generate_rewards:
            self._generate_rewards_matrix()
    
    def _generate_availability_matrix(self):
        """Pre-generate availability matrix for all rounds."""
        print(f"Generating availability matrix: {self.num_rounds} rounds × {self.num_arms} arms")
        self.availability_matrix = np.zeros((self.num_rounds, self.num_arms), dtype=bool)
        
        for round_idx in range(self.num_rounds):
            for arm_idx in range(self.num_arms):
                availability_rate = self.arm_availability_rates[arm_idx]
                self.availability_matrix[round_idx, arm_idx] = self.rng.random() < availability_rate
    
    def _generate_rewards_matrix(self):
        """Pre-generate rewards matrix for all rounds."""
        print(f"Generating rewards matrix: {self.num_rounds} rounds × {self.num_arms} arms")
        self.rewards_matrix = np.zeros((self.num_rounds, self.num_arms))
        
        for round_idx in range(self.num_rounds):
            for arm_idx in range(self.num_arms):
                mean = self.arm_means[arm_idx]
                # Bernoulli distribution
                self.rewards_matrix[round_idx, arm_idx] = self.rng.random() < mean
    
    def get_available_arms_for_round(self, round_idx: int) -> Set[int]:
        """
        Get available arms (links) for a given round.
        
        Args:
            round_idx: Round index
            
        Returns:
            Set of available arm indices
        """
        if self.availability_matrix is not None:
            return set(np.where(self.availability_matrix[round_idx])[0])
        else:
            available_arms = set()
            for arm_idx in range(self.num_arms):
                availability_rate = self.arm_availability_rates[arm_idx]
                if self.rng.random() < availability_rate:
                    available_arms.add(arm_idx)
            return available_arms
    
    def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]:
        """
        Get all feasible path combinations from available arms.
        
        Args:
            available_arms: Set of available arm indices
            
        Returns:
            List of feasible path combinations (sets of arm indices)
        """
        if not available_arms:
            return []
        
        # Create subgraph with only available edges
        available_edges = [self.arm_to_edge[arm] for arm in available_arms]
        subgraph = self.graph.edge_subgraph(available_edges)
        
        # Check if source and destination are in the subgraph
        if self.source not in subgraph.nodes() or self.destination not in subgraph.nodes():
            return []
        
        # Find all simple paths from source to destination
        try:
            all_paths = list(nx.all_simple_paths(subgraph, self.source, self.destination))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        
        # Convert paths to arm combinations
        feasible_combinations = []
        for path in all_paths:
            if len(path) > 1:  # Path must have at least 2 nodes (1 edge)
                # Convert path to edges
                path_edges = []
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    # Handle both directions
                    if edge in self.edge_to_arm:
                        path_edges.append(edge)
                    elif (edge[1], edge[0]) in self.edge_to_arm:
                        path_edges.append((edge[1], edge[0]))
                
                # Convert edges to arm indices
                arm_combination = set()
                for edge in path_edges:
                    if edge in self.edge_to_arm:
                        arm_idx = self.edge_to_arm[edge]
                        if arm_idx in available_arms:
                            arm_combination.add(arm_idx)
                
                if arm_combination and len(arm_combination) <= self.max_combination_size:
                    feasible_combinations.append(arm_combination)
        
        return feasible_combinations
    
    def get_optimal_combination(self, available_arms: Set[int]) -> Optional[Set[int]]:
        """
        Get the optimal path combination from available arms.
        
        Args:
            available_arms: Set of available arm indices
            
        Returns:
            Optimal path combination (set of arm indices) or None if no path exists
        """
        feasible_combinations = self.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return None
        
        # Find combination with maximum expected reward
        best_combination = None
        best_reward = -np.inf
        
        for combination in feasible_combinations:
            expected_reward = sum(self.arm_means[arm] for arm in combination)
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_combination = combination
        
        return best_combination
    
    def get_reward_for_round(self, arm: int, round_idx: int) -> float:
        """
        Get reward for a specific arm in a specific round.
        
        Args:
            arm: Arm index
            round_idx: Round index
            
        Returns:
            Reward value
        """
        if self.rewards_matrix is not None:
            return self.rewards_matrix[round_idx, arm]
        else:
            mean = self.arm_means[arm]
            return float(self.rng.random() < mean)
    
    def reset(self):
        """Reset the environment (regenerate matrices if needed)."""
        if self.availability_matrix is not None:
            self._generate_availability_matrix()
        if self.rewards_matrix is not None:
            self._generate_rewards_matrix()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the environment."""
        return {
            'num_arms': self.num_arms,
            'num_rounds': self.num_rounds,
            'max_combination_size': self.max_combination_size,
            'source': self.source,
            'destination': self.destination,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'arm_means': self.arm_means.copy(),
            'arm_availability_rates': self.arm_availability_rates.copy(),
            'edges': self.edges.copy()
        }
    
    def visualize_network(self, highlight_path: Optional[Set[int]] = None):
        """
        Visualize the network topology.
        
        Args:
            highlight_path: Set of arm indices to highlight (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(self.graph)
            
            # Draw all edges
            nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray')
            
            # Highlight optimal path if provided
            if highlight_path:
                highlight_edges = [self.arm_to_edge[arm] for arm in highlight_path 
                                 if arm in self.arm_to_edge]
                nx.draw_networkx_edges(self.graph, pos, edgelist=highlight_edges, 
                                     edge_color='red', width=3)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                                 node_size=500)
            
            # Highlight source and destination
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.source], 
                                 node_color='green', node_size=700)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.destination], 
                                 node_color='red', node_size=700)
            
            # Add labels
            nx.draw_networkx_labels(self.graph, pos)
            
            plt.title(f"Network Topology (Source: {self.source}, Destination: {self.destination})")
            plt.axis('off')
            plt.show()
            
        except ImportError:
            print("matplotlib not available for visualization")
    
    def get_path_info(self, arm_combination: Set[int]) -> Dict[str, Any]:
        """
        Get detailed information about a path.
        
        Args:
            arm_combination: Set of arm indices representing a path
            
        Returns:
            Dictionary with path information
        """
        if not arm_combination:
            return {'path': [], 'edges': [], 'length': 0, 'expected_reward': 0}
        
        # Convert arms to edges
        edges = [self.arm_to_edge[arm] for arm in arm_combination if arm in self.arm_to_edge]
        
        # Try to reconstruct the path
        path = []
        if edges:
            # Create subgraph with these edges
            subgraph = self.graph.edge_subgraph(edges)
            try:
                # Find path from source to destination
                path = nx.shortest_path(subgraph, self.source, self.destination)
            except nx.NetworkXNoPath:
                path = []
        
        return {
            'path': path,
            'edges': edges,
            'length': len(edges),
            'expected_reward': sum(self.arm_means[arm] for arm in arm_combination)
        } 