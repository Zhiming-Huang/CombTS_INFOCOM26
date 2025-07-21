import numpy as np
import networkx as nx
from typing import List, Set, Dict, Tuple, Optional, Any
import random


class RoutingEnvironment:
    """
    Network routing environment for combinatorial bandits with sleeping arms.
    
    This class simulates a network where links can be unavailable (sleeping arms)
    and provides methods to find feasible paths and generate rewards.
    """
    
    def __init__(self, network_topology: nx.Graph, link_means: Dict[int, float],
                 availability_rates: Optional[Dict[int, float]] = None):
        """
        Initialize the routing environment.
        
        Args:
            network_topology: NetworkX graph representing the network topology
            link_means: Dictionary mapping link_id to mean reward (Bernoulli parameter)
            availability_rates: Dictionary mapping link_id to availability probability
                               (if None, all links are always available)
        """
        self.network = network_topology
        self.link_means = link_means
        self.availability_rates = availability_rates or {link: 1.0 for link in self.network.edges()}
        
        # Create a mapping from edge tuples to link IDs
        self.edge_to_link_id = {}
        self.link_id_to_edge = {}
        
        for i, (u, v) in enumerate(self.network.edges()):
            self.edge_to_link_id[(u, v)] = i
            self.edge_to_link_id[(v, u)] = i  # Undirected graph
            self.link_id_to_edge[i] = (u, v)
        
        self.num_links = len(self.network.edges())
        
        # Validate that all link means are provided
        for link_id in range(self.num_links):
            if link_id not in self.link_means:
                raise ValueError(f"Mean reward not provided for link {link_id}")
    
    def sample_available_links(self) -> Set[int]:
        """
        Sample available links based on availability rates.
        
        Returns:
            Set of available link IDs
        """
        available_links = set()
        
        for link_id in range(self.num_links):
            if link_id in self.availability_rates:
                availability_rate = self.availability_rates[link_id]
                if random.random() < availability_rate:
                    available_links.add(link_id)
            else:
                # If no availability rate specified, link is always available
                available_links.add(link_id)
        
        return available_links
    
    def get_feasible_paths(self, source: int, destination: int, 
                          available_links: Set[int]) -> List[Set[int]]:
        """
        Get all feasible paths from source to destination using only available links.
        
        Args:
            source: Source node
            destination: Destination node
            available_links: Set of available link IDs
            
        Returns:
            List of feasible paths, where each path is a set of link IDs
        """
        # Create a subgraph with only available links
        available_edges = []
        for link_id in available_links:
            u, v = self.link_id_to_edge[link_id]
            available_edges.append((u, v))
        
        subgraph = self.network.edge_subgraph(available_edges)
        
        # Check if source and destination are in the subgraph
        if source not in subgraph.nodes() or destination not in subgraph.nodes():
            return []
        
        # Find all simple paths from source to destination
        try:
            all_paths = list(nx.all_simple_paths(subgraph, source, destination))
        except nx.NetworkXNoPath:
            return []
        
        # Convert paths to sets of link IDs
        feasible_paths = []
        for path in all_paths:
            path_links = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_id = self.edge_to_link_id[(u, v)]
                path_links.add(link_id)
            feasible_paths.append(path_links)
        
        return feasible_paths
    
    def generate_reward(self, link_id: int) -> float:
        """
        Generate reward for a specific link based on its Bernoulli distribution.
        
        Args:
            link_id: ID of the link
            
        Returns:
            Reward (0 or 1)
        """
        if link_id not in self.link_means:
            raise ValueError(f"Link {link_id} not found in link_means")
        
        mean = self.link_means[link_id]
        return np.random.binomial(1, mean)
    
    def generate_path_reward(self, path: Set[int]) -> Dict[int, float]:
        """
        Generate rewards for all links in a path.
        
        Args:
            path: Set of link IDs representing a path
            
        Returns:
            Dictionary mapping link_id to reward
        """
        rewards = {}
        for link_id in path:
            rewards[link_id] = self.generate_reward(link_id)
        return rewards
    
    def get_optimal_path(self, source: int, destination: int) -> Optional[Set[int]]:
        """
        Get the optimal path from source to destination (assuming all links available).
        
        Args:
            source: Source node
            destination: Destination node
            
        Returns:
            Set of link IDs representing the optimal path, or None if no path exists
        """
        try:
            # Find shortest path (assuming equal weights for now)
            shortest_path = nx.shortest_path(self.network, source, destination)
            
            # Convert to set of link IDs
            path_links = set()
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                link_id = self.edge_to_link_id[(u, v)]
                path_links.add(link_id)
            
            return path_links
        except nx.NetworkXNoPath:
            return None
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the network.
        
        Returns:
            Dictionary containing network information
        """
        return {
            'num_nodes': self.network.number_of_nodes(),
            'num_links': self.network.number_of_edges(),
            'link_means': self.link_means.copy(),
            'availability_rates': self.availability_rates.copy(),
            'edge_to_link_id': self.edge_to_link_id.copy(),
            'link_id_to_edge': self.link_id_to_edge.copy()
        } 