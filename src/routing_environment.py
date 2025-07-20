import numpy as np
import networkx as nx

from typing import List, Tuple, Dict, Set
import random


class RoutingEnvironment:
    """

    RoutingEnvironment class responsible for simulating the network environment,
    including link availability and reward generation.
    """
    
    def __init__(self, 
                 num_nodes: int, 
                 edge_availability_probs: Dict[Tuple[int, int], float],
                 edge_reward_probs: Dict[Tuple[int, int], float],
                 source: int = 0,
                 target: int = None):

        """
        Initialize the routing environment.
        
        Args:
            num_nodes: Number of nodes in the network
            edge_availability_probs: Dictionary mapping (u, v) -> probability of edge being available
            edge_reward_probs: Dictionary mapping (u, v) -> probability of edge having reward 1
            source: Source node for routing
            target: Target node for routing (defaults to num_nodes - 1)
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not isinstance(edge_availability_probs, dict) or not isinstance(edge_reward_probs, dict):
            raise TypeError("edge probabilities must be dictionaries")
        if source < 0 or source >= num_nodes:
            raise ValueError(f"source node {source} out of range [0, {num_nodes-1}]")
        
        target = target if target is not None else num_nodes - 1
        if target < 0 or target >= num_nodes:
            raise ValueError(f"target node {target} out of range [0, {num_nodes-1}]")
        
        # Validate probability values
        for edge, prob in edge_availability_probs.items():
            if not (0 <= prob <= 1):
                raise ValueError(f"Availability probability for edge {edge} must be in [0,1], got {prob}")
        for edge, prob in edge_reward_probs.items():
            if not (0 <= prob <= 1):
                raise ValueError(f"Reward probability for edge {edge} must be in [0,1], got {prob}")
        
        self.num_nodes = num_nodes
        self.edge_availability_probs = edge_availability_probs
        self.edge_reward_probs = edge_reward_probs
        self.source = source
        self.target = target
        
        # Create the full network graph
        self.full_graph = nx.DiGraph()
        for (u, v) in edge_availability_probs.keys():
            self.full_graph.add_edge(u, v)
    
    def sample_available_graph(self) -> nx.DiGraph:
        """
        Generate available subgraph by randomly sampling edges based on availability probabilities.
        
        Returns:
            Available subgraph as a NetworkX DiGraph
        """
        available_graph = nx.DiGraph()
        
        # Add all nodes to ensure they exist in the graph
        available_graph.add_nodes_from(range(self.num_nodes))
        
        for (u, v), prob in self.edge_availability_probs.items():
            if random.random() < prob:
                available_graph.add_edge(u, v)
        
        return available_graph
    
    def get_feasible_paths(self, available_graph: nx.DiGraph = None, max_paths: int = 10) -> List[List[int]]:

        """
        Find feasible paths from source to target in the available graph.
        For performance, limits the number of paths returned.
        
        Args:
            available_graph: Available subgraph (if None, samples a new one)
            max_paths: Maximum number of paths to return (default: 10)
            
        Returns:
            List of feasible paths as lists of nodes
        """
        if available_graph is None:
            available_graph = self.sample_available_graph()
        
        try:
            # For better performance, limit the number of paths
            # First try to find shortest paths, then explore others
            if nx.has_path(available_graph, self.source, self.target):
                # Get shortest path first
                shortest_path = nx.shortest_path(available_graph, self.source, self.target)
                paths = [shortest_path]
                
                # If we need more paths and the graph is small enough, get more
                if max_paths > 1:
                    try:
                        # Use a generator to avoid computing all paths at once
                        all_paths_gen = nx.all_simple_paths(available_graph, self.source, self.target)
                        for path in all_paths_gen:
                            if path not in paths:
                                paths.append(path)
                            if len(paths) >= max_paths:
                                break
                    except:
                        # If all_simple_paths fails or takes too long, just return shortest
                        pass
                
                return paths
            else:
                return []
        except nx.NetworkXNoPath:
            return []
    
    def get_reward(self, path: List[int]) -> float:
        """
        Calculate reward for a given path.
        Each edge reward is sampled from Bernoulli distribution.
        
        Args:
            path: List of nodes representing the path
            
        Returns:
            Total reward for the path (sum of edge rewards)
        """
        if len(path) < 2:
            return 0.0
        
        total_reward = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in self.edge_reward_probs:
                # Sample reward from Bernoulli distribution
                edge_reward = 1.0 if random.random() < self.edge_reward_probs[(u, v)] else 0.0
                total_reward += edge_reward
        
        return total_reward
    
    def get_path_edges(self, path: List[int]) -> List[Tuple[int, int]]:
        """

        Get list of edges in a path.
        
        Args:
            path: List of nodes representing the path
            
        Returns:
            List of edges as (u, v) tuples
        """
        edges = []
        for i in range(len(path) - 1):
            edges.append((path[i], path[i + 1]))
        return edges
    
    def get_all_edges(self) -> Set[Tuple[int, int]]:
        """
        Get all edges in the network.
        
        Returns:
            Set of all edges as (u, v) tuples
        """
        return set(self.edge_availability_probs.keys())
