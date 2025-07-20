import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import random


class RoutingEnvironment:
    """
    Environment class for simulating network routing with link availability and reward generation.
    
    This class handles:
    - Network graph with edge availability probabilities
    - Sampling available subgraphs based on link availability
    - Finding feasible paths between source and destination
    - Generating rewards for paths using Bernoulli sampling
    """
    
    def __init__(self, 
                 graph: nx.Graph, 
                 availability_probs: Dict[Tuple[int, int], float],
                 reward_means: Dict[Tuple[int, int], float],
                 source: int, 
                 target: int):
        """
        Initialize the routing environment.
        
        Args:
            graph: NetworkX graph representing the network topology
            availability_probs: Dictionary mapping edges to their availability probabilities
            reward_means: Dictionary mapping edges to their mean rewards (for Bernoulli sampling)
            source: Source node for routing
            target: Target node for routing
        """
        self.graph = graph
        self.availability_probs = availability_probs
        self.reward_means = reward_means
        self.source = source
        self.target = target
        
        # Validate that all edges have availability and reward probabilities
        for edge in self.graph.edges():
            edge_tuple = tuple(sorted(edge))
            if edge_tuple not in self.availability_probs:
                raise ValueError(f"Edge {edge_tuple} missing availability probability")
            if edge_tuple not in self.reward_means:
                raise ValueError(f"Edge {edge_tuple} missing reward mean")
    
    def sample_available_graph(self) -> nx.Graph:
        """
        Sample an available subgraph based on edge availability probabilities.
        
        Returns:
            NetworkX graph representing the available network at this time step
        """
        available_graph = nx.Graph()
        available_graph.add_nodes_from(self.graph.nodes())
        
        for edge in self.graph.edges():
            edge_tuple = tuple(sorted(edge))
            availability_prob = self.availability_probs[edge_tuple]
            
            # Sample edge availability using Bernoulli distribution
            if np.random.random() < availability_prob:
                available_graph.add_edge(edge[0], edge[1])
        
        return available_graph
    
    def get_feasible_paths(self, available_graph: nx.Graph, max_paths: int = None) -> List[List[int]]:
        """
        Find all feasible paths from source to target in the available graph.
        
        Args:
            available_graph: The currently available network graph
            max_paths: Maximum number of paths to return (None for all paths)
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        try:
            # Use simple_paths to find all paths (can be computationally expensive for large graphs)
            if max_paths is None:
                paths = list(nx.all_simple_paths(available_graph, self.source, self.target))
            else:
                paths = []
                path_generator = nx.all_simple_paths(available_graph, self.source, self.target)
                for i, path in enumerate(path_generator):
                    if i >= max_paths:
                        break
                    paths.append(path)
            
            return paths
        except nx.NetworkXNoPath:
            # No path exists between source and target
            return []
    
    def get_reward(self, path: List[int]) -> float:
        """
        Generate reward for a given path using Bernoulli sampling for each edge.
        
        Args:
            path: List of node IDs representing the path
            
        Returns:
            Total reward for the path (sum of edge rewards)
        """
        if len(path) < 2:
            return 0.0
        
        total_reward = 0.0
        
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_tuple = tuple(sorted(edge))
            
            # Sample edge reward using Bernoulli distribution
            reward_mean = self.reward_means[edge_tuple]
            edge_reward = np.random.binomial(1, reward_mean)
            total_reward += edge_reward
        
        return total_reward
    
    def get_path_edges(self, path: List[int]) -> List[Tuple[int, int]]:
        """
        Convert a path (list of nodes) to a list of edges.
        
        Args:
            path: List of node IDs representing the path
            
        Returns:
            List of edges as tuples
        """
        if len(path) < 2:
            return []
        
        edges = []
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i + 1]]))
            edges.append(edge)
        
        return edges
    
    def is_path_available(self, path: List[int], available_graph: nx.Graph) -> bool:
        """
        Check if a given path is available in the current graph.
        
        Args:
            path: List of node IDs representing the path
            available_graph: The currently available network graph
            
        Returns:
            True if all edges in the path are available, False otherwise
        """
        if len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            if not available_graph.has_edge(path[i], path[i + 1]):
                return False
        
        return True


def create_sample_network() -> Tuple[nx.Graph, Dict, Dict, int, int]:
    """
    Create a sample network for testing purposes.
    
    Returns:
        Tuple containing (graph, availability_probs, reward_means, source, target)
    """
    # Create a simple grid network
    G = nx.Graph()
    
    # Add nodes
    nodes = [(i, j) for i in range(3) for j in range(3)]
    node_mapping = {(i, j): i * 3 + j for i, j in nodes}
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    
    G.add_nodes_from(range(9))
    
    # Add edges (grid connectivity)
    edges = []
    for i in range(3):
        for j in range(3):
            current = node_mapping[(i, j)]
            # Right neighbor
            if j < 2:
                neighbor = node_mapping[(i, j + 1)]
                edges.append((current, neighbor))
            # Down neighbor
            if i < 2:
                neighbor = node_mapping[(i + 1, j)]
                edges.append((current, neighbor))
    
    G.add_edges_from(edges)
    
    # Set availability probabilities (higher for some edges)
    availability_probs = {}
    reward_means = {}
    
    for edge in G.edges():
        edge_tuple = tuple(sorted(edge))
        # Random availability between 0.6 and 0.9
        availability_probs[edge_tuple] = np.random.uniform(0.6, 0.9)
        # Random reward mean between 0.3 and 0.8
        reward_means[edge_tuple] = np.random.uniform(0.3, 0.8)
    
    source = 0  # Top-left corner
    target = 8  # Bottom-right corner
    
    return G, availability_probs, reward_means, source, target