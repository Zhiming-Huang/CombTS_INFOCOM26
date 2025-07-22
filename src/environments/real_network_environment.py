import numpy as np
import networkx as nx
from typing import List, Set, Dict, Any, Tuple, Optional
from collections import defaultdict
import random
import os
import json


class RealNetworkEnvironment:
    """
    Real Network Environment for Combinatorial Bandit Algorithms.
    
    This environment uses real network topologies with realistic link availability
    data based on empirical studies and literature.
    
    Attributes:
        graph: NetworkX graph representing the real network topology
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
    
    def __init__(self, network_type: str = "karate", source: int = None, destination: int = None,
                 link_availability_rates: Dict[Tuple[int, int], float] = None,
                 link_reward_means: Dict[Tuple[int, int], float] = None,
                 num_rounds: int = 10000,
                 max_combination_size: Optional[int] = None,
                 pre_generate_availability: bool = True,
                 pre_generate_rewards: bool = True,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the real network environment.
        
        Args:
            network_type: Type of real network ("karate", "les_miserables", "florentine", "internet_as")
            source: Source node ID (if None, will be selected automatically)
            destination: Destination node ID (if None, will be selected automatically)
            link_availability_rates: Dictionary mapping edges to availability rates
            link_reward_means: Dictionary mapping edges to reward means
            num_rounds: Number of simulation rounds
            max_combination_size: Maximum path length (if None, calculated automatically)
            pre_generate_availability: Whether to pre-generate availability matrix
            pre_generate_rewards: Whether to pre-generate rewards matrix
            rng: Random number generator
        """
        self.network_type = network_type
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_rounds = num_rounds
        
        # Load real network topology
        self.graph = self._load_real_network(network_type)
        
        # Set source and destination if not provided
        if source is None or destination is None:
            source, destination = self._select_source_destination()
        
        self.source = source
        self.destination = destination
        
        # Set realistic link availability rates based on network type
        if link_availability_rates is None:
            link_availability_rates = self._get_realistic_availability_rates()
        
        # Set realistic link reward means
        if link_reward_means is None:
            link_reward_means = self._get_realistic_reward_means()
        
        self.link_availability_rates = link_availability_rates
        self.link_reward_means = link_reward_means
        
        # Convert edges to arms (links)
        self.edges = list(self.graph.edges())
        self.num_arms = len(self.edges)
        
        # Create edge to arm index mapping
        self.edge_to_arm = {edge: i for i, edge in enumerate(self.edges)}
        self.arm_to_edge = {i: edge for i, edge in enumerate(self.edges)}
        
        # Set arm means based on link reward means
        self.arm_means = np.zeros(self.num_arms)
        self.arm_availability_rates = np.zeros(self.num_arms)
        
        for edge, mean_reward in self.link_reward_means.items():
            if edge in self.edge_to_arm:
                arm_id = self.edge_to_arm[edge]
                self.arm_means[arm_id] = mean_reward
                self.arm_availability_rates[arm_id] = self.link_availability_rates.get(edge, 0.8)
        
        # Calculate max combination size (maximum path length)
        if max_combination_size is None:
            # Find shortest path length as a reasonable upper bound
            try:
                shortest_path = nx.shortest_path(self.graph, self.source, self.destination)
                self.max_combination_size = len(shortest_path) - 1
            except nx.NetworkXNoPath:
                self.max_combination_size = min(10, self.num_arms)  # Default fallback
        else:
            self.max_combination_size = max_combination_size
        
        # Pre-generate matrices if requested
        if pre_generate_availability:
            self._generate_availability_matrix()
        else:
            self.availability_matrix = None
            
        if pre_generate_rewards:
            self._generate_rewards_matrix()
        else:
            self.rewards_matrix = None
    
    def _load_real_network(self, network_type: str) -> nx.Graph:
        """Load real network topology based on type."""
        if network_type == "karate":
            # Zachary's Karate Club network (34 nodes, 78 edges)
            # Real social network with known community structure
            return nx.karate_club_graph()
        
        elif network_type == "les_miserables":
            # Les Miserables character network (77 nodes, 254 edges)
            # Co-appearance network of characters in the novel
            return nx.les_miserables_graph()
        
        elif network_type == "florentine":
            # Florentine Families network (15 nodes, 20 edges)
            # Marriage and business ties between Florentine families
            return nx.florentine_families_graph()
        
        elif network_type == "internet_as":
            # Internet AS-level topology (simplified)
            # Autonomous System level Internet topology
            return self._create_internet_as_topology()
        
        elif network_type == "power_grid":
            # Power grid network (simplified)
            # Electrical power transmission network
            return self._create_power_grid_topology()
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def _create_internet_as_topology(self) -> nx.Graph:
        """Create a simplified Internet AS-level topology."""
        # Create a hierarchical network structure typical of Internet AS topology
        G = nx.Graph()
        
        # Add core nodes (Tier-1 ISPs)
        core_nodes = list(range(10))
        G.add_nodes_from(core_nodes)
        
        # Connect core nodes in a mesh-like structure
        for i in range(len(core_nodes)):
            for j in range(i+1, len(core_nodes)):
                if self.rng.random() < 0.3:  # 30% connection probability
                    G.add_edge(core_nodes[i], core_nodes[j])
        
        # Add regional nodes (Tier-2 ISPs)
        regional_nodes = list(range(10, 30))
        G.add_nodes_from(regional_nodes)
        
        # Connect regional nodes to core nodes
        for regional in regional_nodes:
            # Each regional connects to 1-3 core nodes
            num_connections = self.rng.integers(1, 4)
            core_connections = self.rng.choice(core_nodes, num_connections, replace=False)
            for core in core_connections:
                G.add_edge(regional, core)
        
        # Add local nodes (Tier-3 ISPs)
        local_nodes = list(range(30, 50))
        G.add_nodes_from(local_nodes)
        
        # Connect local nodes to regional nodes
        for local in local_nodes:
            # Each local connects to 1-2 regional nodes
            num_connections = self.rng.integers(1, 3)
            regional_connections = self.rng.choice(regional_nodes, num_connections, replace=False)
            for regional in regional_connections:
                G.add_edge(local, regional)
        
        # Ensure connectivity
        if not nx.is_connected(G):
            # Add some edges to make it connected
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        return G
    
    def _create_power_grid_topology(self) -> nx.Graph:
        """Create a simplified power grid topology."""
        # Create a power grid network with generation, transmission, and distribution
        G = nx.Graph()
        
        # Add generation nodes (power plants)
        generation_nodes = list(range(5))
        G.add_nodes_from(generation_nodes)
        
        # Add transmission nodes (substations)
        transmission_nodes = list(range(5, 20))
        G.add_nodes_from(transmission_nodes)
        
        # Add distribution nodes (local substations)
        distribution_nodes = list(range(20, 40))
        G.add_nodes_from(distribution_nodes)
        
        # Connect generation to transmission
        for gen in generation_nodes:
            # Each generator connects to 2-4 transmission nodes
            num_connections = self.rng.integers(2, 5)
            trans_connections = self.rng.choice(transmission_nodes, num_connections, replace=False)
            for trans in trans_connections:
                G.add_edge(gen, trans)
        
        # Connect transmission nodes in a mesh
        for i, trans1 in enumerate(transmission_nodes):
            for trans2 in transmission_nodes[i+1:]:
                if self.rng.random() < 0.4:  # 40% connection probability
                    G.add_edge(trans1, trans2)
        
        # Connect transmission to distribution
        for dist in distribution_nodes:
            # Each distribution connects to 1-3 transmission nodes
            num_connections = self.rng.integers(1, 4)
            trans_connections = self.rng.choice(transmission_nodes, num_connections, replace=False)
            for trans in trans_connections:
                G.add_edge(dist, trans)
        
        # Ensure connectivity
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        return G
    
    def _select_source_destination(self) -> Tuple[int, int]:
        """Select source and destination nodes based on network characteristics."""
        nodes = list(self.graph.nodes())
        
        if self.network_type == "karate":
            # In Karate Club, nodes 0 and 33 are the two main leaders
            return 0, 33
        
        elif self.network_type == "les_miserables":
            # Select nodes with high degree (central characters)
            degrees = dict(self.graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[0][0], sorted_nodes[1][0]
        
        elif self.network_type == "florentine":
            # Select nodes with high degree
            degrees = dict(self.graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[0][0], sorted_nodes[1][0]
        
        elif self.network_type == "internet_as":
            # Select core nodes as source and destination
            return 0, 5  # Two different core nodes
        
        elif self.network_type == "power_grid":
            # Select generation and distribution nodes
            return 0, 25  # Generation to distribution
        
        else:
            # Default: select random nodes
            return self.rng.choice(nodes, 2, replace=False)
    
    def _get_realistic_availability_rates(self) -> Dict[Tuple[int, int], float]:
        """Get realistic link availability rates based on network type and literature."""
        availability_rates = {}
        
        if self.network_type == "karate":
            # Social network: high availability (0.85-0.95)
            # Based on social network reliability studies
            for edge in self.graph.edges():
                availability_rates[edge] = self.rng.uniform(0.85, 0.95)
        
        elif self.network_type == "les_miserables":
            # Character network: moderate-high availability (0.80-0.90)
            for edge in self.graph.edges():
                availability_rates[edge] = self.rng.uniform(0.80, 0.90)
        
        elif self.network_type == "florentine":
            # Family network: high availability (0.90-0.98)
            # Family ties are typically very reliable
            for edge in self.graph.edges():
                availability_rates[edge] = self.rng.uniform(0.90, 0.98)
        
        elif self.network_type == "internet_as":
            # Internet AS: variable availability based on tier
            # Core links: 0.99-0.999 (very high)
            # Regional links: 0.95-0.99 (high)
            # Local links: 0.85-0.95 (moderate-high)
            for edge in self.graph.edges():
                u, v = edge
                if u < 10 and v < 10:  # Core links
                    availability_rates[edge] = self.rng.uniform(0.99, 0.999)
                elif (u < 10 and 10 <= v < 30) or (v < 10 and 10 <= u < 30):  # Core-regional
                    availability_rates[edge] = self.rng.uniform(0.95, 0.99)
                elif (10 <= u < 30 and 10 <= v < 30):  # Regional links
                    availability_rates[edge] = self.rng.uniform(0.95, 0.99)
                else:  # Local links
                    availability_rates[edge] = self.rng.uniform(0.85, 0.95)
        
        elif self.network_type == "power_grid":
            # Power grid: very high availability (0.995-0.999)
            # Based on power grid reliability standards
            for edge in self.graph.edges():
                availability_rates[edge] = self.rng.uniform(0.995, 0.999)
        
        else:
            # Default: moderate availability
            for edge in self.graph.edges():
                availability_rates[edge] = self.rng.uniform(0.80, 0.90)
        
        return availability_rates
    
    def _get_realistic_reward_means(self) -> Dict[Tuple[int, int], float]:
        """Get realistic link reward means based on network type."""
        reward_means = {}
        
        if self.network_type == "karate":
            # Social network: rewards based on social influence
            # Central nodes have higher rewards
            degrees = dict(self.graph.degree())
            for edge in self.graph.edges():
                u, v = edge
                avg_degree = (degrees[u] + degrees[v]) / 2
                # Normalize degree to [0.6, 0.9] range
                reward = 0.6 + 0.3 * (avg_degree / max(degrees.values()))
                reward_means[edge] = reward
        
        elif self.network_type == "les_miserables":
            # Character network: rewards based on character importance
            # High-degree characters (main characters) have higher rewards
            degrees = dict(self.graph.degree())
            for edge in self.graph.edges():
                u, v = edge
                avg_degree = (degrees[u] + degrees[v]) / 2
                reward = 0.5 + 0.4 * (avg_degree / max(degrees.values()))
                reward_means[edge] = reward
        
        elif self.network_type == "florentine":
            # Family network: rewards based on family wealth/influence
            # Assume some families are more influential
            family_importance = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}
            for edge in self.graph.edges():
                u, v = edge
                importance_u = family_importance.get(u, 0.5)
                importance_v = family_importance.get(v, 0.5)
                reward = (importance_u + importance_v) / 2
                reward_means[edge] = reward
        
        elif self.network_type == "internet_as":
            # Internet AS: rewards based on bandwidth/capacity
            # Core links have higher capacity
            for edge in self.graph.edges():
                u, v = edge
                if u < 10 and v < 10:  # Core links
                    reward_means[edge] = self.rng.uniform(0.8, 0.95)
                elif (u < 10 and 10 <= v < 30) or (v < 10 and 10 <= u < 30):  # Core-regional
                    reward_means[edge] = self.rng.uniform(0.7, 0.85)
                elif (10 <= u < 30 and 10 <= v < 30):  # Regional links
                    reward_means[edge] = self.rng.uniform(0.6, 0.8)
                else:  # Local links
                    reward_means[edge] = self.rng.uniform(0.5, 0.7)
        
        elif self.network_type == "power_grid":
            # Power grid: rewards based on transmission capacity
            # High-voltage transmission lines have higher capacity
            for edge in self.graph.edges():
                u, v = edge
                if u < 5 or v < 5:  # Generation links
                    reward_means[edge] = self.rng.uniform(0.8, 0.95)
                elif (5 <= u < 20 and 5 <= v < 20):  # Transmission links
                    reward_means[edge] = self.rng.uniform(0.7, 0.9)
                else:  # Distribution links
                    reward_means[edge] = self.rng.uniform(0.6, 0.8)
        
        else:
            # Default: uniform rewards
            for edge in self.graph.edges():
                reward_means[edge] = self.rng.uniform(0.6, 0.9)
        
        return reward_means
    
    def _generate_availability_matrix(self):
        """Pre-generate availability matrix for all rounds."""
        print(f"Generating availability matrix: {self.num_rounds} rounds × {self.num_arms} arms")
        self.availability_matrix = np.zeros((self.num_rounds, self.num_arms), dtype=bool)
        
        for round_num in range(self.num_rounds):
            for arm_id in range(self.num_arms):
                edge = self.arm_to_edge[arm_id]
                availability_rate = self.arm_availability_rates[arm_id]
                self.availability_matrix[round_num, arm_id] = self.rng.random() < availability_rate
    
    def _generate_rewards_matrix(self):
        """Pre-generate rewards matrix for all rounds."""
        print(f"Generating rewards matrix: {self.num_rounds} rounds × {self.num_arms} arms")
        self.rewards_matrix = np.zeros((self.num_rounds, self.num_arms))
        
        for round_num in range(self.num_rounds):
            for arm_id in range(self.num_arms):
                mean_reward = self.arm_means[arm_id]
                # Use Bernoulli distribution for binary rewards
                self.rewards_matrix[round_num, arm_id] = self.rng.random() < mean_reward
    
    def get_available_arms_for_round(self, round_num: int) -> Set[int]:
        """Get available arms for a specific round."""
        if self.availability_matrix is not None:
            return set(np.where(self.availability_matrix[round_num])[0])
        else:
            # Generate on-the-fly
            available_arms = set()
            for arm_id in range(self.num_arms):
                edge = self.arm_to_edge[arm_id]
                availability_rate = self.arm_availability_rates[arm_id]
                if self.rng.random() < availability_rate:
                    available_arms.add(arm_id)
            return available_arms
    
    def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]:
        """Get all feasible path combinations from available arms."""
        if not available_arms:
            return []
        
        # Convert available arms back to edges
        available_edges = set()
        for arm_id in available_arms:
            edge = self.arm_to_edge[arm_id]
            available_edges.add(edge)
        
        # Create subgraph with only available edges
        subgraph = self.graph.edge_subgraph(available_edges)
        
        # Find all simple paths from source to destination
        try:
            all_paths = list(nx.all_simple_paths(subgraph, self.source, self.destination))
        except nx.NetworkXNoPath:
            return []
        
        # Convert paths to arm combinations
        feasible_combinations = []
        for path in all_paths:
            if len(path) - 1 <= self.max_combination_size:  # Path length = number of edges
                path_arms = set()
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self.edge_to_arm:
                        path_arms.add(self.edge_to_arm[edge])
                    elif (edge[1], edge[0]) in self.edge_to_arm:
                        path_arms.add(self.edge_to_arm[(edge[1], edge[0])])
                
                if path_arms and len(path_arms) <= self.max_combination_size:
                    feasible_combinations.append(path_arms)
        
        return feasible_combinations
    
    def get_optimal_combination(self, available_arms: Set[int]) -> Optional[Set[int]]:
        """Get the optimal path combination from available arms."""
        feasible_combinations = self.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return None
        
        # Find combination with highest expected reward
        best_combination = None
        best_reward = -1
        
        for combination in feasible_combinations:
            expected_reward = sum(self.arm_means[arm_id] for arm_id in combination)
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_combination = combination
        
        return best_combination
    
    def get_reward_for_round(self, combination: Set[int], round_num: int) -> Dict[int, float]:
        """Get rewards for a combination of arms in a specific round."""
        rewards = {}
        
        if self.rewards_matrix is not None:
            for arm_id in combination:
                rewards[arm_id] = self.rewards_matrix[round_num, arm_id]
        else:
            # Generate on-the-fly
            for arm_id in combination:
                mean_reward = self.arm_means[arm_id]
                rewards[arm_id] = self.rng.random() < mean_reward
        
        return rewards
    
    def get_path_info(self, combination: Set[int]) -> Dict[str, Any]:
        """Get information about a path combination."""
        if not combination:
            return {'path': [], 'edges': [], 'length': 0, 'expected_reward': 0}
        
        # Convert arms back to edges
        path_edges = []
        for arm_id in combination:
            edge = self.arm_to_edge[arm_id]
            path_edges.append(edge)
        
        # Find the actual path through the network
        path = self._find_path_from_edges(path_edges)
        
        # Calculate expected reward
        expected_reward = sum(self.arm_means[arm_id] for arm_id in combination)
        
        return {
            'path': path,
            'edges': path_edges,
            'length': len(combination),
            'expected_reward': expected_reward
        }
    
    def _find_path_from_edges(self, edges: List[Tuple[int, int]]) -> List[int]:
        """Find the actual path through the network given a set of edges."""
        if not edges:
            return []
        
        # Create a subgraph with only the given edges
        subgraph = self.graph.edge_subgraph(edges)
        
        # Try to find a path from source to destination
        try:
            path = nx.shortest_path(subgraph, self.source, self.destination)
            return path
        except nx.NetworkXNoPath:
            # If no path exists, return empty list
            return []
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the environment."""
        return {
            'network_type': self.network_type,
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
        """Visualize the network topology."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 10))
            
            # Use appropriate layout based on network type
            if self.network_type == "karate":
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif self.network_type == "les_miserables":
                pos = nx.spring_layout(self.graph, k=2, iterations=100)
            elif self.network_type == "florentine":
                pos = nx.spring_layout(self.graph, k=3, iterations=50)
            else:
                pos = nx.spring_layout(self.graph)
            
            # Draw all edges
            nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray')
            
            # Highlight optimal path if provided
            if highlight_path:
                highlight_edges = [self.arm_to_edge[arm] for arm in highlight_path 
                                 if arm in self.arm_to_edge]
                nx.draw_networkx_edges(self.graph, pos, edgelist=highlight_edges, 
                                     edge_color='red', width=3)
            
            # Draw nodes with size based on degree
            degrees = dict(self.graph.degree())
            node_sizes = [degrees[node] * 50 + 100 for node in self.graph.nodes()]
            
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                                 node_size=node_sizes)
            
            # Highlight source and destination
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.source], 
                                 node_color='green', node_size=700)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.destination], 
                                 node_color='red', node_size=700)
            
            # Add labels
            nx.draw_networkx_labels(self.graph, pos, font_size=8)
            
            plt.title(f"Real Network: {self.network_type.title()} "
                     f"(Source: {self.source}, Destination: {self.destination})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for visualization")
    
    def save_network_data(self, filepath: str):
        """Save network data to JSON file."""
        data = {
            'network_type': self.network_type,
            'source': self.source,
            'destination': self.destination,
            'num_rounds': self.num_rounds,
            'max_combination_size': self.max_combination_size,
            'edges': self.edges,
            'link_availability_rates': {str(k): v for k, v in self.link_availability_rates.items()},
            'link_reward_means': {str(k): v for k, v in self.link_reward_means.items()},
            'arm_means': self.arm_means.tolist(),
            'arm_availability_rates': self.arm_availability_rates.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Network data saved to {filepath}")
    
    @classmethod
    def load_network_data(cls, filepath: str, rng: Optional[np.random.Generator] = None):
        """Load network data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the network
        env = cls(
            network_type=data['network_type'],
            source=data['source'],
            destination=data['destination'],
            num_rounds=data['num_rounds'],
            max_combination_size=data['max_combination_size'],
            rng=rng
        )
        
        # Override with loaded data
        env.link_availability_rates = {eval(k): v for k, v in data['link_availability_rates'].items()}
        env.link_reward_means = {eval(k): v for k, v in data['link_reward_means'].items()}
        env.arm_means = np.array(data['arm_means'])
        env.arm_availability_rates = np.array(data['arm_availability_rates'])
        
        return env 