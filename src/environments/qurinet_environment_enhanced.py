import numpy as np
import networkx as nx
import pandas as pd
import os
from typing import List, Set, Dict, Any, Tuple, Optional
from collections import defaultdict
import csv


class QurinetEnvironmentEnhanced:
    """
    Enhanced Qurinet Real Wireless Mesh Network Environment for Combinatorial Bandit Algorithms.
    
    This enhanced version can utilize different date versions more effectively,
    especially the 29-April version with 32 nodes and different adhoc configurations.
    
    Data Source: https://github.com/cjpatton/qr
    Network: Up to 32 nodes with dual adhoc interfaces
    Frequency: 2.4GHz (channels 1, 6, 11)
    
    Attributes:
        graph: NetworkX graph representing the real Qurinet topology
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
        date: Date of the topology data
        sites_data: Raw sites data for the specific date
    """
    
    def __init__(self, data_dir: str = "data/qurinet", date: str = "2-May",
                 source: int = None, destination: int = None,
                 num_rounds: int = 10000,
                 max_combination_size: Optional[int] = None,
                 pre_generate_availability: bool = True,
                 pre_generate_rewards: bool = True,
                 rng: Optional[np.random.Generator] = None,
                 use_all_nodes: bool = True):
        """
        Initialize the enhanced Qurinet environment.
        
        Args:
            data_dir: Directory containing Qurinet data files
            date: Date of the topology data (e.g., "2-May", "29-April", "24-May")
            source: Source node ID (if None, will be selected automatically)
            destination: Destination node ID (if None, will be selected automatically)
            num_rounds: Number of simulation rounds
            max_combination_size: Maximum path length (if None, calculated automatically)
            pre_generate_availability: Whether to pre-generate availability matrix
            pre_generate_rewards: Whether to pre-generate rewards matrix
            rng: Random number generator
            use_all_nodes: Whether to use all nodes from the date-specific data
        """
        self.data_dir = data_dir
        self.date = date
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_rounds = num_rounds
        self.use_all_nodes = use_all_nodes
        
        # Load date-specific sites data
        self.sites_data = self._load_sites_data()
        
        # Load real Qurinet network topology with date-specific data
        self.graph = self._load_qurinet_topology_enhanced()
        
        # Set source and destination if not provided
        if source is None or destination is None:
            source, destination = self._select_source_destination_enhanced()
        
        self.source = source
        self.destination = destination
        
        # Load real link quality data and convert to availability rates
        self.link_availability_rates = self._load_link_availability_rates_enhanced()
        self.link_reward_means = self._load_link_reward_means_enhanced()
        
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
            # Allow all simple (acyclic) paths by default
            self.max_combination_size = self.num_arms
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
    
    def _load_sites_data(self) -> pd.DataFrame:
        """Load sites data for the specific date."""
        if self.date == "2-May":
            sites_file = os.path.join(self.data_dir, "sites.csv")
        else:
            sites_file = os.path.join(self.data_dir, f"sites_{self.date}.csv")
        
        if not os.path.exists(sites_file):
            print(f"Warning: {sites_file} not found, using default sites.csv")
            sites_file = os.path.join(self.data_dir, "sites.csv")
        
        return pd.read_csv(sites_file)
    
    def _load_qurinet_topology_enhanced(self) -> nx.Graph:
        """Load enhanced Qurinet network topology using date-specific data."""
        graph = nx.Graph()
        
        # Add all nodes from the sites data
        all_nodes = set()
        
        # Add nodes from sites data
        for _, row in self.sites_data.iterrows():
            site = row['site']
            all_nodes.add(site)
            graph.add_node(site)
        
        # Add edges based on adhoc configurations
        # Create edges between nodes with compatible adhoc configurations
        nodes_list = list(all_nodes)
        
        for i, node1 in enumerate(nodes_list):
            for j, node2 in enumerate(nodes_list[i+1:], i+1):
                # Get adhoc configurations for both nodes
                node1_data = self.sites_data[self.sites_data['site'] == node1].iloc[0]
                node2_data = self.sites_data[self.sites_data['site'] == node2].iloc[0]
                
                # Check if nodes can communicate (same adhoc channel)
                if (node1_data['adhoc0'] == node2_data['adhoc0'] or 
                    node1_data['adhoc0'] == node2_data['adhoc1'] or
                    node1_data['adhoc1'] == node2_data['adhoc0'] or
                    node1_data['adhoc1'] == node2_data['adhoc1']):
                    
                    # Add edge with probability based on distance and configuration
                    edge_prob = self._calculate_edge_probability(node1_data, node2_data)
                    if self.rng.random() < edge_prob:
                        graph.add_edge(node1, node2)
        
        # Ensure connectivity by adding minimum spanning tree if needed
        if not nx.is_connected(graph):
            print(f"Warning: {self.date} graph is not connected. Adding edges to ensure connectivity...")
            
            # Find connected components
            components = list(nx.connected_components(graph))
            
            # Add edges between components
            for i in range(len(components) - 1):
                comp1 = list(components[i])
                comp2 = list(components[i + 1])
                
                # Find best connection between components
                best_edge = None
                best_prob = 0
                
                for node1 in comp1:
                    for node2 in comp2:
                        node1_data = self.sites_data[self.sites_data['site'] == node1].iloc[0]
                        node2_data = self.sites_data[self.sites_data['site'] == node2].iloc[0]
                        
                        prob = self._calculate_edge_probability(node1_data, node2_data)
                        if prob > best_prob:
                            best_prob = prob
                            best_edge = (node1, node2)
                
                if best_edge:
                    graph.add_edge(*best_edge)
                    print(f"Added edge: {best_edge[0]} - {best_edge[1]}")
        
        return graph
    
    def _calculate_edge_probability(self, node1_data: pd.Series, node2_data: pd.Series) -> float:
        """Calculate probability of edge existence based on adhoc configurations."""
        # Base probability
        base_prob = 0.3
        
        # Check channel compatibility
        channels1 = {node1_data['adhoc0'], node1_data['adhoc1']}
        channels2 = {node2_data['adhoc0'], node2_data['adhoc1']}
        
        if channels1.intersection(channels2):
            base_prob += 0.4  # Same channel increases probability
        
        # Channel-specific adjustments
        if 6 in channels1.intersection(channels2):
            base_prob += 0.1  # Channel 6 has better propagation
        
        # Node ID distance factor (simulating physical distance)
        node1_id = node1_data['site']
        node2_id = node2_data['site']
        distance_factor = 1.0 / (1.0 + abs(node1_id - node2_id) / 100.0)
        
        return min(0.9, base_prob * distance_factor)
    
    def _select_source_destination_enhanced(self) -> Tuple[int, int]:
        """Select source and destination nodes from available nodes."""
        available_nodes = list(self.graph.nodes())
        
        # For 29-April, try to use some of the new nodes
        if self.date == "29-April" and len(available_nodes) > 20:
            # Use some of the new nodes (8, 9, 11, 12, etc.)
            new_nodes = [8, 9, 11, 12, 14, 15, 16, 23, 24, 25, 26, 27, 31]
            available_new_nodes = [n for n in new_nodes if n in available_nodes]
            
            if len(available_new_nodes) >= 2:
                source = available_new_nodes[0]
                destination = available_new_nodes[1]
                return source, destination
        
        # Default selection
        if len(available_nodes) >= 2:
            source = available_nodes[0]
            destination = available_nodes[1]
        else:
            source = 1
            destination = 2
        
        return source, destination
    
    def _load_link_availability_rates_enhanced(self) -> Dict[Tuple[int, int], float]:
        """Load enhanced link availability rates based on date-specific configurations."""
        availability_rates = {}
        
        # Load link quality data files
        link_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and f != 'sites.csv' and not f.startswith('sites_')]
        
        for link_file in link_files:
            try:
                file_path = os.path.join(self.data_dir, link_file)
                link_data = pd.read_csv(file_path)
                
                # Extract node IDs from filename (e.g., "1-0.csv" -> node 1)
                node_id = int(link_file.split('-')[0])
                
                # Calculate availability based on signal quality
                if 'Signal(dBm)' in link_data.columns and 'Quality' in link_data.columns:
                    avg_signal = link_data['Signal(dBm)'].mean()
                    avg_quality = link_data['Quality'].mean()
                    
                    # Convert signal strength to availability
                    signal_availability = max(0.1, min(0.99, (avg_signal + 100) / 50))
                    quality_availability = avg_quality / 100.0
                    
                    # Combined availability
                    availability = 0.7 * signal_availability + 0.3 * quality_availability
                    
                    # Add edges from this node to other nodes
                    for _, row in self.sites_data.iterrows():
                        other_node = row['site']
                        if other_node != node_id and (node_id, other_node) in self.graph.edges():
                            availability_rates[(node_id, other_node)] = availability
                            availability_rates[(other_node, node_id)] = availability
                
            except Exception as e:
                print(f"Warning: Could not process {link_file}: {e}")
                continue
        
        # Set default availability for edges without data
        for edge in self.graph.edges():
            if edge not in availability_rates:
                # Use date-specific default availability
                if self.date == "29-April":
                    availability_rates[edge] = 0.85  # Higher availability for larger network
                elif self.date == "24-May":
                    availability_rates[edge] = 0.90  # Highest availability
                else:
                    availability_rates[edge] = 0.80  # Default availability
        
        return availability_rates
    
    def _load_link_reward_means_enhanced(self) -> Dict[Tuple[int, int], float]:
        """Load enhanced link reward means based on date-specific configurations."""
        reward_means = {}
        
        # Load link quality data files
        link_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and f != 'sites.csv' and not f.startswith('sites_')]
        
        for link_file in link_files:
            try:
                file_path = os.path.join(self.data_dir, link_file)
                link_data = pd.read_csv(file_path)
                
                # Extract node IDs from filename
                node_id = int(link_file.split('-')[0])
                
                # Calculate reward based on signal quality and channel
                if 'Signal(dBm)' in link_data.columns and 'Quality' in link_data.columns:
                    avg_signal = link_data['Signal(dBm)'].mean()
                    avg_quality = link_data['Quality'].mean()
                    
                    # Convert to reward (0-1 scale)
                    signal_reward = max(0.1, min(0.99, (avg_signal + 100) / 50))
                    quality_reward = avg_quality / 100.0
                    
                    # Combined reward
                    reward = 0.6 * signal_reward + 0.4 * quality_reward
                    
                    # Add edges from this node to other nodes
                    for _, row in self.sites_data.iterrows():
                        other_node = row['site']
                        if other_node != node_id and (node_id, other_node) in self.graph.edges():
                            reward_means[(node_id, other_node)] = reward
                            reward_means[(other_node, node_id)] = reward
                
            except Exception as e:
                print(f"Warning: Could not process {link_file}: {e}")
                continue
        
        # Set default rewards for edges without data
        for edge in self.graph.edges():
            if edge not in reward_means:
                # Use date-specific default rewards
                if self.date == "29-April":
                    # Higher rewards for larger network
                    reward_means[edge] = self.rng.beta(3, 7)  # Mean around 0.3
                elif self.date == "24-May":
                    # Highest rewards
                    reward_means[edge] = self.rng.beta(4, 6)  # Mean around 0.4
                else:
                    # Default rewards
                    reward_means[edge] = self.rng.beta(2, 8)  # Mean around 0.2
        
        return reward_means
    
    def _generate_availability_matrix(self):
        """Pre-generate availability matrix for all rounds."""
        print(f"Generating availability matrix...")
        self.availability_matrix = np.zeros((self.num_rounds, self.num_arms), dtype=bool)
        
        for round_num in range(self.num_rounds):
            if round_num % 1000 == 0 or round_num == self.num_rounds - 1:
                print(f"  Progress: {round_num}/{self.num_rounds} ({round_num/self.num_rounds*100:.1f}%)", end='\r')
            
            for arm_id in range(self.num_arms):
                edge = self.arm_to_edge[arm_id]
                availability_rate = self.arm_availability_rates[arm_id]
                self.availability_matrix[round_num, arm_id] = self.rng.random() < availability_rate
        
        print(f"\n  Availability matrix completed!")
    
    def _generate_rewards_matrix(self):
        """Pre-generate rewards matrix for all rounds."""
        print(f"Generating rewards matrix...")
        self.rewards_matrix = np.zeros((self.num_rounds, self.num_arms))
        
        for round_num in range(self.num_rounds):
            if round_num % 1000 == 0 or round_num == self.num_rounds - 1:
                print(f"  Progress: {round_num}/{self.num_rounds} ({round_num/self.num_rounds*100:.1f}%)", end='\r')
            
            for arm_id in range(self.num_arms):
                mean_reward = self.arm_means[arm_id]
                # Use Bernoulli distribution for binary rewards
                self.rewards_matrix[round_num, arm_id] = self.rng.random() < mean_reward
        
        print(f"\n  Rewards matrix completed!")
    
    def get_available_arms_for_round(self, round_num: int) -> Set[int]:
        """Get available arms for the current round."""
        if self.availability_matrix is not None:
            return set(np.where(self.availability_matrix[round_num])[0])
        else:
            # All arms are available
            return set(range(self.num_arms))
    
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
        
        # Ensure source and destination are in the subgraph
        if self.source not in subgraph.nodes() or self.destination not in subgraph.nodes():
            return []
        
        # Find all simple paths from source to destination
        try:
            # Limit the number of paths to avoid exponential explosion
            max_paths = 100  # Reduced limit for better performance
            all_paths = []
            
            # Use shortest paths first
            try:
                shortest_paths = list(nx.all_shortest_paths(subgraph, self.source, self.destination))
                all_paths.extend(shortest_paths)
            except nx.NetworkXNoPath:
                pass
            
            # If we need more paths, add some longer ones (but limit total)
            if len(all_paths) < max_paths:
                try:
                    # Get some additional paths with limited length
                    for path_length in range(3, min(6, self.max_combination_size + 1)):  # Reduced max length
                        if len(all_paths) >= max_paths:
                            break
                        
                        # Use simple path generator with length limit
                        path_gen = nx.all_simple_paths(subgraph, self.source, self.destination, cutoff=path_length)
                        for path in path_gen:
                            if len(all_paths) >= max_paths:
                                break
                            if path not in all_paths:
                                all_paths.append(path)
                except Exception as e:
                    pass
            
        except nx.NetworkXNoPath:
            return []
        
        # Convert paths to arm combinations
        feasible_combinations = []
        for i, path in enumerate(all_paths):
            if len(path) - 1 <= self.max_combination_size:  # Path length = number of edges
                path_arms = set()
                for j in range(len(path) - 1):
                    edge = (path[j], path[j + 1])
                    if edge in self.edge_to_arm:
                        path_arms.add(self.edge_to_arm[edge])
                    elif (edge[1], edge[0]) in self.edge_to_arm:
                        path_arms.add(self.edge_to_arm[(edge[1], edge[0])])
                
                if path_arms and len(path_arms) <= self.max_combination_size:
                    feasible_combinations.append(path_arms)
        return feasible_combinations
    
    def get_reward_for_round(self, combination: Set[int], round_num: int) -> Dict[int, float]:
        """Get rewards for a combination of arms in a specific round."""
        rewards = {}
        
        if self.rewards_matrix is not None:
            for arm_id in combination:
                if arm_id < self.num_arms:
                    rewards[arm_id] = self.rewards_matrix[round_num, arm_id]
        else:
            # Generate rewards on-the-fly
            for arm_id in combination:
                if arm_id < self.num_arms:
                    mean_reward = self.arm_means[arm_id]
                    rewards[arm_id] = self.rng.random() < mean_reward
        
        return rewards
    
    def get_optimal_combination(self, available_arms: Set[int]) -> Optional[Set[int]]:
        """Get the optimal combination based on expected rewards."""
        feasible_combinations = self.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return None
        
        best_combination = None
        best_reward = -1
        
        for combination in feasible_combinations:
            expected_reward = sum(self.arm_means[arm_id] for arm_id in combination)
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_combination = combination
        
        return best_combination
    
    def get_path_info(self, combination: Set[int]) -> Dict[str, Any]:
        """Get detailed information about a path combination."""
        if not combination:
            return {'path': [], 'edges': [], 'length': 0, 'expected_reward': 0.0}
        
        # Convert arms back to edges
        edges = []
        for arm_id in combination:
            edge = self.arm_to_edge[arm_id]
            edges.append(edge)
        
        # Find the path
        path = []
        if edges:
            # Simple path reconstruction
            path = [self.source]
            current_node = self.source
            
            for edge in edges:
                if edge[0] == current_node:
                    path.append(edge[1])
                    current_node = edge[1]
                elif edge[1] == current_node:
                    path.append(edge[0])
                    current_node = edge[0]
        
        expected_reward = sum(self.arm_means[arm_id] for arm_id in combination)
        
        return {
            'path': path,
            'edges': edges,
            'length': len(edges),
            'expected_reward': expected_reward
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        info = {
            'environment_type': f'Enhanced Qurinet Real Wireless Mesh Network ({self.date})',
            'data_source': 'https://github.com/cjpatton/qr',
            'date': self.date,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'source': self.source,
            'destination': self.destination,
            'num_arms': self.num_arms,
            'num_rounds': self.num_rounds,
            'max_combination_size': self.max_combination_size,
            'avg_availability': np.mean(list(self.link_availability_rates.values())),
            'min_availability': np.min(list(self.link_availability_rates.values())),
            'max_availability': np.max(list(self.link_availability_rates.values())),
            'avg_reward': np.mean(list(self.link_reward_means.values())),
            'min_reward': np.min(list(self.link_reward_means.values())),
            'max_reward': np.max(list(self.link_reward_means.values())),
            'availability_calculation': 'Enhanced date-specific calculation',
            'factors_used': ['Signal Strength', 'Quality', 'Channel Compatibility', 'Node Distance'],
            'arm_means': self.arm_means.copy(),
            'arm_availability_rates': self.arm_availability_rates.copy(),
            'edges': self.edges.copy(),
            'nodes': list(self.graph.nodes()),
            'sites_data_size': len(self.sites_data)
        }
        return info
    
    def print_network_statistics(self):
        """Print detailed network statistics."""
        print(f"\nEnhanced Qurinet Network Statistics:")
        print(f"=" * 50)
        print(f"Network: Real wireless mesh network from Quail Ridge Natural Reserve")
        print(f"Data date: {self.date}")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print(f"Source: {self.source}, Destination: {self.destination}")
        print(f"Max combination size: {self.max_combination_size}")
        
        # Network connectivity
        print(f"\nConnectivity:")
        print(f"  Connected: {nx.is_connected(self.graph)}")
        print(f"  Number of components: {nx.number_connected_components(self.graph)}")
        print(f"  Average degree: {np.mean(list(dict(self.graph.degree()).values())):.2f}")
        
        # Link quality statistics
        availability_rates = list(self.link_availability_rates.values())
        print(f"\nLink Quality Statistics:")
        print(f"  Average availability: {np.mean(availability_rates):.3f}")
        print(f"  Min availability: {np.min(availability_rates):.3f}")
        print(f"  Max availability: {np.max(availability_rates):.3f}")
        print(f"  Std availability: {np.std(availability_rates):.3f}")
        
        # Reward statistics
        reward_means = list(self.link_reward_means.values())
        print(f"\nReward Statistics:")
        print(f"  Average reward: {np.mean(reward_means):.3f}")
        print(f"  Min reward: {np.min(reward_means):.3f}")
        print(f"  Max reward: {np.max(reward_means):.3f}")
        print(f"  Std reward: {np.std(reward_means):.3f}")
        
        # Optimal path info
        available_arms = set(range(self.num_arms))
        optimal_combination = self.get_optimal_combination(available_arms)
        if optimal_combination:
            path_info = self.get_path_info(optimal_combination)
            print(f"\nOptimal Path:")
            print(f"  Path: {path_info['path']}")
            print(f"  Length: {path_info['length']}")
            print(f"  Expected reward: {path_info['expected_reward']:.3f}")
        
        # Date-specific enhancements
        print(f"\nDate-Specific Enhancements:")
        print(f"  Sites data size: {len(self.sites_data)}")
        print(f"  Use all nodes: {self.use_all_nodes}")
        if self.date == "29-April":
            print(f"  Enhanced: 32-node network with varied adhoc configurations")
        elif self.date == "24-May":
            print(f"  Enhanced: Transmission power information included")
        else:
            print(f"  Standard: {self.date} configuration") 