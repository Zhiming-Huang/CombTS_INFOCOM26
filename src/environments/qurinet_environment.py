import numpy as np
import networkx as nx
import pandas as pd
import os
from typing import List, Set, Dict, Any, Tuple, Optional
from collections import defaultdict
import csv


class QurinetEnvironment:
    """
    Qurinet Real Wireless Mesh Network Environment for Combinatorial Bandit Algorithms.
    
    This environment uses real topology and link quality data from the Qurinet wireless
    mesh network deployed at Quail Ridge Natural Reserve.
    
    Data Source: https://github.com/cjpatton/qr
    Network: 20 nodes with dual adhoc interfaces
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
    """
    
    def __init__(self, data_dir: str = "data/qurinet", date: str = "2-May",
                 source: int = None, destination: int = None,
                 num_rounds: int = 10000,
                 max_combination_size: Optional[int] = None,
                 pre_generate_availability: bool = True,
                 pre_generate_rewards: bool = True,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the Qurinet environment.
        
        Args:
            data_dir: Directory containing Qurinet data files
            date: Date of the topology data (e.g., "2-May", "24-May")
            source: Source node ID (if None, will be selected automatically)
            destination: Destination node ID (if None, will be selected automatically)
            num_rounds: Number of simulation rounds
            max_combination_size: Maximum path length (if None, calculated automatically)
            pre_generate_availability: Whether to pre-generate availability matrix
            pre_generate_rewards: Whether to pre-generate rewards matrix
            rng: Random number generator
        """
        self.data_dir = data_dir
        self.date = date
        self.rng = rng if rng is not None else np.random.default_rng()
        self.num_rounds = num_rounds
        
        # Load real Qurinet network topology
        self.graph = self._load_qurinet_topology()
        
        # Set source and destination if not provided
        if source is None or destination is None:
            source, destination = self._select_source_destination()
        
        self.source = source
        self.destination = destination
        
        # Load real link quality data and convert to availability rates
        self.link_availability_rates = self._load_link_availability_rates()
        self.link_reward_means = self._load_link_reward_means()
        
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
    
    def _load_qurinet_topology(self) -> nx.Graph:
        """Load real Qurinet network topology from data files."""
        G = nx.Graph()
        
        # Load sites information
        sites_file = os.path.join(self.data_dir, "sites.csv")
        if not os.path.exists(sites_file):
            raise FileNotFoundError(f"Sites file not found: {sites_file}")
        
        sites_data = pd.read_csv(sites_file)
        nodes = sites_data['site'].tolist()
        
        # Add nodes to graph
        G.add_nodes_from(nodes)
        
        # Load link information from scan files
        links = set()
        
        for node in nodes:
            # Check both adhoc interfaces (0 and 1)
            for interface in [0, 1]:
                scan_file = os.path.join(self.data_dir, f"{node}-{interface}.csv")
                if os.path.exists(scan_file):
                    try:
                        scan_data = pd.read_csv(scan_file)
                        
                        # Extract links from scan data
                        for _, row in scan_data.iterrows():
                            essid = row['ESSID']
                            if essid != 'qurinet' and '-' in essid:  # Skip non-qurinet networks
                                # Parse ESSID to get target node and interface
                                # Format: "N-I" where N is node number, I is interface
                                parts = essid.split('-')
                                if len(parts) == 2:
                                    target_node = int(parts[0])
                                    target_interface = int(parts[1])
                                    
                                    # Add bidirectional link
                                    if target_node in nodes:
                                        link = (node, target_node)
                                        links.add(link)
                    except Exception as e:
                        print(f"Warning: Could not process {scan_file}: {e}")
        
        # Add edges to graph (remove self-loops and duplicates)
        clean_links = set()
        for link in links:
            if link[0] != link[1]:  # Remove self-loops
                clean_links.add(link)
        
        G.add_edges_from(clean_links)
        
        # Ensure connectivity by adding some edges if graph is disconnected
        if not nx.is_connected(G):
            print("Warning: Qurinet graph is not connected. Adding edges to ensure connectivity...")
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
                print(f"Added edge: {node1} - {node2}")
        
        return G
    
    def _load_link_availability_rates(self) -> Dict[Tuple[int, int], float]:
        """Set all link availability rates to 1.0 (all links always available)."""
        return {edge: 1.0 for edge in self.graph.edges()}
    
    def _signal_to_availability(self, signal_dbm: float) -> float:
        """Convert signal strength (dBm) to availability rate."""
        # Signal strength to availability mapping
        # -100dBm: 0.1 (very poor)
        # -80dBm: 0.3 (poor)
        # -60dBm: 0.6 (moderate)
        # -40dBm: 0.9 (good)
        # -20dBm: 0.99 (excellent)
        
        if signal_dbm >= -20:
            return 0.99
        elif signal_dbm >= -40:
            return 0.9
        elif signal_dbm >= -60:
            return 0.6
        elif signal_dbm >= -80:
            return 0.3
        else:
            return 0.1
    
    def _channel_congestion(self, channel: int) -> float:
        """Calculate channel congestion factor based on channel usage."""
        # Channel congestion analysis from data
        # Channel 1: most crowded (11 links), Channel 6: least crowded (3 links), Channel 11: medium (7 links)
        channel_counts = {1: 11, 6: 3, 11: 7}
        max_count = max(channel_counts.values())
        congestion = channel_counts[channel] / max_count
        # Higher congestion reduces availability
        return 1 - congestion * 0.3
    
    def _power_to_availability(self, txpwr: int) -> float:
        """Convert transmission power to availability factor."""
        # Transmission power range: 14-19 dBm
        if txpwr >= 18:
            return 0.95  # High power
        elif txpwr >= 16:
            return 0.85  # Medium power
        else:
            return 0.75  # Low power
    
    def _frequency_interference(self, freq1: float, freq2: float) -> float:
        """Calculate frequency interference factor between two frequencies."""
        # Frequency values: 2.412, 2.437, 2.462 GHz
        freq_diff = abs(freq1 - freq2)
        if freq_diff <= 0.025:  # Adjacent channels
            return 0.7
        elif freq_diff <= 0.050:  # Separated channels
            return 0.85
        else:  # Far channels
            return 0.95
    
    def _signal_to_distance_estimate(self, signal_dbm: float, txpwr: int = 16) -> float:
        """Estimate distance based on signal strength using free space path loss model."""
        # Free space path loss: PL = 20*log10(d) + 20*log10(f) + 32.44
        # where d is distance(km), f is frequency(GHz)
        path_loss = txpwr - signal_dbm
        distance = 10**((path_loss - 20*np.log10(2.4) - 32.44)/20)
        return distance
    
    def _estimate_etx(self, signal_dbm: float, quality: float) -> float:
        """Estimate ETX (Expected Transmission Count) based on signal strength and quality."""
        # ETX estimation based on empirical models for wireless networks
        # ETX = 1 / (forward_delivery_ratio * reverse_delivery_ratio)
        
        # Convert signal strength to delivery ratio (simplified model)
        # Signal strength ranges: -90 to -30 dBm
        # Quality ranges: 0 to 100
        
        # Normalize signal strength to [0, 1] range
        signal_norm = max(0, min(1, (signal_dbm + 90) / 60))  # -90 to -30 dBm
        
        # Normalize quality to [0, 1] range
        quality_norm = quality / 100.0
        
        # Estimate delivery ratio based on signal strength and quality
        # Higher signal strength and quality = higher delivery ratio
        delivery_ratio = 0.1 + 0.8 * (signal_norm * 0.7 + quality_norm * 0.3)
        
        # Ensure delivery ratio is in reasonable range
        delivery_ratio = max(0.1, min(0.95, delivery_ratio))
        
        # ETX = 1 / (delivery_ratio^2) for bidirectional link
        # Assuming symmetric forward and reverse delivery ratios
        etx = 1.0 / (delivery_ratio ** 2)
        
        # Ensure ETX is in reasonable range [1.0, 10.0]
        etx = max(1.0, min(10.0, etx))
        
        return etx
    
    def _improved_availability_calculation(self, signal_dbm: float, quality: float, 
                                         channel: int, frequency: float, 
                                         txpwr: Optional[int] = None) -> float:
        """Calculate improved availability using multiple factors."""
        # 1. Base signal strength availability
        signal_availability = self._signal_to_availability(signal_dbm)
        
        # 2. Quality metric
        quality_availability = quality / 100.0
        
        # 3. Channel congestion factor
        channel_availability = self._channel_congestion(channel)
        
        # 4. Transmission power factor (if available)
        power_availability = 1.0
        if txpwr is not None:
            power_availability = self._power_to_availability(txpwr)
        
        # 5. Distance-based factor (estimated from signal strength)
        distance = self._signal_to_distance_estimate(signal_dbm, txpwr or 16)
        distance_availability = max(0.5, 1.0 - distance * 0.1)  # Decrease with distance
        
        # 6. Frequency interference factor (assuming 2.4GHz band)
        freq_interference = self._frequency_interference(frequency, frequency)  # Same frequency = no interference
        
        # 7. Weighted combination
        combined_availability = (
            0.4 * signal_availability +
            0.2 * quality_availability +
            0.15 * channel_availability +
            0.1 * power_availability +
            0.1 * distance_availability +
            0.05 * freq_interference
        )
        
        return np.clip(combined_availability, 0.1, 0.99)
    
    def _load_link_reward_means(self) -> Dict[Tuple[int, int], float]:
        """Compute reward mean for each link using 1/ETX (Expected Transmission Count), normalized to [0.1, 0.99]."""
        reward_means = {}
        # Load sites information
        sites_file = os.path.join(self.data_dir, "sites.csv")
        sites_data = pd.read_csv(sites_file)
        nodes = sites_data['site'].tolist()
        # Load transmission power data if available
        txpwr_data = {}
        sites_24may_file = os.path.join(self.data_dir, "sites_24-May.csv")
        if os.path.exists(sites_24may_file):
            try:
                sites_24may_data = pd.read_csv(sites_24may_file)
                for _, row in sites_24may_data.iterrows():
                    site = row['site']
                    if pd.notna(row['adhoc0_txpwr']):
                        txpwr_data[site] = row['adhoc0_txpwr']
            except Exception as e:
                print(f"Warning: Could not load transmission power data: {e}")
        # Collect all estimated distances for each link
        link_distances = defaultdict(list)
        for node in nodes:
            for interface in [0, 1]:
                scan_file = os.path.join(self.data_dir, f"{node}-{interface}.csv")
                if os.path.exists(scan_file):
                    try:
                        scan_data = pd.read_csv(scan_file)
                        for _, row in scan_data.iterrows():
                            essid = row['ESSID']
                            if essid != 'qurinet' and '-' in essid:
                                parts = essid.split('-')
                                if len(parts) == 2:
                                    target_node = int(parts[0])
                                    if target_node in nodes:
                                        signal_dbm = row['Signal(dBm)']
                                        txpwr = txpwr_data.get(node, 16)
                                        distance = self._signal_to_distance_estimate(signal_dbm, txpwr)
                                        link1 = (node, target_node)
                                        link2 = (target_node, node)
                                        link_distances[link1].append(distance)
                                        link_distances[link2].append(distance)
                    except Exception as e:
                        print(f"Warning: Could not process {scan_file}: {e}")
        # Compute average distance for each edge
        edge_avg_dist = {}
        for edge in self.graph.edges():
            if link_distances[edge]:
                avg_dist = np.mean(link_distances[edge])
                edge_avg_dist[edge] = avg_dist
            else:
                edge_avg_dist[edge] = 1.0  # Default moderate distance
        
        # Calculate ETX-based rewards
        # ETX = Expected Transmission Count, reward = 1/ETX
        link_etx_data = defaultdict(list)
        
        # Collect signal strength and quality data for ETX estimation
        for node in nodes:
            for interface in [0, 1]:
                scan_file = os.path.join(self.data_dir, f"{node}-{interface}.csv")
                if os.path.exists(scan_file):
                    try:
                        scan_data = pd.read_csv(scan_file)
                        for _, row in scan_data.iterrows():
                            essid = row['ESSID']
                            if essid != 'qurinet' and '-' in essid:
                                parts = essid.split('-')
                                if len(parts) == 2:
                                    target_node = int(parts[0])
                                    if target_node in nodes:
                                        signal_dbm = row['Signal(dBm)']
                                        quality = row['Quality']
                                        link1 = (node, target_node)
                                        link2 = (target_node, node)
                                        
                                        # Estimate ETX based on signal strength and quality
                                        etx = self._estimate_etx(signal_dbm, quality)
                                        link_etx_data[link1].append(etx)
                                        link_etx_data[link2].append(etx)
                    except Exception as e:
                        print(f"Warning: Could not process {scan_file}: {e}")
        
        # Compute average ETX for each edge
        edge_avg_etx = {}
        for edge in self.graph.edges():
            if link_etx_data[edge]:
                avg_etx = np.mean(link_etx_data[edge])
                edge_avg_etx[edge] = avg_etx
            else:
                edge_avg_etx[edge] = 2.0  # Default moderate ETX
        
        # Calculate rewards as 1/ETX and normalize to [0.1, 0.99]
        all_etx_values = list(edge_avg_etx.values())
        min_etx, max_etx = min(all_etx_values), max(all_etx_values)
        
        for edge, avg_etx in edge_avg_etx.items():
            # Reward = 1/ETX (higher ETX = lower reward)
            inv_etx = 1.0 / avg_etx
            if max_etx > min_etx:
                # Normalize to [0.1, 0.99]
                norm_reward = 0.1 + 0.89 * (inv_etx - 1.0/max_etx) / (1.0/min_etx - 1.0/max_etx)
            else:
                norm_reward = 0.5  # All ETX values equal
            reward_means[edge] = float(np.clip(norm_reward, 0.1, 0.99))
        
        # Alternative methods (uncomment to use):
        # Method 1: Linear distance decay
        # all_distances = list(edge_avg_dist.values())
        # max_distance = max(all_distances) if all_distances else 1.0
        # for edge, avg_dist in edge_avg_dist.items():
        #     raw_reward = max(0.1, 1.0 - avg_dist / max_distance)
        #     reward_means[edge] = float(np.clip(raw_reward, 0.1, 0.99))
        
        # Method 2: Exponential decay
        # scale_factor = 2.0
        # for edge, avg_dist in edge_avg_dist.items():
        #     raw_reward = np.exp(-avg_dist / scale_factor)
        #     reward_means[edge] = float(np.clip(raw_reward, 0.1, 0.99))
        
        return reward_means
    
    def _select_source_destination(self) -> Tuple[int, int]:
        """Select source and destination nodes based on network characteristics."""
        nodes = list(self.graph.nodes())
        
        # Select nodes with high degree as source and destination
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Choose source and destination from high-degree nodes
        source = sorted_nodes[0][0]
        destination = sorted_nodes[1][0]
        
        return source, destination
    
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
        """All arms are always available since link availability is set to 1."""
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
        """Get comprehensive environment information."""
        info = {
            'environment_type': 'Qurinet Real Wireless Mesh Network (Improved Availability)',
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
            'availability_calculation': 'Improved multi-factor calculation',
            'factors_used': ['Signal Strength', 'Quality', 'Channel Congestion', 'Transmission Power', 'Distance', 'Frequency Interference'],
            'arm_means': self.arm_means.copy(),
            'arm_availability_rates': self.arm_availability_rates.copy(),
            'edges': self.edges.copy(),
            'nodes': list(self.graph.nodes())
        }
        return info
    
    def visualize_network(self, highlight_path: Optional[Set[int]] = None):
        """Visualize the Qurinet network topology."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 10))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
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
            node_sizes = [degrees[node] * 100 + 200 for node in self.graph.nodes()]
            
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                                 node_size=node_sizes)
            
            # Highlight source and destination
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.source], 
                                 node_color='green', node_size=700)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.destination], 
                                 node_color='red', node_size=700)
            
            # Add labels
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
            
            plt.title(f"Qurinet Real Wireless Mesh Network\n"
                     f"Date: {self.date}, Source: {self.source}, Destination: {self.destination}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for visualization")
    
    def print_network_statistics(self):
        """Print detailed network statistics."""
        print(f"\nQurinet Network Statistics:")
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
        
        # Print improved availability calculation info
        print(f"\nImproved Availability Calculation:")
        print(f"  Method: Multi-factor weighted calculation")
        print(f"  Factors: Signal Strength (40%), Quality (20%), Channel Congestion (15%)")
        print(f"  Additional: Transmission Power (10%), Distance (10%), Frequency Interference (5%)")
    
    def analyze_availability_factors(self) -> Dict[str, Any]:
        """Analyze the impact of different factors on availability calculation."""
        # Load sample data for analysis
        sample_data = []
        sites_file = os.path.join(self.data_dir, "sites.csv")
        sites_data = pd.read_csv(sites_file)
        nodes = sites_data['site'].tolist()
        
        # Load transmission power data
        txpwr_data = {}
        sites_24may_file = os.path.join(self.data_dir, "sites_24-May.csv")
        if os.path.exists(sites_24may_file):
            try:
                sites_24may_data = pd.read_csv(sites_24may_file)
                for _, row in sites_24may_data.iterrows():
                    site = row['site']
                    if pd.notna(row['adhoc0_txpwr']):
                        txpwr_data[site] = row['adhoc0_txpwr']
            except Exception:
                pass
        
        # Collect sample data
        for node in nodes[:5]:  # Analyze first 5 nodes
            scan_file = os.path.join(self.data_dir, f"{node}-0.csv")
            if os.path.exists(scan_file):
                try:
                    scan_data = pd.read_csv(scan_file)
                    for _, row in scan_data.iterrows():
                        if row['ESSID'] != 'qurinet' and '-' in row['ESSID']:
                            sample_data.append({
                                'signal_dbm': row['Signal(dBm)'],
                                'quality': row['Quality'],
                                'channel': row['Channel'],
                                'frequency': row['Frequency(Ghz)'],
                                'txpwr': txpwr_data.get(node)
                            })
                            if len(sample_data) >= 50:  # Limit sample size
                                break
                    if len(sample_data) >= 50:
                        break
                except Exception:
                    continue
        
        # Analyze factor impacts
        factor_analysis = {
            'signal_strength': [],
            'quality': [],
            'channel_congestion': [],
            'transmission_power': [],
            'distance': [],
            'frequency_interference': [],
            'improved_total': []
        }
        
        for sample in sample_data:
            # Calculate individual factors
            signal_avail = self._signal_to_availability(sample['signal_dbm'])
            quality_avail = sample['quality'] / 100.0
            channel_avail = self._channel_congestion(sample['channel'])
            power_avail = self._power_to_availability(sample['txpwr']) if sample['txpwr'] else 1.0
            distance = self._signal_to_distance_estimate(sample['signal_dbm'], sample['txpwr'] or 16)
            distance_avail = max(0.5, 1.0 - distance * 0.1)
            freq_avail = self._frequency_interference(sample['frequency'], sample['frequency'])
            
            # Calculate improved total
            improved_total = self._improved_availability_calculation(
                sample['signal_dbm'], sample['quality'], sample['channel'], 
                sample['frequency'], sample['txpwr']
            )
            
            factor_analysis['signal_strength'].append(signal_avail)
            factor_analysis['quality'].append(quality_avail)
            factor_analysis['channel_congestion'].append(channel_avail)
            factor_analysis['transmission_power'].append(power_avail)
            factor_analysis['distance'].append(distance_avail)
            factor_analysis['frequency_interference'].append(freq_avail)
            factor_analysis['improved_total'].append(improved_total)
        
        # Calculate statistics
        analysis_results = {}
        for factor, values in factor_analysis.items():
            if values:
                analysis_results[factor] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return analysis_results 