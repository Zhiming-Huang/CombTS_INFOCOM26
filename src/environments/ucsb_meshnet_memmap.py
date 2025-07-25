import numpy as np
import networkx as nx
import os
import tempfile
from typing import List, Tuple, Dict, Any, Optional

class UCSBMeshnetMemmapEnvironment:
    def __init__(self, neighbortable_files: List[str], routes_per_minute: int = 4,
                 source: str = None, destination: str = None, pre_generate_rewards: bool = True,
                 use_persistent_files: bool = False, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None,
                 max_feasible_combinations: int = 50, max_path_length: int = 8, max_paths_per_algorithm: int = 25):
        """
        Args:
            neighbortable_files: List of neighbortable file paths (sorted by timestamp)
            routes_per_minute: Number of rounds per minute (per neighbortable)
            source: Source node IP (str)
            destination: Destination node IP (str)
            pre_generate_rewards: Whether to pre-generate rewards matrix
            use_persistent_files: Whether to use persistent memmap files
            seed: Random seed
            rng: NumPy random generator (if provided, overrides seed)
            max_feasible_combinations: Maximum number of feasible combinations to return
            max_path_length: Maximum path length (number of hops)
            max_paths_per_algorithm: Maximum paths to find per path-finding algorithm
        """
        self.neighbortable_files = neighbortable_files
        self.routes_per_minute = routes_per_minute
        self.num_timestamps = len(neighbortable_files)
        self.num_rounds = self.num_timestamps * routes_per_minute
        self.pre_generate_rewards = pre_generate_rewards
        self.use_persistent_files = use_persistent_files
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        
        # Feasible combination control parameters
        self.max_feasible_combinations = max_feasible_combinations
        self.max_path_length = max_path_length
        self.max_paths_per_algorithm = max_paths_per_algorithm

        # Parse all unique nodes
        self.nodes = set()
        for f in neighbortable_files:
            with open(f, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    self.nodes.add(parts[0])
                    for i in range(1, len(parts), 2):
                        self.nodes.add(parts[i])
        self.nodes = sorted(self.nodes)

        # Choose source and destination if not given
        if source is None or destination is None:
            self.source, self.destination = self.nodes[0], self.nodes[-1]
        else:
            self.source, self.destination = source, destination

        # Precompute all ETTs for normalization
        self.all_etts = []
        for f in neighbortable_files:
            with open(f, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    for i in range(2, len(parts), 2):
                        try:
                            ett = float(parts[i])
                            if ett < 1000:
                                self.all_etts.append(ett)
                        except Exception:
                            continue
        self.ett_min = min(self.all_etts)
        self.ett_max = max(self.all_etts)

        # Pre-generate topology and rewards for each round
        self._generate_memmaps()

    def _generate_memmaps(self):
        # Setup memmap files
        if self.use_persistent_files:
            self.topology_file = f"ucsb_topology_{self.num_rounds}.dat"
            self.rewards_file = f"ucsb_rewards_{self.num_rounds}.dat"
            self.avg_rewards_file = f"ucsb_avg_rewards_{self.num_rounds}.dat"
        else:
            self.topology_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat').name
            self.rewards_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat').name
            self.avg_rewards_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat').name

        # For each round, store adjacency matrix (N x N, bool) and reward matrix (N x N, float)
        # Also store average rewards matrix (N x N, float) for expected reward calculation
        N = len(self.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.topology_memmap = np.memmap(self.topology_file, dtype=bool, mode='w+', shape=(self.num_rounds, N, N))
        self.rewards_memmap = np.memmap(self.rewards_file, dtype=float, mode='w+', shape=(self.num_rounds, N, N))
        self.avg_rewards_memmap = np.memmap(self.avg_rewards_file, dtype=float, mode='w+', shape=(N, N))

        # Initialize average rewards tracking
        link_reward_sums = np.zeros((N, N))
        link_count = np.zeros((N, N))

        for t_idx, f in enumerate(self.neighbortable_files):
            # Build graph for this timestamp
            G = nx.Graph()
            with open(f, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    src = parts[0]
                    for i in range(1, len(parts), 2):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                        except Exception:
                            continue
                        if ett < 1000:
                            G.add_edge(src, dst, ett=ett)
            # ETT normalized to reward mean
            for r in range(self.routes_per_minute):
                round_idx = t_idx * self.routes_per_minute + r
                for i, u in enumerate(self.nodes):
                    for j, v in enumerate(self.nodes):
                        if G.has_edge(u, v):
                            ett = G[u][v]['ett']
                            # Directly use ETT-based reward, ensuring it's in [0,1] range
                            reward = 1.0 - (ett - self.ett_min) / (self.ett_max - self.ett_min + 1e-8)
                            reward = np.clip(reward, 0.0, 1.0)  # Ensure reward is in [0,1]
                            self.topology_memmap[round_idx, i, j] = True
                            self.rewards_memmap[round_idx, i, j] = reward
                            
                            # Track for average calculation
                            link_reward_sums[i, j] += reward
                            link_count[i, j] += 1
                        else:
                            self.topology_memmap[round_idx, i, j] = False
                            self.rewards_memmap[round_idx, i, j] = 0.0

        # Calculate and store average rewards
        for i in range(N):
            for j in range(N):
                if link_count[i, j] > 0:
                    self.avg_rewards_memmap[i, j] = link_reward_sums[i, j] / link_count[i, j]
                else:
                    self.avg_rewards_memmap[i, j] = 0.0

        # Count all directed links (node pairs) that actually appeared
        self.arms = set()
        for f in self.neighbortable_files:
            with open(f, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    src = parts[0]
                    for i in range(1, len(parts), 2):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                        except Exception:
                            continue
                        if ett < 1000:
                            self.arms.add((src, dst))
        self.arms = sorted(self.arms)
        self.num_arms = len(self.arms)
        self.arm_to_link = {i: arm for i, arm in enumerate(self.arms)}
        self.link_to_arm = {arm: i for i, arm in enumerate(self.arms)}
        self.max_combination_size = len(self.nodes) - 1

    def get_available_links_for_round(self, round_idx: int) -> np.ndarray:
        return self.topology_memmap[round_idx]

    def get_reward_for_link(self, u: str, v: str, round_idx: int) -> float:
        i, j = self.node_idx[u], self.node_idx[v]
        return self.rewards_memmap[round_idx, i, j]

    def get_available_arms_for_round(self, round_idx: int):
        adj = self.topology_memmap[round_idx]
        available = []
        for idx, (u, v) in enumerate(self.arms):
            i, j = self.node_idx[u], self.node_idx[v]
            if adj[i, j]:
                available.append(idx)
        return available

    def get_reward_for_round(self, arm: int, round_idx: int) -> float:
        u, v = self.arm_to_link[arm]
        i, j = self.node_idx[u], self.node_idx[v]
        return self.rewards_memmap[round_idx, i, j]
    
    def get_expected_reward_for_path(self, path_arms: set, round_idx: int) -> float:
        """
        Calculate the expected reward for a path (set of arm indices).
        
        Args:
            path_arms: Set of arm indices representing a path
            round_idx: Current round index
            
        Returns:
            Expected reward for the path
        """
        total_expected_reward = 0.0
        for arm in path_arms:
            u, v = self.arm_to_link[arm]
            
            # Use the pre-computed average reward for this link
            avg_reward = self.avg_rewards_memmap[self.node_idx[u], self.node_idx[v]]
            total_expected_reward += avg_reward
        
        return total_expected_reward
    
    def _get_ett_for_link(self, u: str, v: str, round_idx: int) -> float:
        """
        Get the ETT value for a link at a specific round.
        This is used to calculate expected rewards when pre_generate_rewards=True.
        """
        # Find which timestamp file this round corresponds to
        timestamp_idx = round_idx // self.routes_per_minute
        if timestamp_idx >= len(self.neighbortable_files):
            return None
            
        # Read the ETT value from the corresponding neighbortable file
        neighbortable_file = self.neighbortable_files[timestamp_idx]
        try:
            with open(neighbortable_file, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    src = parts[0]
                    for i in range(1, len(parts), 2):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000 and ((src == u and dst == v) or (src == v and dst == u)):
                                return ett
                        except Exception:
                            continue
        except Exception:
            pass
        
        return None
    
    def get_optimal_path_expected_reward(self, round_idx: int) -> float:
        """
        Find the optimal path and return its expected reward for the given round.
        
        Args:
            round_idx: Current round index
            
        Returns:
            Expected reward of the optimal path
        """
        available_arms = set(self.get_available_arms_for_round(round_idx))
        feasible_combinations = self.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            return 0.0
        
        # Find the path with maximum expected reward
        max_expected_reward = 0.0
        for path_arms in feasible_combinations:
            expected_reward = self.get_expected_reward_for_path(path_arms, round_idx)
            if expected_reward > max_expected_reward:
                max_expected_reward = expected_reward
        
        return max_expected_reward

    def get_feasible_combinations(self, available_arms: set) -> list:
        """
        Get feasible path combinations from available arms (arm indices).
        Returns: List of sets of arm indices, each set is a path from source to destination.
        """
        if not available_arms:
            return []
        
        # Convert available arms to (u, v) edges
        available_edges = set(self.arm_to_link[arm] for arm in available_arms)
        
        # Build subgraph using undirected graph
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(available_edges)
        
        # Find all simple paths from source to destination
        try:
            # Find all simple paths up to max_path_length hops
            all_paths = list(nx.all_simple_paths(G, self.source, self.destination, cutoff=self.max_path_length))
            
            # Filter paths to ensure they have at least 2 nodes (1 edge)
            valid_paths = [path for path in all_paths if len(path) >= 2]
            
            # Limit the number of paths to avoid computational explosion
            if len(valid_paths) > self.max_paths_per_algorithm:
                # Prioritize shorter paths
                valid_paths.sort(key=len)
                valid_paths = valid_paths[:self.max_paths_per_algorithm]
                
        except nx.NetworkXNoPath:
            valid_paths = []
        except Exception as e:
            print(f"Warning: Error finding paths: {e}")
            valid_paths = []
        
        # Convert paths to arm index sets
        feasible_combinations = []
        for path in valid_paths:
            path_arms = set()
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in self.link_to_arm:
                    path_arms.add(self.link_to_arm[edge])
                elif (edge[1], edge[0]) in self.link_to_arm:
                    path_arms.add(self.link_to_arm[(edge[1], edge[0])])
            
            # Only add if we have valid arms and path is not too long
            if path_arms and len(path_arms) <= self.max_combination_size:
                feasible_combinations.append(path_arms)
        
        return feasible_combinations

    def get_nodes(self) -> List[str]:
        return self.nodes

    def get_source_destination(self) -> Tuple[str, str]:
        return self.source, self.destination

    def cleanup(self):
        os.remove(self.topology_file)
        os.remove(self.rewards_file)
        os.remove(self.avg_rewards_file) 