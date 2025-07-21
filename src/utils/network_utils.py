import networkx as nx
from typing import Dict, Tuple


def create_3x3_mesh_network() -> nx.Graph:
    """
    Create a 3x3 mesh network topology.
    
    Nodes are numbered as follows:
    0 -- 1 -- 2
    |    |    |
    3 -- 4 -- 5
    |    |    |
    6 -- 7 -- 8
    
    Returns:
        NetworkX graph representing the 3x3 mesh
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(9):
        G.add_node(i)
    
    # Add horizontal edges
    for row in range(3):
        for col in range(2):
            node1 = row * 3 + col
            node2 = row * 3 + col + 1
            G.add_edge(node1, node2)
    
    # Add vertical edges
    for row in range(2):
        for col in range(3):
            node1 = row * 3 + col
            node2 = (row + 1) * 3 + col
            G.add_edge(node1, node2)
    
    return G


def create_test_case_link_means() -> Dict[int, float]:
    """
    Create link means for the test case.
    
    The optimal path should be 0 -> 1 -> 2 -> 5 -> 8 
    with mean reward 0.9 for each link in this path.
    Other links have mean reward 0.8.
    
    Returns:
        Dictionary mapping link_id to mean reward
    """
    # Create the network to get the actual edge order
    G = create_3x3_mesh_network()
    edges = list(G.edges())
    
    # The optimal path is 0 -> 1 -> 2 -> 5 -> 8
    # We need to find the link IDs for these edges
    optimal_edges = [(0, 1), (1, 2), (2, 5), (5, 8)]
    optimal_links = set()
    
    for u, v in optimal_edges:
        if (u, v) in edges:
            link_id = edges.index((u, v))
        elif (v, u) in edges:
            link_id = edges.index((v, u))
        else:
            continue
        optimal_links.add(link_id)
    
    link_means = {}
    for link_id in range(12):  # 12 links in 3x3 mesh
        if link_id in optimal_links:
            link_means[link_id] = 0.9
        else:
            link_means[link_id] = 0.8
    
    return link_means


def create_availability_rates(rate: float = 0.8) -> Dict[int, float]:
    """
    Create availability rates for all links.
    
    Args:
        rate: Availability probability for each link (default: 0.8)
        
    Returns:
        Dictionary mapping link_id to availability rate
    """
    return {link_id: rate for link_id in range(12)}


def get_optimal_path_info() -> Dict[str, any]:
    """
    Get information about the optimal path for the test case.
    
    Returns:
        Dictionary containing optimal path information
    """
    # Get the optimal links dynamically
    link_means = create_test_case_link_means()
    optimal_links = {link_id for link_id, mean in link_means.items() if mean == 0.9}
    
    return {
        'optimal_path': optimal_links,  # Link IDs for optimal path
        'optimal_path_nodes': [0, 1, 2, 5, 8],  # Node sequence
        'expected_reward': len(optimal_links) * 0.9,  # 4 links * 0.9
        'description': 'Path from node 0 to node 8 via nodes 1, 2, 5'
    }


def print_network_info(G: nx.Graph, link_means: Dict[int, float]):
    """
    Print information about the network and link configuration.
    
    Args:
        G: NetworkX graph
        link_means: Dictionary mapping link_id to mean reward
    """
    print("Network Information:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("\nLink Configuration:")
    
    for i, (u, v) in enumerate(G.edges()):
        mean_reward = link_means.get(i, "N/A")
        print(f"Link {i}: {u}-{v} (mean reward: {mean_reward})")
    
    print(f"\nOptimal path information:")
    optimal_info = get_optimal_path_info()
    print(f"Optimal path links: {optimal_info['optimal_path']}")
    print(f"Optimal path nodes: {optimal_info['optimal_path_nodes']}")
    print(f"Expected reward: {optimal_info['expected_reward']}")
    print(f"Description: {optimal_info['description']}") 