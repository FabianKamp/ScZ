import networkx as nx
import numpy as np
import pandas as pd

#%% This script contains useful functions that ease the work with network x
#TODO: Define conversion functions for all functions that are defined in network.py

def shortest_path_length(adjacency):
    """
    Takes Functional Connectivity Matrix and returns the shortest path length between all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: NxN dimensional array of shortest pathlengths
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Adjacency matrix must be pd.Dataframe or np.ndarray."
    adjacency = np.asarray(adjacency)
    N = adjacency.shape[0]
    G = nx.from_numpy_matrix(adjacency)
    shortest_path = list(nx.shortest_path_length(G, weight='weight'))
    shortest_path_matrix=np.zeros((N,N))

    for n, tup in enumerate(shortest_path):             # Shortestpath is list of tuples one for each region n
        for m, val in tup[1].items():                   # Element [1] in the tuple contain dictionary of pathlength of n to other regions
            shortest_path_matrix[n,m] = val             # Extract value from dictionary and puts it into matrix

    return shortest_path_matrix

def weighted_traingles_iter(adjacency, normalize=True):
    # TODO change division by degree.
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Adjacency matrix must be pd.Dataframe or np.ndarray."    # Checks format so function can be used alone.
    adjacency = np.asarray(adjacency)

    G = nx.from_numpy_matrix(adjacency)

    if normalize:   # Normalizes if set to True
        max_weight = max(d.get('weight', 1) for u, v, d in G.edges(data=True))
    else:
        max_weight = 1

    nodes_nbrs = G.adj.items()
    def wt(u, v):
        return G[u][v].get('weight', 1) / max_weight

    for i, nbrs in nodes_nbrs:
        inbrs = set(nbrs) - {i}
        weighted_triangles = 0
        seen = set()
        for j in inbrs:
            seen.add(j)
            # This prevents double counting.
            jnbrs = set(G[j]) - seen
            # Only compute the edge weight once, before the inner inner
            # loop.
            wij = wt(i, j)
            weighted_triangles += sum((wij * wt(j, k) * wt(k, i)) ** (1 / 3) for k in inbrs & jnbrs)

        yield (i, len(nbrs), (1/2) * weighted_triangles)


def clustering(adjacency, normalize=True):
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Adjacency matrix must be pd.Dataframe or np.ndarray."

    if isinstance(adjacency, pd.DataFrame):
        idx = list(adjacency.index)                 # Saves indices of Dataframe
    else:
        idx = list(range(adjacency.shape[0]))       # Sets indices otherwise

    adjacency = np.array(adjacency)                 # Copies to np.array

    triangl_iter = weighted_traingles_iter(adjacency, normalize=normalize)
    clusterc = {v: 0 if d < 2 else (2*t) / (d * (d - 1)) for v, d, t in triangl_iter}               # Output dictionary with clustercoefficients for each node
    clusterc = pd.Series(clusterc, index = idx)                                                     # Converting dictionary to pd.Series

    return clusterc



def rewire(adjacency, niter=10, seed=None):
    """
    Takes in adjacency matrix, translates it into a networkX network, generates a random reference network
    :param adjacency: nxn dimensional adjacency matrix
    :param niter: number of rewiring iterations
    :param seed: seed for random number generation
    :return: pd.DataFrame of rewired adjacency matrix
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    if isinstance(adjacency, pd.DataFrame):
        nodes = list(adjacency.index)
    else:
        nodes = np.arange(adjacency.shape[0])
    graph = nx.from_numpy_matrix(np.asarray(adjacency))                 # Convert to networkX graph
    random_nx = nx.random_reference(graph, niter=niter, seed=seed)      # Generate random reference using networkX
    random_adj = nx.to_numpy_matrix(random_nx)                             # Convert to numpy
    random_adj = pd.DataFrame(random_adj, columns=nodes, index=nodes)   # Converts to pd.DataFrame

    return random_adj

def assortativity(adjacency):
    """
    Computes assortativity coefficient of adjacency matrix using networkx immplementation.
    :param adjacency: array like adjacency matrix of network
    :return: float assortativity coeff
    """
    adj = np.array(adjacency)
    G = nx.from_numpy_matrix(adj)
    assortativity = nx.degree_pearson_correlation_coefficient(G, weight='weight')

    return assortativity