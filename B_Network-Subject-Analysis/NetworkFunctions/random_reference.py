import pandas as pd
import numpy as np
import network as net
import corr_functions as func
import scipy.stats as stats

def weighted_random(adjacency, niter):
    import random

    adj = adjacency.copy()
    num_nodes = adj.shape[0]

    for iter in range(niter):
        # Sign switching
        PositiveEdges = [(i,j) for i in range(num_nodes-1) for j in range(i+1, num_nodes) if adj[i,j] > 0]
        NegativeEdges = [(i,j) for i in range(num_nodes-1) for j in range(i+1, num_nodes) if adj[i,j] < 0]

        while len(NegativeEdges) > 0:
            a,b = random.choice(NegativeEdges)
            NegativeEdges.remove((a, b))
            SecondEdges = [(c,d) for (c, d) in NegativeEdges if ((a, d) in PositiveEdges or (d, a) in PositiveEdges)
                            and ((b, c) in PositiveEdges or (c,b) in PositiveEdges)]
            if len(SecondEdges) > 0:
                c,d = random.choice(SecondEdges)
                NegativeEdges.remove((c,d))
                if (a,d) in PositiveEdges: PositiveEdges.remove((a,d))
                else: PositiveEdges.remove((d,a))
                if (b,c) in PositiveEdges: PositiveEdges.remove((b,c))
                else: PositiveEdges.remove((c,b))

                adj[a,b] *= -1; adj[a,d] *= -1
                adj[b,a] *= -1; adj[b,c] *= -1
                adj[c,b] *= -1; adj[c,d] *= -1
                adj[d,c] *= -1; adj[d,a] *= -1

    num_neg = np.sum(adj<0)/2
    sign_switches = np.sum(adj != adjacency)/4
    print(int(num_neg), ' negative edges. ', int(sign_switches), ' sign switches were performed. ')

    # Weight randomization
    # Positive connections / negative connections

    pos_adj = np.array(adj, copy=True)
    pos_adj[pos_adj <= 0] = 0
    neg_adj = np.array(adj, copy=True)
    neg_adj[neg_adj >= 0] = 0

    adj_list = [pos_adj, neg_adj]
    random_adj_list = []

    for signed_adj in adj_list:

        strengths = np.sum(signed_adj, axis=-1)  # Computes the strength of each node
        list_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if
                      signed_adj[i, j] != 0 and i != j]
        num_edges = len(list_edges)  # Counts number of edges that are not zero
        list_weights = [signed_adj[i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes) if
                        signed_adj[i, j] != 0 and i != j]
        sorted_weights = sorted(list_weights)[::-1]  # Sort from highest to lowest value
        random_adj = np.zeros(signed_adj.shape)
        for edge in range(num_edges):
            r_strengths = np.sum(random_adj, axis=-1)
            e = {(i, j): (strengths[i] - r_strengths[i]) * (strengths[j] - r_strengths[j]) for i, j in list_edges}
            sorted_e = [i[0] for i in sorted(e.items(), key=lambda item: item[1])[::-1]]  # Sort from highest to lowest

            selected_edge = random.choice(list_edges)  # list_edges
            i, j = selected_edge
            rank = sorted_e.index((i, j))

            random_adj[i, j] = sorted_weights[rank]
            random_adj[j, i] = sorted_weights[rank]

            list_edges.remove((i, j))
            del sorted_weights[rank]

        random_adj_list.append(random_adj)

    pos_random = random_adj_list[0]
    neg_random = random_adj_list[1]
    random_adj = pos_random + neg_random
    adj = random_adj

    return adj

def hqs(DataMat):
    """
    1. Compute Covmatrix.
    2. Compute mean and variance of its off-diagonal elements
    3. Compute mean of diagonal elements
    4. Use algo from zalesky2012

    :param tc:
    :return:
    """
    CovMat = func.covariance_mat(DataMat)
    LowerTri = np.tril_indices(CovMat.shape[0], k=1)

    Covariances = CovMat[LowerTri]
    Variances = np.diag(CovMat)

    e_cov = np.mean(Covariances)
    v_cov = np.var(Covariances)

    e_var = np.mean(Variances)
    temp = np.floor((e_var ** 2 - e_cov ** 2) / v_cov)

    m = np.max([2, temp])
    mu = np.sqrt(e_cov/m)
    sigma = -mu**2 + np.sqrt(mu**4 + v_cov/m)
    std = np.sqrt(sigma)
    X = np.random.normal(loc=mu, scale=std, size=DataMat.shape)
    C = np.matmul(X, X.T)

    # Conversion to Correlation Matrix
    a = 1 / np.sqrt(np.diag(C))
    a = np.diag(a)
    temp = np.matmul(C,a)
    CorrMat = np.matmul(a,temp)

    return CorrMat


def brute_force():
    pass

def rewired_net(adjacency, niter=10, seed=None):
    """
    Generates random reference network using the Maslov-Sneppen Algorithm.
    :param adjacency: n x n dimensional pd.DataFrame
    :param niter: int specifying number of iterations to randomize the network
    :param seed: int seed of randomization to make network reproducible
    :return: network.network random network that can be used as null model
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    assert adjacency.shape[0] == adjacency.shape[1], "Adjacency matrix must be square."
    if seed: np.random.seed(seed)                   # Set seed to make random network replicable

    num_nodes = adjacency.shape[0]        # Specify number of nodes
    if isinstance(adjacency, pd.DataFrame):
        node_list=list(adjacency.index)
        adjacency = np.asarray(adjacency)
    else:
        node_list = np.arange(num_nodes)  # Specify list of nodes for index

    adj_mat = np.array(adjacency, copy=True)        # Create two copies of the adjacency matrix
    random_adj = np.array(adj_mat, copy=True)       # Random network is rewired in the following
    for r in range(niter):                          # Number rewiring iterations
        edge_list = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if j != i and not np.isclose(random_adj[i,j], 0)] # Create list of all edges
        num_edges = len(edge_list)/2  # number of edges
        rewired = []  # create list of rewired edges
        e = 0   # counter for rewired edges
        s = 0   # counter that stops rewiring if there are no matching edge pairs are found during the last 30 iterations

        while e<num_edges and s<30:
            first_idx, second_idx = np.random.choice(len(edge_list), 2, replace=False) # Randomly choose two different edges in network
            i, j = edge_list[first_idx]                                     # Node indices in adjacency matrix of first edge
            n, m = edge_list[second_idx]                                    # Node indices in adjacency matrix of second edge

            if n in [i,j] or m in [i,j]:                        # All nodes should be different
                s += 1
                continue
            if (i,n) in rewired or (j,m) in rewired:            # Check if the new edge was already rewired
                s += 1
                continue
            s = 0

            random_adj[i,n]=adj_mat[i,j]             # Rewire i to n
            random_adj[n,i]=adj_mat[i,j]             # Mirror edge to yield complete adjacency mat
            random_adj[j,m]=adj_mat[n,m]             # Rewire j to m
            random_adj[m,j]=adj_mat[n,m]             # Mirror edge

            edge_list=[edge for edge in edge_list if edge != (i,j) and edge != (j,i) and edge != (n,m) and edge != (m,n)]
            rewired.extend([(i,n),(j,m),(n,i),(m,j)])
            e += 2
        adj_mat = np.array(random_adj, copy=True)

    random_adj=pd.DataFrame(random_adj, index=node_list, columns=node_list) # Convert to DataFrame
    return random_adj

def rewire_nx(adjacency, niter=10, seed=None):
    """
    Takes in adjacency matrix, translates it into a networkX network, generates a random reference network
    :param adjacency: nxn dimensional adjacency matrix
    :param niter: number of rewiring iterations
    :param seed: seed for random number generation
    :return: pd.DataFrame of rewired adjacency matrix
    """
    import networkx as nx

    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    if isinstance(adjacency, pd.DataFrame):
        nodes = list(adjacency.index)
    else:
        nodes = np.arange(adjacency.shape[0])

    graph = nx.from_numpy_matrix(np.asarray(adjacency))                 # Convert to networkX graph
    random_nx = nx.random_reference(graph, niter=niter, seed=seed)      # Generate random reference using networkX
    random_adj = random_nx.to_numpy_matrix                              # Convert to numpy
    random_adj = pd.DataFrame(random_adj, columns=nodes, index=nodes)   # Converts to pd.DataFrame

    return random_adj



