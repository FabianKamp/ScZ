import pandas as pd
import numpy as np
import NetworkFunctions.network as net
import NetworkFunctions.corr_functions as func
import random

def hqs(tc):
    """
    Returns a random network that is matched to the input networks covariance matrix.
    Using Hirschberger-Qi-Steuer Algorithm as cited in Zalesky 2012b

    :return: n x n dimensional pd.Dataframe
    TODO debug code, control if input has to be is positive finite
    """
    if not isinstance(tc, (np.ndarray, pd.DataFrame)):
        raise ValueError('Timecourse has to be np.ndarray or pd.DataFrame')
    tc = np.array(tc)
    #C = NetworkFunctions.covariance_mat(tc)
    tc = tc - np.mean(tc, axis=-1, keepdims=True)
    C = np.matmul(tc, tc.T)

    diag_sum = np.sum(np.diagonal(C))
    diag_len = len(np.diagonal(C))
    diag_mean = diag_sum / diag_len

    off_mean = (np.sum(C) - diag_sum) / (C.size - diag_len)
    off_var = C - off_mean
    np.fill_diagonal(off_var,0)                     # Sets diagonal values to zero
    off_var = np.sum(off_var ** 2)/(C.size-diag_len)

    m = max(2, np.round(diag_mean**2 - off_mean**2) / off_var)
    mu = np.sqrt(off_mean / m)
    sigma = -(mu**2) + np.sqrt(mu ** 4 + (off_var / m))
    std = np.sqrt(sigma)
    X = np.random.normal(mu, std, tc.shape)
    X = X-np.mean(X, axis=-1, keepdims=True)
    random_covmat = np.matmul(X, X.T)

    covmat_diag = np.diagonal(random_covmat)
    inv_covmat_diag = np.diag(1/np.sqrt(covmat_diag))        # Invert the diagonal array and convert into matrix
    random_adjacency_mat = np.matmul((np.matmul(inv_covmat_diag, random_covmat)), inv_covmat_diag)    # Convert from covariance matrix into correlation matrix

    random_net=net.network(random_adjacency_mat)

    return random_net

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

def weighted_random(adjacency, niter):
    """
    Computes random reference network for weighted networks. Following the algorithm by Rubinov and Sporns 2011
    "Weight-conserving characterization of complex functional brain networks".
    :param adjacency: np.ndarray or pd.DataFrame
    :return: randomized adjacency matrix as pd.DataFrame
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    import random

    adj = np.array(adjacency, copy=True)
    num_nodes = adj.shape[0]

    for iter in range(niter):

        # Sign switching
        positive_edges = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if adj[i,j] > 0 and i != j]
        negative_edges = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if adj[i,j] < 0 and i != j]

        if len(positive_edges) > len(negative_edges):
            more_edges = positive_edges.copy()
            less_edges = negative_edges.copy()
        else:
            more_edges = negative_edges.copy()
            less_edges = positive_edges.copy()

        n_edges = len(less_edges)/2
        for i in range(int(n_edges/2)):

            first_edge = random.choice(less_edges)
            a, b = first_edge

            second_list = [(c,d) for c,d in less_edges if (a,c) in more_edges and (b,d) in more_edges and c not in (a,b)
                           and d not in (a,b)]

            if len(second_list) == 0:
                continue

            second_edge = random.choice(second_list)
            c, d = second_edge

            adj[a,[b,c]] *= -1
            adj[b,[a,d]] *= -1
            adj[c,[a,d]] *= -1
            adj[d,[b,c]] *= -1

            less_edges.remove((a, b))
            less_edges.remove((b, a))
            less_edges.remove((c, d))
            less_edges.remove((d, c))

            more_edges.remove((a,c))
            more_edges.remove((c,a))
            more_edges.remove((b,d))
            more_edges.remove((d,b))

            n_edges = len(less_edges)/2

        if n_edges != 0:
            print('Could not invert the sign of ', len(less_edges), ' elements.')

        # Weight randomization
        # Positive connections / negative connections

        pos_adj = np.array(adj, copy=True)
        pos_adj[pos_adj <= 0] = 0
        neg_adj = np.array(adj, copy=True)
        neg_adj[neg_adj >= 0] = 0

        adj_list = [pos_adj, neg_adj]
        random_adj_list = []

        for signed_adj in adj_list:

            strengths = np.sum(signed_adj, axis=-1) # Computes the strength of each node

            list_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if
                         signed_adj[i, j] != 0 and i != j]
            num_edges = len(list_edges)    # Counts number of edges that are not zero

            list_weights = [signed_adj[i,j] for i in range(num_nodes) for j in range(i+1, num_nodes) if signed_adj[i,j] != 0 and i != j]
            sorted_weights = sorted(list_weights)[::-1] # Sort from highest to lowest value

            random_adj = np.zeros(signed_adj.shape)
            for edge in range(num_edges):

                r_strengths = np.sum(random_adj, axis=-1)
                e = {(i,j): (strengths[i] - r_strengths[i])*(strengths[j] - r_strengths[j]) for i,j in list_edges}
                sorted_e = [i[0] for i in sorted(e.items(), key=lambda item: item[1])[::-1]] # Sort from highest to lowest

                selected_edge = random.choice(list_edges) #list_edges
                i,j = selected_edge
                rank = sorted_e.index((i, j))

                random_adj[i,j] = sorted_weights[rank]
                random_adj[j,i] = sorted_weights[rank]

                list_edges.remove((i,j))
                del sorted_weights[rank]

            random_adj_list.append(random_adj)

        pos_random = random_adj_list[0]
        neg_random = random_adj_list[1]
        random_adj = pos_random + neg_random
        adj = random_adj
        print(iter+1, ' iteration of randomization.')

    return adj

