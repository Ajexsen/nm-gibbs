from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


def sim_vector(x, y):
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return 0
    x = x.reshape((1, len(x)))
    y = y.reshape((1, len(y)))
    cos_sim = cosine_similarity(x, y)
    lenX = np.linalg.norm(x)
    lenY = np.linalg.norm(y)
    len_sim = lenX / lenY if lenX < lenY else lenY / lenX
    sim = (cos_sim + len_sim) / 2
    return sim


# def compute_B_similarity(X, Z, A):
#     N = X.shape[0]
#     K = A.shape[0]
#     B_sim = np.zeros((N, K))
#     for j in range(K):
#         if Z[j] == 1:
#             for i in range(N):
#                 B_sim[i][j] = sim_vector(X[i], A[j])
#     return B_sim


def compute_B_similarity(X, Z, A):
    N = X.shape[0]
    K = A.shape[0]
    B_sim = np.zeros((N, K))
    B_sum = 0
    for j in range(K):
        if Z[j] == 1:
            for i in range(N):
                B_sim[i][j] = 1 / np.linalg.norm(X[i] - A[j])
                # B_sum += B_sim[i][j]
    # B_sim = B_sim / B_sum
    return B_sim


def compute_similarity_fromB(B, X, A):
    N, K = B.shape
    B_sim = np.zeros((N, K))
    for j in range(K):
        for i in range(N):
            if B[i][j] == 1:
                dist = np.linalg.norm(X[i] - A[j])
                if dist < 1:
                    dist = 1
                B_sim[i][j] = 1 / dist
            else:
                B_sim[i][j] = 0
    return B_sim


def compute_distance_fromB(B, X, A):
    N, K = B.shape
    B_dist = np.zeros((N, K))
    for j in range(K):
        for i in range(N):
            if B[i][j] == 1:
                B_dist[i][j] = np.linalg.norm(X[i] - A[j])
            else:
                B_dist[i][j] = -1
    return B_dist


def compute_distance_Z(X, Z, A):
    N = X.shape[0]
    K = A.shape[0]
    B_sim = np.zeros((N, K))
    B_sum = 0
    for j in range(K):
        if Z[j] == 1:
            for i in range(N):
                B_sim[i][j] = np.linalg.norm(X[i] - A[j])
                # B_sum += B_sim[i][j]
    # B_sim = B_sim / B_sum
    return B_sim


def Z2B(X, Z, A):
    N = X.shape[0]
    K = A.shape[0]  # K features (global)
    B_sim = compute_B_similarity(X, Z, A)
    B = np.zeros((N, K))
    t = np.count_nonzero(Z)
    for i in range(t):
        if np.argmax(B_sim) == 0:
            break
        (l, g) = np.unravel_index(np.argmax(B_sim), B_sim.shape)
        # B_sim[l, g] = 0
        B_sim[:, g] = 0
        B_sim[l, :] = 0
        B[l][g] = 1
    return B


def Z2B_hungarian(X, Z, A):
    N = X.shape[0]
    K = A.shape[0]  # K features (global)
    match = Z.nonzero()[0] # index of global feature that have a match, [0]: tuple -> narray
    B_sim = compute_distance_Z(X, Z, A)[:, match]
    B = np.zeros((N, K))
    row_ind, col_ind = linear_sum_assignment(B_sim)
    B[row_ind, match[col_ind]] = 1
    return B


# TODO: only make fitting size, no Inf (int.max)
# TODO: make dict -> (array[0] -> global_3, array[1] -> global_10, array[2] -> global_17)
def Z2B_bi(X, Z, A):
    N = X.shape[0]
    K = A.shape[0]  # K features (global)
    B = np.zeros((N, K))
    B_dist = compute_distance_Z(X, Z, A)

    b_graph = nx.Graph()
    for g in range(K):
        if Z[g] == 0:
            B_dist[:, g] = 9223372036854775807  # max Int
        for l in range(N):
            b_graph.add_weighted_edges_from([(l, N + g, B_dist[l][g])])  # 1-sim -> min matching
    M = nx.bipartite.minimum_weight_full_matching(b_graph)

    for g in range(K):
        if Z[g] == 1:
            B[M[N + g], g] = 1
    return B


def Z2B_hun_all(X, Z, A):
    N = len(X)  # count of objects (cars)
    K = A.shape[0]  # Nr. features
    B = []
    for car in range(N):
        Bi = Z2B_hungarian(X[car], Z[car], A)
        B.append(Bi)
    return B


def Z2B_all(X, Z, A):
    N = len(X)  # count of objects (cars)
    K = A.shape[0]  # Nr. features
    B = []
    for car in range(N):
        Bi = Z2B(X[car], Z[car], A)
        B.append(Bi)
    return B


def Z2B_rand(Z, n):
    N = n  # X size
    K = Z.shape[0]  # 1 if full Z, 0 if loc Z
    B = np.zeros((N, K))
    for i in range(K):
        if Z[i] == 1:
            rand = np.random.randint(0, N)
            B[rand, i] = 1
    return B


# return a list of all local features grouped by matching global feature
def groupX(X, B, bias=None):
    assert len(X) == len(B)
    C = len(X)
    K = B[0].shape[1]  # features
    W = X[0].shape[1]  # weights
    grouped_weight = [np.empty((0, W)) for i in range(K)]
    grouped_bias = [np.empty((0, 1)) for i in range(K)]
    for car in range(C):
        xi = X[car]
        Bi = B[car]
        for feature_loc in range(xi.shape[0]):
            match = [i for i in range(K) if Bi[feature_loc][i] > 0]
            for feature_glo in match:
                grouped_weight[feature_glo] = np.vstack((grouped_weight[feature_glo], xi[feature_loc]))
    return grouped_weight


def B_to_assignment(B):
    J = len(B)
    assignment = [[] for _ in range(J)]
    for i in range(J):
        bi = B[i]
        n = bi.shape[0]
        assignment_i = []
        for j in range(n):
            if np.count_nonzero(bi[j]) != 0:
                match_ind = np.nonzero(bi[j])[0][0]
            else:
                match_ind = None  # no match
            assignment_i.append(match_ind)
        assignment[i] = assignment_i
    return assignment


def comp(x):
    x = np.vstack(x)
    size_x = len(x)
    comp = [np.linalg.norm(x[i] - x[j]) for i in range(size_x) for j in range(size_x)]
    maxi = max(comp)
    mini = min(comp)
    avg = np.mean(comp)
    return maxi, mini, avg


def group_sigma(X, B):
    grouped = groupX(X, B)
    K = B[0].shape[1]  # global features
    W = X[0].shape[1]  # weights
    g_sigma = np.array([np.var(grouped[i], axis=0) for i in range(K)]) # np.zeros((K, W))
    return g_sigma


def compute_distance(X, A):
    N = X.shape[0]
    K = A.shape[0]
    B_sim = np.zeros((N, K))
    B_sum = 0
    for j in range(K):
        for i in range(N):
            B_sim[i][j] = np.linalg.norm(X[i] - A[j])
            # B_sum += B_sim[i][j]
    # B_sim = B_sim / B_sum
    return B_sim


def A2B(X, A):
    N = X.shape[0]
    K = A.shape[0]  # K features (global)
    B_dist = compute_distance(X, A)
    B = np.zeros((N, K))
    t = min(N, K)
    maxval = np.max(B_dist)
    for i in range(t):
        (l, g) = np.unravel_index(np.argmin(B_dist), B_dist.shape)
        B_dist[:, g] = maxval
        B_dist[l, :] = maxval
        B[l][g] = 1
    return B


def A2B_hun(X, A):
    N = X.shape[0]
    K = A.shape[0]  # K features (global)
    B_dist = compute_distance(X, A)
    B = np.zeros((N, K))
    row_ind, col_ind = linear_sum_assignment(B_dist)
    B[row_ind, col_ind] = 1
    return B


def A2B_all(X, A):
    N = len(X)  # count of objects (cars)
    B = []
    for car in range(N):
        Bi = A2B(X[car], A)
        B.append(Bi)
    return B


def A2B_hun_all(X, A):
    N = len(X)  # count of objects (cars)
    B = []
    for car in range(N):
        Bi = A2B_hun(X[car], A)
        B.append(Bi)
    return B