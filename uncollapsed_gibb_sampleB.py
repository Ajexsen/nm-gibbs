import numpy as np
import logging
import random

import scipy
import scipy.stats

from B_function import *
from B_matching import *
from monte_carlo import MonteCarlo

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')


# Z2B = Z2B_hungarian
# Z2B_all = Z2B_hun_all


def exp_computable(log_like):
    return -744 <= log_like <= 709


def likelihood_ratio(log_0, log_1):
    log_diff = log_0 - log_1
    if exp_computable(log_0) and exp_computable(log_0):
        pz0 = np.exp(log_0)
        pz1 = np.exp(log_1)
        Znk_is_0 = pz0 / (pz0 + pz1)
    elif exp_computable(log_diff):
        pz0 = np.exp(log_diff)
        pz1 = np.exp(0)
        Znk_is_0 = pz0 / (pz0 + pz1)
    else:
        if log_0 > log_1:
            Znk_is_0 = 1
        elif log_0 < log_1:
            Znk_is_0 = 0
        else:
            Znk_is_0 = 0.5  # this happends already in logdiff
    return Znk_is_0


class UncollapsedGibbs(MonteCarlo):
    """
    sample the corpus to train the parameters
    """

    def _initialize(self, data, alpha=1.0, beta=1.0, sigma_a=1.0, sigma_x=1.0, initial_A=None, A_prior=None, draw="w"):

        if draw is "w":
            self.draw_new_feature = self.local_worst_match
        elif draw is "r":
            self.draw_new_feature = self.random_local
        else:
            print("draw method not found: {}".format(draw))
            self.draw_new_feature = self.local_worst_match


        # initial_B=None
        # no initial_Z=None
        self._counter = 0;

        self._alpha = alpha
        self._beta = beta
        # self._sigma_x = sigma_x
        self._sigma_a = sigma_a

        # Data matrix
        self._X = data
        self._N = len(self._X)
        self._W = self._X[0].shape[1]

        sigma_x_start = sigma_x
        self._sigma_xj = [sigma_x_start for i in range(self._N)]
        # for j in range(len(self._X)):
        #     self._sigma_xj.append(np.std(self._X[j]))

        self._N = len(self._X)
        self._W = self._X[0].shape[1]

        # use average size as init size of global
        all_size = [self._X[n].shape[0] for n in range(self._N)]
        avg_size = np.mean(all_size)
        self._K = int(avg_size)
        self._K = 1
        # self._K = 1 # start with one

        self._A = np.zeros((self._K, self._W))
        # if initial_A is None:
        #     # initialize A by random sampling from X
        #     self._A = self.initialize_A()
        # else:
        #     self._A = initial_A

        # self._B = A2B_all(self._X, self._A)
        self._B = [np.zeros((self._X[i].shape[0], self._A.shape[0])) for i in range(self._N)]
        self._B_star = [np.zeros(self._B[i].shape) for i in range(self._N)]

        # self.update_sigma_x()
        # print("sigma_xj: ", self._sigma_xj)

        # if initial_B is None:
        #     # initialize Z from IBP(alpha)
        #     self._B = A2B_all(self._X, self._A)
        # else:
        #     self._B = initial_B

        if A_prior is None:
            self._A_prior = np.zeros((1, self._W))
        else:
            self._A_prior = A_prior
        assert (self._A_prior.shape == (1, self._W))

        # assert (self._A.shape == (self._K, self._D))
        # self.C = CAPACITY
        # self.STATE_CATACLYSM_INIT = np.zeros(shape=(self._N, self._K))
        # self.PA = np.ones(shape=(self._N, self._K)) * (self._N * self.C)  # Experimental
        # self.PB = np.ones(shape=(self._N, self._K)) * (self._N * self.C)  # Experimental
        # self.G = GAMMA
        # logging.debug("N: {}, K: {}, C: {}".format(self._N, self._K, self.C))

    @property
    def learning(self):
        self._counter += 1

        # assert (self._A.shape == (self._K, self._D))
        # assert (self._X.shape == (self._N, self._D))

        # sample every object (car)
        order = np.random.permutation(self._N)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            import time
            start_sample = time.time()
            singleton_features = self.sample_B_star(object_index)
            finish_sample = time.time()
            print("Car-", object_index, " : ", finish_sample - start_sample, " sec")
            if self._metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                self.metropolis_hastings_K_new(object_index, singleton_features)

        # regularize matrices -> iterate through all (check)
        # self.B_star_allOne()
        # Btemp = self._B_star
        # log_like = self.log_likelihood_model()
        # print("log_likelihood (Before B matching):", log_like, ", K =", self._K)
        self._B = self.B_matching()
        self.regularize_matrices()
        self.sample_A()

        # TODO:
        if self._alpha_hyper_parameter is not None:
            self._alpha = self.sample_alpha()
        #
        if self._sigma_x_hyper_parameter is not None:
            # self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter)
                self.update_sigma_x()
            # self._sigma_x = self.sample_sigma(self._sigma_x_hyper_parameter, self._X - np.dot(self._B, self._A))
            # self._sigma_x = self.sample_sigma_x()
        # if self._sigma_a_hyper_parameter is not None:
            # self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter)
            # self._sigma_a = self.sample_sigma(self._sigma_a_hyper_parameter,
            #                                   self._A - np.tile(self._A_prior, (self._K, 1)))
        log_like = self.log_likelihood_model()
        return log_like

    def update_sigma_x(self):
        for j in range(len(self._X)):
            self._sigma_xj[j] = self.sample_sigma(self._sigma_x_hyper_parameter,
                                                  self._X[j] - np.dot(self._B[j], self._A))

    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def initialize_A(self, X=None, Z=None):
        if X is None:
            X = self._X
        K = self._K  # random sample K from X
        W = self._W
        A = np.zeros((K, W))
        X = np.vstack(X)
        np.random.shuffle(X)
        for i in range(K):
            A[i, :] = X[i, :]
        return A

    # TODO: sample B from object
    def sample_B_star(self, object_index):
        X = self._X[object_index]
        loc_feature = X.shape[0]
        B = self._B
        Bi = self._B[object_index].sum(axis=0)

        m = sum(B).sum(axis=0)
        new_m = (m - Bi).astype(np.float)

        # TODO: new_m too big -> B multiple 1 in one global
        # need to normalize every round?
        log_prob_z1 = np.log(new_m / self._N)
        log_prob_z0 = np.log(self._N - new_m / self._N)
        # log_prob_z0 = np.log(1.0 - new_m / self._N)

        singleton_features = [nk for nk in range(self._K) if Bi[nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]

        order = np.random.permutation(self._K)

        # for local and for global
        import time
        for (feature_counter, glo_feature_index) in enumerate(order):
            if glo_feature_index in non_singleton_features:
                for loc_feature_index in range(loc_feature):
                    self._B_star[object_index][loc_feature_index, glo_feature_index] = 0
                    prob_z0 = self.log_likelihood_B_star(object_index, glo_feature_index)
                    prob_z0 += log_prob_z0[glo_feature_index]
                    # prob_z0 = np.exp(prob_z0)

                    # compute the log likelihood when Znk=1
                    self._B_star[object_index][loc_feature_index, glo_feature_index] = 1
                    prob_z1 = self.log_likelihood_B_star(object_index, glo_feature_index)
                    prob_z1 += log_prob_z1[glo_feature_index]
                    # prob_z1 = np.exp(prob_z1)

                    Znk_is_0 = likelihood_ratio(prob_z0, prob_z1)

                    # logging.debug("object: {}, Feature: {}, log_like_0: {}, log_like_1: {}".format(object_index, feature_index, p0, p1))

                    # TODO: replace Z if too many features
                    r = random.random()
                    if r < Znk_is_0:
                        self._B_star[object_index][loc_feature_index, glo_feature_index] = 0
                    else:
                        self._B_star[object_index][loc_feature_index, glo_feature_index] = 1

        return singleton_features

    """
    sample K_new using metropolis hastings algorithm
    """

    def metropolis_hastings_K_new(self, object_index, singleton_features):
        # if type(object_index) != list:
        #     object_index = [object_index]

        # sample K_new from the metropolis hastings3 proposal distribution,
        # i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N)

        max_new = self._X[object_index].shape[0]
        if K_temp > max_new:
            K_temp = max_new
        # if K_temp <= 0 and len(singleton_features) <= 0:
        if K_temp <= 0 and len(singleton_features) <= 0:
            print("No K_new or singletons.")
            return False

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-D matrix

        non_singleton = [k for k in list(range(self._K)) if k not in singleton_features]
        # TODO: new feature from local
        A_prior, feature_index = self.draw_new_feature(object_index, K_temp)
        A_temp = A_prior #+ np.random.normal(0, self._sigma_a, (K_temp, self._W))
        A_new = np.delete(self._A, singleton_features, axis=0)
        A_new = np.vstack((A_new, A_temp))

        n_loc = self._X[object_index].shape[0] #+ K_temp #len(non_singleton) + K_temp
        n_glo = self._K + K_temp
        B_add = np.zeros((n_loc, K_temp))
        for i in range(K_temp):
            B_add[feature_index[i], i] = 1

        # construct the A_old and Z_old
        A_old = self._A
        B_old = self._B[object_index]
        K_old = self._K

        # assert (A_old.shape == (K_old, self._D))
        # assert (A_new.shape == (K_new, self._D))
        # assert (Z_old.shape == (len(object_index), K_old))
        # assert (Z_new.shape == (len(object_index), K_new))

        # compute the probability of using old features
        prob_old = self.log_likelihood_B(object_index, B_old, A_old)
        # prob_old = np.exp(prob_old)

        # TODO: if add new feature match to which one? -> the one pick from local
        # feature_index
        B_new = B_old
        B_new[feature_index, :] = 0
        B_new = np.delete(B_new, singleton_features, axis=1)
        B_new = np.hstack((B_new, B_add))
        # B_new = np.hstack((self._B[object_index], B_add))

        B_star_new = np.delete(self._B_star[object_index], singleton_features, axis=1)
        B_star_new = np.hstack((B_star_new, B_add))
        # Z_new = np.hstack((self._Z[[object_index], non_singleton], None))
        #                    np.ones(K_temp)))

        K_new = self._K + K_temp - len(singleton_features)

        # compute the probability of generating new features
        prob_new = self.log_likelihood_B(object_index, B_new, A_new)
        # prob_new = np.exp(prob_new)

        # compute the probability of generating new features
        prob = likelihood_ratio(prob_new, prob_old)  # prob_new / (prob_old + prob_new)

        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < prob:
        # if True or random.random() < prob:
            # construct A_new and Z_new
            B_star = self._B_star
            B = self._B
            for i in range(len(B)):
                if i == object_index:
                    B[i] = B_new
                    B_star[i] = B_star_new
                n_loc = B[i].shape[0]
                B_add_zero = np.zeros((n_loc, K_temp))

                B[i] = np.delete(B[i], singleton_features, axis=1)
                B[i] = np.hstack((B[i], B_add_zero))

                B_star[i] = np.delete(B_star[i], singleton_features, axis=1)
                B_star[i] = np.hstack((B_star[i], B_add_zero))

            self._B = B
            self._B_star = B_star

            self._B[object_index] = B_new
            self._B_star[object_index] = B_star_new

            self._K = K_new
            self._A = A_new
            print("K:", K_new, ", single_remove:", len(singleton_features), "(", prob, ")")
            return True
        print("K_new rejected.", "(", prob, ")")
        return False

    """
    """

    def sample_A(self):
        # sample every feature
        D = len(self._X)
        (K, W) = self._A.shape
        X = self._X
        B = self._B
        X_group = groupX(X, B)

        for feature_index in range(K):
            if len(X_group[feature_index]) > 1:
                mean_i = np.array([np.mean(X_group[feature_index], axis=0)])
                var_i = np.array([np.var(X_group[feature_index], axis=0)])
                rand = 0 #np.random.normal(0, 1, (1, W))
                self._A[feature_index] = mean_i + var_i * rand

    """
    sample noise variances, i.e., sigma_x
    """

    def sample_sigma_x(self):
        # X - ZA => mean(X - BA)
        N = self._N
        K = self._K
        W = self._W

        X = self._X
        B = self._B
        A = self._A

        # sum matched X / Z (occurrence count)
        diff = np.zeros((K, W))
        for i in range(N):
            xi = X[i]
            bi = B[i]
            # TODO: replace with group X
            bi_size = np.count_nonzero(bi)
            (l, g) = np.nonzero(bi)
            for k in range(bi_size):
                loc = l[k]
                glo = g[k]
                diff[glo] += xi[loc] - A[glo]

        # TODO: calc count of each feature
        B_sum = 0  # Z.sum(axis=0).reshape((K, 1))  # occurrence count
        diff = diff / 0  # mean

        return self.sample_sigma(self._sigma_x_hyper_parameter,
                                 diff)

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """

    def regularize_matrices(self):
        X = self._X
        A = self._A
        B = self._B
        n = len(B)
        f = A.shape[0]  # global feature count
        # for i in range(n):
        #     Bi = B[i]
        #     Xi = X[i]
        #     # Bi_sim = -compute_similarity_fromB(Bi, Xi, A)
        #     # row_ind, col_ind = linear_sum_assignment(Bi_sim)
        #     # B_new[row_ind, col_ind] = 1
        #     Bi_dist = -compute_distance_fromB(Bi, Xi, A)
        #     B_new = B_min_dist(Bi_dist)
        #     B[i] = B_new

        glob_count = sum([B[i].sum(axis=0) for i in range(n)])
        used_feature = glob_count.nonzero()[0]
        not_used = [glob for glob in range(f) if glob not in used_feature]

        for i in range(n):
            B[i] = np.delete(B[i], not_used, axis=1)
            self._B_star[i] = np.delete(self._B_star[i], not_used, axis=1)
        A = np.delete(A, not_used, axis=0)
        print("removed global feature: ", not_used)

        self._B = B
        self._A = A
        self._K = A.shape[0]

        # assert (self._Z.shape == (self._N, self._K))
        # # assert (self._A.shape == (self._K, self._D))

    def B_matching(self):
        X = self._X
        A = self._A
        B = self._B_star
        N = len(X)
        B_sim = [None for i in range(N)]
        for i in range(N):
            Bi = B[i]
            Xi = X[i]
            B_sim[i] = compute_similarity_fromB(Bi, Xi, A)
        # Bibartite B matching
        local_nodes = []
        global_nodes = [i for i in range(A.shape[0])]
        local_cap = 1  # {1: 1, 2: 1, ...}
        global_cap = N  # {1: 5, 2: 5, ...}

        wts = {}
        for n in range(N):
            Bi_sim = B_sim[n]
            loc, glo = Bi_sim.shape
            for i in range(loc):
                loc_ind = str(n) + '.' + str(i)
                local_nodes.append(loc_ind)
                wts[loc_ind] = {}
                for j in range(glo):
                    if Bi_sim[i][j] == 0:
                        wts[loc_ind][j] = 0
                    else:
                        wts[loc_ind][j] = Bi_sim[i][j]
        p = solve_wbm(local_nodes, global_nodes, wts, local_cap, global_cap)
        # print_solution(p)

        selected_edges = get_selected_edges(p)
        B_new = [np.zeros(B[i].shape) for i in range(N)]
        B_match = get_B_from_selected_edge(selected_edges, B_new)

        # Create a Networkx graph. Use colors from the WBM solution above (selected_edges)
        # graph = nx.Graph()
        # colors = []
        # for u in local_nodes:
        #     for v in global_nodes:
        #         edge_color = 'g' if (str(u), str(v)) in selected_edges else 'r'
        #         if wts[u][v] > 0:
        #             graph.add_edge('u_' + str(u), 'v_' + str(v))
        #             colors.append(edge_color)
        #
        # pos = get_bipartite_positions(graph)
        #
        # nx.draw_networkx(graph, pos, with_labels=True, edge_color=colors,
        #                  font_size=20, alpha=0.5, width=1)
        #
        #
        # plt.axis('off')
        # plt.show()
        # print("done")

        # TODO: match too much
        # B_sim
        B_match_new = [None for i in range(N)]
        for i in range(N):
            if np.all(B_match[i].sum(axis=1) <= 1):
                B_match_new[i] = B_match[i]
                continue
            match = B_match[i].sum(axis=0).nonzero()[0]
            B_dist = B_sim[i]
            l, g = B_match[i].shape
            for loc in range(l):
                for glo in range(g):
                    if not B_match[i][loc][glo] == 1:
                        B_dist[loc][glo] = 0
            B_dist = B_sim[i][:, match]
            B_empty = np.zeros((l, g))
            row_ind, col_ind = linear_sum_assignment(-B_dist)
            B_empty[row_ind, match[col_ind]] = 1
            B_match_new[i] = B_empty

        return B_match_new

    def log_likelihood_B_star(self, object_index, glo_feature_index, B_star=None):
        if B_star is None:
            B_star = self._B_star[object_index]
        X = self._X[object_index]
        # sigma_x = self._sigma_xj[object_index]

        # sigma_x = np.var(X)
        sigma_0 = np.std(self._A)
        mu_0 = np.mean(self._A, axis=0)

        if sigma_0 == 0:
            sigma_0 = 1

        A = self._A
        n = X.shape[0]
        K = A.shape[0]
        D = A.shape[1]
        # sigma_x = 2 ** D
        sigma_x = self._sigma_xj[object_index]


        log_likelihood = 0
        log_likelihood = np.linalg.norm((mu_0/sigma_0) + np.sum([B_star[i][glo_feature_index] * (X[i]**2 / np.sqrt(sigma_x)) for i in range(n)])) ** 2
        log_likelihood = log_likelihood / (1/sigma_0**2 + np.sum([B_star[i][glo_feature_index] / sigma_x for i in range(n)]))
        log_likelihood = 0.5 * log_likelihood

        # fix_term = D * np.log(2 * np.pi * (sigma_layer ** 2))
        #
        # glo = glo_feature_index
        # theta_sq = np.dot(A[glo], A[glo].T)
        # B_error_sum = sum([
        #     B_star[loc][glo] * np.linalg.norm(X[loc] - A[glo]) ** 2 / sigma_x
        #     for loc in range(n)])
        # log_likelihood += theta_sq / (sigma_layer ** 2) + fix_term + B_error_sum
        #
        # log_likelihood = -0.5 * log_likelihood

        return log_likelihood

    def log_likelihood_B(self, object_index, B=None, A=None):
        X = self._X[object_index]
        if B is None:
            B = self._B[object_index]
        if A is None:
            A = self._A
        (N, D) = X.shape
        # D = 1

        K = A.shape[0]  # feature count

        sigma_x = self._sigma_xj[object_index]

        log_likelihood = X - np.dot(B, A)
        (row, column) = log_likelihood.shape

        if row > column:
            log_likelihood = np.trace(np.dot(log_likelihood.transpose(), log_likelihood))
        else:
            log_likelihood = np.trace(np.dot(log_likelihood, log_likelihood.transpose()))

        diff = log_likelihood
        log_likelihood = -0.5 * log_likelihood / np.power(sigma_x, 2)
        log_likelihood -= N * D * 0.5 * np.log(2 * np.pi * np.power(sigma_x, 2))

        # store loglike to observe the changes (testing purpose)
        self.log_like[self.log_c % self.log_n] = log_likelihood
        self.log_c += 1

        return log_likelihood

    """
    compute the log-likelihood of the model
    """

    def log_likelihood_model(self):
        # TODO: ALL

        # N: car count
        # D: global feature count
        # K: weight count
        # assert (len(X) == len(Z))
        X = self._X
        N = len(X)

        # average log_like from all models
        log_likelihood = np.sum([self.log_likelihood_B(i) for i in range(N)])

        return log_likelihood

    def result(self):
        X = self._X
        B = self._B
        A = self._A
        grouped = groupX(X, B)
        size = len(grouped)
        # add A[i] to each group
        for i in range(size):
            grouped[i] = np.vstack((A[i], grouped[i]))
        return grouped

    # return a un-matched feature by random
    def random_local(self, object_index, K_new):
        X = self._X[object_index]
        N, W = X.shape
        x_rand = np.zeros((K_new, W))
        feature_index = []

        rand = np.random.permutation(N)  # shuffle
        for i in range(K_new):
            if i >= N:
                break
            i_new = rand[i]
            x_rand[i, :] = X[i_new, :]
            feature_index.append(i_new)
        return x_rand, feature_index

    # return a un-matched feature by random
    def random_local_none(self, object_index, K_new):
        X = self._X[object_index]
        B = self._B[object_index]

        assert X.shape[0] == B.shape[0]

        match_sum = B.sum(axis=1)
        W = X.shape[1]
        n = len(match_sum)
        x_rand = np.zeros((K_new, W))
        feature_index = []

        non_matching = [i for i in range(n) if match_sum[i] == 0]
        non_matching = np.random.permutation(non_matching)  # shuffle
        for i in range(K_new):
            if i >= len(non_matching):
                break
            i_new = non_matching[i]
            x_rand[i, :] = X[i_new, :]
            feature_index.append(i_new)
        return x_rand, feature_index

    def local_worst_match(self, object_index, K_new):
        X = self._X[object_index]
        B = self._B[object_index]
        loc, glo = B.shape

        if K_new >= loc:
            K_new = loc

        B_diff = compute_distance(X, self._A)
        matched = B.nonzero()
        not_mat = [i for i in range(loc) if i not in matched[0]]
        diff = B_diff.mean(axis=1)

        for row, col in zip(matched[0], matched[1]):
            diff[row] = B_diff[row][col]

        sort_index = np.argsort(diff)[::-1]

        feature_index = sort_index[:K_new]
        x_rand = np.zeros((K_new, X.shape[1]))
        for i in range(K_new):
            x_rand[i, :] = X[feature_index[i], :]

        # TODO: sort by diff
        return x_rand, feature_index

    def get_A(self):
        return self._A

    def get_A_sigma(self):
        return group_sigma(self._X, self._B)

    def get_assignments(self):
        N = self._N
        B = self._B
        assignments = []
        for c in range(N):
            bi = B[c]
            assignment_j = []
            for i in range(bi.shape[0]):
                k = np.count_nonzero(bi[i])
                if k == 0:
                    index = -1
                elif k == 1:
                    index = np.nonzero(bi[i])[0][0]
                else:
                    index = -2
                    print("ERROR: more than one match. object: " + c + ", l_feature: " + i)
                assignment_j.append(index)
            assignments.append(assignment_j)
        return assignments


    def B_star_allOne(self):
        # for testing purpose (all ones in B_star)
        B_star = self._B_star
        N = len(B_star)
        B_star = [np.ones(B_star[i].shape) for i in range(N)]
        self._B_star = B_star