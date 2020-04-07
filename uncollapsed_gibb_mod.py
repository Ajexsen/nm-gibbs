import numpy as np
import logging
import random

import scipy
import scipy.stats

from B_function import *
from monte_carlo import MonteCarlo

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')

Z2B = Z2B_hungarian
Z2B_all = Z2B_hun_all


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
            Znk_is_0 = 0.5 # this happends already in logdiff
    return Znk_is_0


class UncollapsedGibbs(MonteCarlo):
    """
    sample the corpus to train the parameters
    """

    def _initialize(self, data, alpha=1.0, beta=1.0, sigma_a=1.0, sigma_x=1.0, initial_Z=None, initial_A=None, A_prior=None, draw="w"):

        if draw is "w":
            self.draw_new_feature = self.local_worst_match
        elif draw is "r":
            self.draw_new_feature = self.random_local
        else:
            print("draw method not found: {}".format(draw))
            self.draw_new_feature = self.local_worst_match

        self.B_count = 0
        self.B_same = 0

        self._counter = 0

        self._test_c = 0
        self.probZ = []

        self._alpha = alpha
        self._beta = beta
        self._sigma_x = sigma_x
        self._sigma_a = sigma_a

        # Data matrix
        self._X = data
        self._N = len(self._X)

        sigma_x_start = sigma_x
        self._sigma_xj = [sigma_x_start for i in range(self._N)]
        # for j in range(len(self._X)):
        #     self._sigma_xj.append(np.std(self._X[j]))

        # (self._N, self._D) = self._X.shape
        # self._D = self._X[0].shape[0]
        self._W = self._X[0].shape[1]
        self._F = np.max([self._X[n].shape[0] for n in range(self._N)])


        if initial_Z is None:
            # initialize Z from IBP(alpha)
            self._Z = np.zeros((self._N, 0), dtype=np.int) #self.initialize_Z()
        else:
            self._Z = initial_Z

        if self._Z.shape[1] > 75:
            self._Z = self._Z[:,:75]

        # assert (self._Z.shape[0] == self._N)

        # make sure Z matrix is a binary matrix
        assert (self._Z.dtype == np.int)
        # assert (self._Z.max() == 1 and self._Z.min() == 0)

        # record down the number of features
        self._K = self._Z.shape[1]

        if A_prior is None:
            self._A_prior = np.zeros((1, self._W))
        else:
            self._A_prior = A_prior
        assert (self._A_prior.shape == (1, self._W))

        if initial_A is None:
            # initialize A by random sampling from X
            self._A = self.initialize_A()
        else:
            self._A = initial_A
        # assert (self._A.shape == (self._K, self._D))

        # build B matrix
        self._B = Z2B_all(self._X, self._Z, self._A)
        #
        #
        # self.C = CAPACITY
        # self.STATE_CATACLYSM_INIT = np.zeros(shape=(self._N, self._K))
        # self.PA = np.ones(shape=(self._N, self._K)) * (self._N * self.C)  # Experimental
        # self.PB = np.ones(shape=(self._N, self._K)) * (self._N * self.C)  # Experimental
        # self.G = GAMMA
        # logging.debug("N: {}, K: {}, C: {}".format(self._N, self._K, self.C))

    @property
    def learning(self):
        self._counter += 1

        assert (self._Z.shape == (self._N, self._K))
        # assert (self._A.shape == (self._K, self._D))
        # assert (self._X.shape == (self._N, self._D))

        # sample every object (car)
        order = np.random.permutation(self._N)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_Zn(object_index)

            if self._metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                self.metropolis_hastings_K_new(object_index, singleton_features)

        # regularize matrices -> iterate through all (check)
        self.regularize_matrices()

        self.sample_A()

        # ??????
        # self._B = Z2B_all(self._X, self._Z, self._A)

        # TODO:
        if self._alpha_hyper_parameter is not None:
            self._alpha = self.sample_alpha()

        if self._sigma_x_hyper_parameter is not None:
            # self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter)
            # self._sigma_x = self.sample_sigma(self._sigma_x_hyper_parameter, self._X - np.dot(self._Z, self._A))
            # self._sigma_x = self.sample_sigma_x()
            self.update_sigma_x()
        if self._sigma_a_hyper_parameter is not None:
            # self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter)
            self._sigma_a = self.sample_sigma(self._sigma_a_hyper_parameter,
                                              self._A - np.tile(self._A_prior, (self._K, 1)))

        log_like = self.log_likelihood_model()
        return log_like

    def update_sigma_x(self):
        for j in range(len(self._X)):
            self._sigma_xj[j] = self.sample_sigma(self._sigma_x_hyper_parameter,
                                                  self._X[j] - np.dot(self._B[j], self._A))

    def initialize_A(self, X=None, Z=None):
        if X is None:
            X = self._X
        if Z is None:
            Z = self._Z
        K = self._K  # random sample K from X
        W = self._W
        A = np.zeros((K, W))
        X = np.vstack(X)
        np.random.shuffle(X)

        for i in range(K):
            A[i, :] = X[i, :]

        return A

    def sample_Zn(self, object_index):
        assert (type(object_index) == int or type(object_index) == np.int32 or type(object_index) == np.int64)

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(np.float)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        # ak = self._alpha / self._K
        # M = (new_m + ak) / (self._N + ak)
        # log_prob_z1 = np.log(M)
        # log_prob_z0 = np.log(1.0 - M)

        log_prob_z1 = np.log(new_m / self._N)
        log_prob_z0 = np.log(1.0 - new_m / self._N)

        # find all singleton features possessed by current object

        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]

        # store log-likelihood for comparison
        ps = []

        order = np.random.permutation(self._K)
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                # old_Znk = self._Z[object_index, feature_index]

                # try to use global feature (feature_index) in object (object_index) -> 0/1
                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0
                prob_z0, B_z0, diff_z0 = self.log_likelihood_X(object_index)
                prob_z0 += log_prob_z0[feature_index]
                # prob_z0 = np.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1
                prob_z1, B_z1, diff_z1 = self.log_likelihood_X(object_index)
                prob_z1 += log_prob_z1[feature_index]
                # prob_z1 = np.exp(prob_z1)

                Znk_is_0 = likelihood_ratio(prob_z0, prob_z1)
                ps.append((diff_z0, prob_z0, diff_z1, prob_z1, Znk_is_0))

                # logging.debug("object: {}, Feature: {}, log_like_0: {}, log_like_1: {}".format(object_index, feature_index, p0, p1))


                # TODO: replace Z if too many features
                if random.random() < Znk_is_0:
                    self._Z[object_index] = B_z0.sum(axis=0)
                    # self._Z[object_index, feature_index] = 0
                    self._B[object_index] = B_z0
                else:
                    self._Z[object_index] = B_z1.sum(axis=0)
                    # self._Z[object_index, feature_index] = 1
                    self._B[object_index] = B_z1

        self.probZ.append(ps)
        # self.regularize_Z_with_B(object_index)
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

        # if K_temp <= 0 and len(singleton_features) <= 0:
        if K_temp <= 0 and len(singleton_features) <= 0:
            return False

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-D matrix
        #

        # TODO: new feature from local
        # A_prior = np.tile(self._A_prior, (K_temp, 1)) # random from local
        # A_prior = self.random_local(object_index, K_temp)
        A_prior, _ = self.draw_new_feature(object_index, K_temp)
        A_temp = A_prior #+ np.random.normal(0, self._sigma_a, (K_temp, self._W))

        A_new = np.vstack((self._A[[k for k in list(range(self._K)) if k not in singleton_features], :], A_temp))
        # generate new z matrix row
        Z_new = np.hstack((self._Z[[object_index], [k for k in list(range(self._K)) if k not in singleton_features]],
                           np.ones(K_temp)))

        K_new = self._K + K_temp - len(singleton_features)

        # compute the probability of generating new features
        prob_new, B_new, _ = self.log_likelihood_X(object_index, Z_new, A_new)
        # prob_new = np.exp(prob_new)

        # construct the A_old and Z_old
        A_old = self._A
        Z_old = self._Z[object_index, :]
        K_old = self._K

        # assert (A_old.shape == (K_old, self._D))
        # assert (A_new.shape == (K_new, self._D))
        # assert (Z_old.shape == (len(object_index), K_old))
        # assert (Z_new.shape == (len(object_index), K_new))

        # compute the probability of using old features
        prob_old, B_old, _ = self.log_likelihood_X(object_index, Z_old, A_old)
        # prob_old = np.exp(prob_old)


        # compute the probability of generating new features
        prob_add = likelihood_ratio(prob_new, prob_old) # prob_new / (prob_old + prob_new)

        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < prob_add:
            # construct A_new and Z_new
            self._A = A_new
            self._Z = np.hstack((self._Z[:, [k for k in list(range(self._K)) if k not in singleton_features]],
                                 np.zeros((self._N, K_temp))))
            self._Z[object_index, :] = Z_new
            # TODO: MOD B

            B = self._B
            for i in range(len(B)):
                n_loc = B[i].shape[0]
                B[i] = np.delete(B[i], singleton_features, axis=1)
                B[i] = np.hstack((B[i], np.zeros((n_loc, K_temp))))
            self._B = B
            self._B[object_index] = B_new
            self.regularize_Z_with_B(object_index)
            self._K = K_new
            print("++++++ Added ", K_temp, "new local nodes.")
            return True
        else:
            print("------ nope! ", K_temp, "-", prob_old, prob_new, "->", prob_add)

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
                rand = np.random.normal(0, 1, (1, W))
                self._A[feature_index] = mean_i + var_i * rand
        return

    """
    sample noise variances, i.e., sigma_x
    """

    def sample_sigma_x(self):
        # X - ZA => mean(X - BA)
        N = self._N
        K = self._K
        W = self._W

        X = self._X
        Z = self._Z
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
        z_sum = Z.sum(axis=0).reshape((K, 1))  # occurrence count
        diff = diff / z_sum  # mean

        return self.sample_sigma(self._sigma_x_hyper_parameter,
                                 diff)

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """

    def regularize_matrices(self):

        assert (self._Z.shape == (self._N, self._K))
        Z_sum = np.sum(self._Z, axis=0)
        assert (len(Z_sum) == self._K)
        indices = np.nonzero(Z_sum)[0]
        # assert(np.min(indices)>=0 and np.max(indices)<self._K)
        not_used = [glob for glob in range(self._A.shape[0]) if glob not in indices]
        print("removed global feature: ", not_used)

        # print self._K, indices, [k for k in range(self._K) if k not in indices]
        self._Z = self._Z[:, [k for k in indices]]
        self._A = self._A[[k for k in indices], :]

        # iterate through B and remove from A and B
        B = self._B
        for i in range(self._N):
            B[i] = B[i][:, [k for k in indices]]
        self._B = B

        self._K = self._Z.shape[1]
        assert (self._Z.shape == (self._N, self._K))


        # assert (self._A.shape == (self._K, self._D))

    """
    compute the log-likelihood of the data X
    @param X: a 2-D np array
    @param Z: a 2-D np boolean array
    @param A: a 2-D np array, integrate A out if it is set to None
    """

    def log_likelihood_X(self, object_index=None, Z=None, A=None):
        X = self._X
        if object_index is not None and object_index < len(X):
            X = X[object_index]
        if A is None:
            A = self._A
        # else -> A = A_new or A_old
        if Z is None and object_index is not None:
            Z = self._Z[object_index]
        # else -> Z = Z_new or Z_old

        # assert (X.shape[0] == Z.shape[0])
        B = Z2B(X, Z, A)
        self._B[object_index] = B
        (N, D) = X.shape
        D = 1
        K = A.shape[0]  # feature count

        # assert (A.shape == (K, D))

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

        return log_likelihood, B, diff

    """
    compute the log-likelihood of the model
    """

    def log_likelihood_model(self):
        # TODO: ALL

        A = self._A
        Z = self._Z
        X = self._X

        # N: car count
        # D: global feature count
        # K: weight count
        # assert (len(X) == len(Z))
        N = len(Z)
        (K, D) = A.shape

        B = self._B
        # average log_like from all models
        log_likelihood = np.sum([self.log_likelihood_X(i)[0] for i in range(N)]) / N

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

    def regularize_Z_with_B(self, object_index):
        B = self._B[object_index]
        K = B.shape[1]
        non_zero_index = np.nonzero(np.sum(B, axis=0))[0]
        for i in range(K):
            if i not in non_zero_index:
                self._Z[object_index][i] = 0
        return

    # return a un-matched feature by random
    # TODO: if no unmatch, random from global.
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