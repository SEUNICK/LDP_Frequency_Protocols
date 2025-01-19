import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math
import os


# Random Wheel Spinner (RWS)
class RWS:
    def __init__(self, epsilon, domain_size, cpu_count=0):
        # initialize the protocol
        self.d = domain_size
        self.epsilon = epsilon
        e_epsilon = np.e ** epsilon

        # get the optimal rounded k with analytical MSE
        k = domain_size / (e_epsilon + 1)
        self.k = 1
        if k > 1:
            k1 = int(k)
            k2 = int(math.ceil(k))
            mse1 = self.get_ana_MSE_by_k(k1)
            mse2 = self.get_ana_MSE_by_k(k2)
            self.k = k1
            if mse2 < mse1:
                self.k = k2
        k = self.k
        self.p = e_epsilon / (k * e_epsilon + domain_size - k)
        self.q = 1 / (k * e_epsilon + domain_size - k)
        self.pure_p = k * self.p
        self.pure_q = self.pure_p * (k - 1) / (domain_size - 1) + (1 - self.pure_p) * k / (domain_size - 1)

        # initialize multiprocessing
        self.cpu_count = cpu_count
        system_cpu_count = multiprocessing.cpu_count()
        if cpu_count <= 0 or cpu_count > system_cpu_count:
            self.cpu_count = system_cpu_count
        if self.cpu_count > 61 and os.name == "nt":
            self.cpu_count = 61

    def perturb_data_list_mp(self, data_list):
        """
        perturb with multiprocessing, i.e., split the data into cpu_count pieces and perturb each piece
        """
        data_list_chunk = np.array_split(data_list, self.cpu_count)
        n = np.size(data_list)
        n_list = np.full(self.cpu_count, n)
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.perturb_data_list, data_list_chunk, n_list))
        pool.shutdown()
        data_list_perturb = np.concatenate(results, axis=0)
        return data_list_perturb

    def perturb_data_list(self, data_list, seed_n=0):
        """
        perturb a list of data and randomly generate the seed to generate the subset
        """
        if seed_n == 0:
            seed_n = np.size(data_list)
        data_list_perturb = np.zeros((np.size(data_list), 2), dtype=int)
        rng = np.random.default_rng()
        for i in range(np.size(data_list)):
            seed = rng.integers(seed_n)
            data_list_perturb[i] = self.perturb(data_list[i], seed)
        return data_list_perturb

    def perturb(self, v, seed):
        """
        perturb a private value with RWS
        """
        rng1 = np.random.default_rng(seed)
        subset = rng1.choice(self.d, self.k, replace=False)
        rng2 = np.random.default_rng()
        p_y = np.full(self.d, self.q)
        p_y[(v - subset) % self.d] = self.p
        y = rng2.choice(self.d, p=p_y)
        return [y, seed]

    def aggregate_data_list_mp(self, data_list_perturb):
        """
        aggregate with multiprocessing, i.e., split the perturbed data into cpu_count pieces and aggregate each piece
        """
        data_list_perturb_chunk = np.array_split(data_list_perturb, self.cpu_count)
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.aggregate_data_list, data_list_perturb_chunk))
        pool.shutdown()
        data_list_estimate = np.sum(results, axis=0)
        return data_list_estimate

    def aggregate_data_list(self, data_list_perturb):
        """
        aggregate the perturbed data to estimate the true count
        """
        data_list_perturb_sum = np.zeros(self.d)
        n = np.shape(data_list_perturb)[0]
        for i in range(n):
            seed = data_list_perturb[i][1]
            rng = np.random.default_rng(seed)
            subset = rng.choice(self.d, self.k, replace=False)
            subset_support = (subset + data_list_perturb[i][0]) % self.d
            data_list_perturb_sum[subset_support] += 1
        data_list_estimate = (data_list_perturb_sum - n * self.pure_q) / (
                self.pure_p - self.pure_q)
        return data_list_estimate

    def get_ana_MSE(self, n=1):
        """
        calculate the analytical MSE for GRR with initialize parameters and specified n,
        analytical n*MSE is equivalent to the analytical MSE with n=1
        """
        d = self.d
        pure_p = self.pure_p
        pure_q = self.pure_q
        mse = pure_q * (1 - pure_q) / (pure_p - pure_q) / (pure_p - pure_q) / n + (1 - pure_p - pure_q) / d / (
                pure_p - pure_q) / n
        return mse

    def get_ana_MSE_by_k(self, k, n=1):
        """
        calculate the analytical MSE for different k to get the optimal rounded k
        """
        d = self.d
        e_epsilon = np.e ** self.epsilon
        pure_p = k * e_epsilon / (k * e_epsilon + d - k)
        pure_q = pure_p * (k - 1) / (d - 1) + (1 - pure_p) * k / (d - 1)
        mse = pure_q * (1 - pure_q) / (pure_p - pure_q) / (pure_p - pure_q) / n + (1 - pure_p - pure_q) / d / (
                pure_p - pure_q) / n
        return mse
