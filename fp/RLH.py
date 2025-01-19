import numpy as np
import multiprocessing
import math
from concurrent.futures import ProcessPoolExecutor
import os


# Re-optimized Local Hashing(OLH)
class RLH:
    def __init__(self, epsilon, domain_size, cpu_count=0):
        # initialize the protocol
        self.d = domain_size
        self.epsilon = epsilon
        e_epsilon = np.e ** epsilon
        h = np.sqrt((domain_size - 1 + 1 / e_epsilon) / (domain_size - 1 + e_epsilon))

        # get the optimal rounded g with analytical MSE
        g = h * e_epsilon + 1
        g1 = int(g)
        g2 = int(math.ceil(g))
        mse1 = self.get_ana_MSE_by_g(g1)
        mse2 = self.get_ana_MSE_by_g(g2)
        self.g = g1
        if mse2 < mse1:
            self.g = g2
        self.p = e_epsilon / (e_epsilon + self.g - 1)
        self.q = 1 / (e_epsilon + self.g - 1)
        self.pure_p = self.p
        self.pure_q = 1 / self.g

        # initialize multiprocessing
        self.cpu_count = cpu_count
        system_cpu_count = multiprocessing.cpu_count()
        if cpu_count <= 0 or cpu_count > system_cpu_count:
            self.cpu_count = system_cpu_count
        if self.cpu_count > 61 and os.name == "nt":
            self.cpu_count = 61

    def perturb_data_list_mp(self, data_list):
        """
        perturb with multiprocessing, i.e., split the data into cpu_count pieces and perturb each piece,
        also attach each value with the user id or seed
        """
        data_list_chunk = np.array_split(data_list, self.cpu_count)
        start_list = np.zeros(self.cpu_count, dtype=int)
        start = 0
        for i in range(self.cpu_count):
            start_list[i] = start
            start += np.size(data_list_chunk[i])
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.perturb_data_list, data_list_chunk, start_list))
        pool.shutdown()
        data_list_perturb = np.concatenate(results, axis=0)
        return data_list_perturb

    def perturb_data_list(self, data_list, start_seed=0):
        """
        perturb a list of data
        """
        n = np.size(data_list)
        data_list_perturb = np.zeros((n, 2), dtype=int)
        for i in range(n):
            seed = start_seed + i
            data_list_perturb[i] = self.perturb(data_list[i], seed)
        return data_list_perturb

    def perturb(self, v, seed):
        """
        perturb a private value with RLH
        """
        rng1 = np.random.default_rng(seed)
        b = rng1.integers(self.g, size=self.d)
        y = b[v]
        rng2 = np.random.default_rng()
        p_sample = rng2.random()
        if p_sample > self.p - self.q:
            y = rng2.integers(self.g)
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
            b = rng.integers(self.g, size=self.d)
            data_list_perturb_sum[b == data_list_perturb[i][0]] += 1
        a = 1.0 * self.g / (self.p * self.g - 1)
        b = 1.0 * n / (self.p * self.g - 1)
        data_list_estimate = a * data_list_perturb_sum - b
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

    def get_ana_MSE_by_g(self, g, n=1):
        """
        calculate the analytical MSE for different g to get the optimal rounded g
        """
        d = self.d
        e_epsilon = np.e ** self.epsilon
        pure_p = e_epsilon / (e_epsilon + g - 1)
        pure_q = 1 / g
        mse = pure_q * (1 - pure_q) / (pure_p - pure_q) / (pure_p - pure_q) / n + (1 - pure_p - pure_q) / d / (
                pure_p - pure_q) / n
        return mse
