import numpy as np
import multiprocessing
import xxhash
from concurrent.futures import ProcessPoolExecutor
import os


# this protocol is based on the code of Wang (https://github.com/vvv214/LDP_Protocols/blob/master/olh.py)
# Optimized Local Hashing(OLH)
class OLH:
    def __init__(self, epsilon, domain_size, cpu_count=0):
        # initialize the protocol
        self.d = domain_size
        self.epsilon = epsilon
        self.g = int(round(np.e ** epsilon)) + 1
        self.p = np.e ** epsilon / (np.e ** epsilon + self.g - 1)
        self.q = 1 / (np.e ** epsilon + self.g - 1)
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
        perturb a private value with OLH
        """
        x = (xxhash.xxh32(str(v), seed=seed).intdigest() % self.g)
        y = x
        rng = np.random.default_rng()
        p_sample = rng.random()
        if p_sample > self.p - self.q:
            y = rng.integers(self.g)
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
            for v in range(self.d):
                if data_list_perturb[i][0] == (
                        xxhash.xxh32(str(v), seed=data_list_perturb[i][1]).intdigest() % self.g):
                    data_list_perturb_sum[v] += 1

        # this part is equivalent to
        # data_list_estimate = (data_list_perturb_sum - n * self.pure_q) / (self.pure_p - self.pure_q)
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
