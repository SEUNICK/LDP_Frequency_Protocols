import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os


# Generalized Randomized Response (GRR)
class GRR:
    def __init__(self, epsilon, domain_size, cpu_count=0):
        # initialize the protocol
        self.d = domain_size
        self.epsilon = epsilon
        self.p = np.e ** epsilon / (np.e ** epsilon + domain_size - 1)
        self.q = 1 / (np.e ** epsilon + domain_size - 1)
        self.pure_p = self.p
        self.pure_q = self.q

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
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.perturb_data_list, data_list_chunk))
        pool.shutdown()
        data_list_perturb = np.hstack(results)
        return data_list_perturb

    def perturb_data_list(self, data_list):
        """
        perturb a list of data
        """
        data_list_perturb = np.zeros(shape=np.size(data_list), dtype=int)
        for i in range(np.size(data_list)):
            data_list_perturb[i] = self.perturb(data_list[i])
        return data_list_perturb

    def perturb(self, v):
        """
        perturb a private value with GRR
        """
        rng = np.random.default_rng()
        rand = rng.random()
        if rand <= self.p:
            index_perturb = v
        else:
            index_perturb = rng.integers(self.d - 1)
            if index_perturb >= v:
                index_perturb += 1
        return index_perturb

    def aggregate_data_list(self, data_list_perturb):
        """
        aggregate the perturbed datalist to estimate the true count
        """
        n = np.size(data_list_perturb)
        data_list_perturb_sum = np.zeros(self.d)
        for i in range(n):
            data_list_perturb_sum[data_list_perturb[i]] += 1
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
