import numpy as np
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os


# Re-optimized Unary Encoding (RUE)
class RUE:
    def __init__(self, epsilon, domain_size, cpu_count=0):
        # initialize the protocol
        self.d = domain_size
        self.epsilon = epsilon
        e_epsilon = np.e ** epsilon
        h = np.sqrt((domain_size - 1 + 1 / e_epsilon) / (domain_size - 1 + e_epsilon))
        self.p = 1 / (h + 1)
        self.q = 1 / (h * e_epsilon + 1)
        self.pure_p = self.p
        self.pure_q = self.q

        # initialize multiprocessing
        self.cpu_count = cpu_count
        system_cpu_count = multiprocessing.cpu_count()
        if cpu_count <= 0 or cpu_count > system_cpu_count:
            self.cpu_count = system_cpu_count
        if self.cpu_count > 61 and os.name == "nt":
            self.cpu_count = 61

    def perturb_data_list_sum_mp(self, data_list):
        """
        perturb with multiprocessing, i.e., split the data into cpu_count pieces and perturb each piece
        the communication cost of OUE is high, here we directly calculate suggest number of each value to save memory
        but this operation is indeed part of the job of aggregation
        to get the accurate aggregation time should use perturb_data_list_mp or perturb_data_list to get perturbed data
        """
        data_list_chunk = np.array_split(data_list, self.cpu_count)
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.perturb_data_list_sum, data_list_chunk))
        pool.shutdown()
        data_list_perturb_sum = np.array(results).sum(axis=0)
        return data_list_perturb_sum

    def perturb_data_list_sum(self, data_list):
        """
        get perturbed data and calculate suggest number of each value to save memory
        """
        data_list_perturb_sum = np.zeros(self.d)
        n = np.size(data_list)
        for i in range(n):
            bit_vector = self.perturb(data_list[i])
            data_list_perturb_sum += np.unpackbits(bit_vector, count=self.d)
        return data_list_perturb_sum

    def perturb_data_list_mp(self, data_list):
        """
        perturb with multiprocessing, i.e., split the data into cpu_count pieces and perturb each piece
        the communication cost of OUE is high, this function will cost a lot of memory
        the memory cost is about several MB to GB depending on the domain size and the number of data
        """
        data_list_chunk = np.array_split(data_list, self.cpu_count)
        pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        results = list(pool.map(self.perturb_data_list, data_list_chunk))
        pool.shutdown()
        data_list_perturb = np.vstack(results)
        return data_list_perturb

    def perturb_data_list(self, data_list):
        """
        perturb a list of data
        """
        n = np.size(data_list)
        data_list_perturb = np.zeros((n, math.ceil(self.d / 8)), dtype="uint8")
        for i in range(n):
            data_list_perturb[i] = self.perturb(data_list[i])
        return data_list_perturb

    def perturb(self, v):
        """
        perturb a private value with RUE
        """
        rng = np.random.default_rng()
        index_perturb = rng.choice([1, 0], size=self.d, p=[self.q, 1 - self.q])
        index_perturb[v] = 0
        if rng.random() < self.p:
            index_perturb[v] = 1
        bit_vector = np.packbits(index_perturb)
        return bit_vector

    def aggregate_data_list_sum(self, data_list_perturb_sum, n):
        """
        estimate the true count with the suggest number
        this function handles the result of perturb_data_list_sum_mp or perturb_data_list_sum
        """
        data_list_estimate = (data_list_perturb_sum - n * self.pure_q) / (
                self.pure_p - self.pure_q)
        return data_list_estimate

    def aggregate_data_list(self, data_list_perturb):
        """
        aggregate the perturbed datalist to estimate the true count
        """
        n = np.shape(data_list_perturb)[0]
        data_list_perturb_sum = np.zeros(self.d)
        for i in range(n):
            data_list_perturb_sum += np.unpackbits(data_list_perturb[i], count=self.d)
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
