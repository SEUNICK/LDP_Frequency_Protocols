from fp.GRR import GRR
from fp.OLH import OLH
from fp.OUE import OUE
from fp.RLH import RLH
from fp.RUE import RUE
from fp.SS import SS
from fp.RWS import RWS
import numpy as np
import math
import csv
import os
import pyarrow.parquet as pq


def generate_zipf_data(s, d, n):
    """
    generate zipf's data with domain {0,1,....,d-1}
    :param s: parameter of zipf's distribution
    :param d: domain size
    :param n: total number
    """
    # generate the same distribution for each experiment
    rng = np.random.default_rng(6)
    # values to sample
    v = np.arange(1, d + 1)
    # probabilities
    p = 1.0 / np.power(v, s)
    # normalized probabilities
    p /= np.sum(p)
    rng.choice(v, size=n, replace=True, p=p)
    data = rng.choice(v, size=n, replace=True, p=p)
    data = data - 1
    return data


def read_taxi():
    """
    read and normalize taxi time records to [0,1)
    """
    path = "yellow_tripdata_2024-03.parquet"
    trips = pq.read_table(path)
    date_time_list = trips["tpep_pickup_datetime"].to_numpy()
    time_list = np.zeros(date_time_list.size)
    for i in range(date_time_list.size):
        time_srt = date_time_list[i].astype(str)[11:19].split(":")
        seconds = int(time_srt[0]) * 3600 + int(time_srt[1]) * 60 + int(time_srt[2])
        time_list[i] = seconds
    normalized_taxi = time_list / (24 * 60 * 60)
    return normalized_taxi


def get_data_list(data_name, d):
    """
    get formatted experiment dataset by name and domain size
    :return:
        data_list: data for experiment with domain {0,1,....,d-1}
        data_list_sum: true count for each value in the domain
    """
    if data_name == "synthetic":
        data_list, data_list_sum = get_data_list_zipf(d)
    if data_name == "taxi":
        data_list, data_list_sum = get_data_list_taxi(d)
    return data_list, data_list_sum


def get_data_list_zipf(d):
    """
    get formatted synthetic zipf's distribution data_list with domain {0,1,....,d-1} and data_list_sum
    """
    data_list = generate_zipf_data(1.1, d, 100000)
    data_list_sum = np.zeros(d)
    for i in range(np.size(data_list)):
        data_list_sum[data_list[i]] += 1
    return data_list, data_list_sum


def get_data_list_taxi(d):
    """
    get formatted taxi data_list with domain {0,1,....,d-1} and data_list_sum
    """
    n = np.size(normalized_taxi)
    data_list = np.zeros(n, dtype=int)
    data_list_sum = np.zeros(d)
    for i in range(n):
        data_list[i] = math.floor(normalized_taxi[i] * d)
        if data_list[i] >= d:
            data_list[i] = d - 1
        data_list_sum[data_list[i]] += 1
    return data_list, data_list_sum


def get_save_name(protocol_name, dataset_name, d, epsilon):
    save_name = protocol_name + "_" + dataset_name + "_" + str(d) + "_" + str(epsilon) + "_"
    return save_name


def get_mse(x_est, x_theory):
    diff = x_est - x_theory
    diff2 = diff * diff
    return diff2.sum() / np.size(diff2)


def run_protocol(data_list, data_list_sum, epsilon, d, protocol_name, data_name, repeat_i=0):
    """
    the i-th run of the specified protocol and parameters and get the MSE of i-th experiment
    """
    n = np.size(data_list)
    # initialize the ldp frequency protocol
    prot = eval(protocol_name)(epsilon, d, run_cpu_count)

    # run the i-th experiment and save the estimate result and true result
    # _mp means multiprocessing to save time
    # _sum means directly calculate suggest number instead of raw perturbed data to save memory
    # but this operation is indeed part of the job of aggregation
    # protocol_list1 only multiprocessing in perturbation
    # protocol_list1 use _sum to save the memory due to the high communication cost
    protocol_list1 = np.array(["OUE", "SS", "RUE"])

    # protocol_list2 multiprocessing in both perturbation and aggregation
    protocol_list2 = np.array(["OLH", "RLH", "RWS"])

    if not os.path.exists("results/" + protocol_name):
        os.makedirs("results/" + protocol_name)
    if protocol_name in protocol_list1:
        data_perturb = prot.perturb_data_list_sum_mp(data_list)
        data_estimate = prot.aggregate_data_list_sum(data_perturb, n)
    else:
        data_perturb = prot.perturb_data_list_mp(data_list)
        if protocol_name in protocol_list2:
            data_estimate = prot.aggregate_data_list_mp(data_perturb)
        else:
            data_estimate = prot.aggregate_data_list(data_perturb)

    # calculate the MSE from count to frequency
    mse = get_mse(data_estimate, data_list_sum) / n / n
    print(protocol_name, mse)
    return mse


def get_ana_MSE(protocol_name, epsilon, d, n=1):
    prot = eval(protocol_name)(epsilon, d)
    return prot.get_ana_MSE(n)


if __name__ == '__main__':
    print("start test")
    print('-' * 10)
    # experimental parameters
    run_cpu_count = 0
    run_repeat_time = 1
    run_protocol_list = np.array(["GRR", "OUE", "OLH", "SS", "RUE", "RLH", "RWS"])

    run_epsilon = 4.0
    run_d = 32

    # test generate and read data
    print("test generate synthetic data")

    name = 'synthetic'
    data_list, data_list_sum = get_data_list(name, run_d)

    print("generate synthetic data success")
    print('-' * 10)

    print("test read taxi data")
    name = 'taxi'
    normalized_taxi = read_taxi()
    data_list, data_list_sum = get_data_list(name, run_d)

    print("read taxi data success")
    print('-' * 10)

    # test get analytical n*MSE for each protocol
    print("test calculate analytical n*MSE with epsilon:", run_epsilon, "and d:", run_d)

    for protocol_name in run_protocol_list:
        ana_nMSE = get_ana_MSE(protocol_name, run_epsilon, run_d)
        print(protocol_name, ana_nMSE)

    print("calculate taxi data success")
    print('-' * 10)

    # test get empirical MSE with synthetic data
    print("test get empirical MSE with synthetic data")
    name = 'synthetic'
    print("epsilon:", run_epsilon, ", d:", run_d)
    data_list, data_list_sum = get_data_list(name, run_d)
    for protocol_name in run_protocol_list:
        run_protocol(data_list, data_list_sum, run_epsilon, run_d, protocol_name, name)
    print("get empirical MSE success")
    print('-' * 10)

    print("finish all test")
