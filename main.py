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
    print("taxi records number", normalized_taxi.size)
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

    # determine whether the result of the i-th run exists
    save_name = get_save_name(protocol_name, data_name, d, epsilon)
    save_path = "results/" + protocol_name + "/" + save_name + str(repeat_i) + ".npz"
    if os.path.exists(save_path):
        # read saved results
        npzfile = np.load(save_path)
        data_estimate = npzfile["estimate_data"]
        data_list_sum = npzfile["data_list_sum"]
    else:
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
        np.savez(save_path, estimate_data=data_estimate,
                 data_list_sum=data_list_sum)

    # calculate the MSE from count to frequency
    mse = get_mse(data_estimate, data_list_sum) / n / n
    print(protocol_name, mse)
    return mse


def scan_epsilon(data_name, epsilon_range, d, repeat_time, protocol_list):
    """
    scan epsilon and run each specified epsilon and protocol multi-times and get the average MSE for each epsilon and protocol
    """
    data_list, data_list_sum = get_data_list(data_name, d)
    protocol_size = np.size(protocol_list)
    mes_list = np.zeros((protocol_size, epsilon_range.size), dtype='float64')
    for i in range(epsilon_range.size):
        mse_total_list = np.zeros((protocol_size), dtype='float64')
        for j in range(repeat_time):
            print(data_name, "epsilon:", epsilon_range[i], "repeat_time:", j)
            for k in range(protocol_size):
                mse_tmp = run_protocol(data_list, data_list_sum, epsilon_range[i], d, protocol_list[k], data_name, j)
                mse_total_list[k] += mse_tmp
        for k in range(protocol_size):
            mes_list[k][i] = mse_total_list[k] / repeat_time
    path = "results/" + data_name + "_epsilon_range_mse.csv"
    first_row = np.append(protocol_list, "epsilon")
    write_csv(path, first_row, mes_list, epsilon_range)
    return 0


def scan_d(data_name, epsilon, d_range, repeat_time, protocol_list):
    """
    scan domain size d and run each specified d and protocol multi-times and get the average MSE for each d and protocol
    """
    protocol_size = np.size(protocol_list)
    mes_list = np.zeros((protocol_size, d_range.size), dtype='float64')
    for i in range(np.size(d_range)):
        data_list, data_list_sum = get_data_list(data_name, d_range[i])
        mse_total_list = np.zeros((protocol_size), dtype='float64')
        for j in range(repeat_time):
            print(name, "d:", d_range[i], "repeat_time:", j)
            for k in range(protocol_size):
                mse_tmp = run_protocol(data_list, data_list_sum, epsilon, d_range[i], protocol_list[k], data_name, j)
                mse_total_list[k] += mse_tmp
        for k in range(protocol_size):
            mes_list[k][i] = mse_total_list[k] / repeat_time

    path = "results/" + data_name + "_d_range_mse.csv"
    first_row = np.append(protocol_list, "d")
    write_csv(path, first_row, mes_list, d_range)
    return 0


def write_csv(path, first_row, mes_list, range_list):
    """
    save the average experiment result into a CSV file
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        line = np.zeros(np.size(first_row), dtype='float64')
        for i in range(range_list.size):
            for j in range(np.shape(mes_list)[0]):
                line[j] = mes_list[j][i]
            line[np.size(first_row) - 1] = range_list[i]
            writer.writerow(line)


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_cpu_count = 0
    run_repeat_time = 100
    run_protocol_list = np.array(["GRR", "OUE", "OLH", "SS", "RUE", "RLH", "RWS"])

    # scan d parameters
    run_epsilon = 4.0
    run_d_range = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    # scan epsilon parameters
    run_d = 128
    run_epsilon_range = np.arange(1, 5.1, 0.25)

    # run synthetic data
    name = 'synthetic'
    scan_d(name, run_epsilon, run_d_range, run_repeat_time, run_protocol_list)
    scan_epsilon(name, run_epsilon_range, run_d, run_repeat_time, run_protocol_list)

    # run taxi data
    name = 'taxi'
    normalized_taxi = read_taxi()
    scan_d(name, run_epsilon, run_d_range, run_repeat_time, run_protocol_list)
    scan_epsilon(name, run_epsilon_range, run_d, run_repeat_time, run_protocol_list)

    print("end")
