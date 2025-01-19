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
import pyarrow.parquet as pq
import timeit


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


def run_protocol_runtime(data_list, epsilon, d, protocol_name):
    """
    get the aggregation time for the specified protocol and parameters without multiprocessing
    """
    global prot, data_perturb
    prot = eval(protocol_name)(epsilon, d, run_cpu_count)

    # only multiprocessing in perturbation
    data_perturb = prot.perturb_data_list_mp(data_list)
    runtime = timeit.timeit(stmt="prot.aggregate_data_list(data_perturb)", globals=globals(), number=1)
    return runtime


def scan_d_runtime(data_name, epsilon, d_range, protocol_list):
    """
    scan domain size d and run each specified d and protocol to get the aggregation time
    """
    protocol_size = np.size(protocol_list)
    runtime_list = np.zeros((protocol_size, d_range.size), dtype='float64')
    for i in range(np.size(d_range)):
        data_list, data_list_sum = get_data_list(data_name, d_range[i])
        for k in range(protocol_size):
            runtime = run_protocol_runtime(data_list, epsilon, d_range[i], protocol_list[k])
            runtime_list[k][i] = runtime
            print(data_name, protocol_list[k], "d:", d_range[i], "runtime:", runtime)
    path = "results/" + data_name + "_runtime.csv"
    first_row = np.append(protocol_list, "d")
    write_csv(path, first_row, runtime_list, d_range)
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
    run_epsilon = 4.0
    run_d_range = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    run_protocol_list = np.array(["GRR", "OUE", "OLH", "SS", "RUE", "RLH", "RWS"])

    # run synthetic data
    name = 'synthetic'
    scan_d_runtime(name, run_epsilon, run_d_range, run_protocol_list)

    # run taxi data
    name = 'taxi'
    normalized_taxi = read_taxi()
    scan_d_runtime(name, run_epsilon, run_d_range, run_protocol_list)

    print("end")
