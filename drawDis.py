import numpy as np
import math
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

# for PDF backend
plt.rcParams['pdf.fonttype'] = 42
# for PS backend
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 6

plt.rcParams['legend.fontsize'] = 6

plt.rcParams['xtick.labelsize'] = 6

plt.rcParams['ytick.labelsize'] = 6


def generate_zipf_data(s, d, n):
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


def get_data_list_zipf(d):
    data_list = generate_zipf_data(1.1, d, 100000)
    data_list_sum = np.zeros(d)
    for i in range(np.size(data_list)):
        data_list_sum[data_list[i]] += 1
    return data_list, data_list_sum


def read_taxi():
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
    if data_name == "synthetic":
        data_list, data_list_sum = get_data_list_zipf(d)
    if data_name == "taxi":
        data_list, data_list_sum = get_data_list_taxi(d)
    return data_list, data_list_sum


def get_data_list_zipf(d):
    data_list = generate_zipf_data(1.1, d, 100000)
    data_list_sum = np.zeros(d)
    for i in range(np.size(data_list)):
        data_list_sum[data_list[i]] += 1
    return data_list, data_list_sum


def get_data_list_taxi(d):
    n = np.size(normalized_taxi)
    data_list = np.zeros(n, dtype=int)
    data_list_sum = np.zeros(d)
    for i in range(n):
        data_list[i] = math.floor(normalized_taxi[i] * d)
        if data_list[i] >= d:
            data_list[i] = d - 1
        data_list_sum[data_list[i]] += 1
    return data_list, data_list_sum


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_d = 128

    # get experimental dataset
    normalized_taxi = read_taxi()
    data_list_syn, data_list_sum_syn = get_data_list("synthetic", run_d)
    data_list_taxi, data_list_sum_taxi = get_data_list("taxi", run_d)

    # plot figures
    bins = np.arange(run_d + 1)
    plt.figure(figsize=(3.5, 2))
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.xlabel("Value")
    plt.xticks(np.array([1, 16, 32, 48, 64, 80, 96, 112, 128]))
    plt.hist(data_list_syn, bins=bins, density=True, align="right")
    plt.tight_layout()
    plt.savefig('draw/synthetic.eps')
    plt.show()

    plt.figure(figsize=(3.5, 2))
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.xlabel("Value")
    plt.xticks(np.array([1, 16, 32, 48, 64, 80, 96, 112, 128]))
    plt.hist(data_list_taxi, bins=bins, density=True, align="right")
    plt.tight_layout()
    plt.savefig('draw/taxi.eps')
    plt.show()

    print("end")
