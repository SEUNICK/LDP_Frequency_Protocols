import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# for PDF backend
plt.rcParams['pdf.fonttype'] = 42
# for PS backend
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 6

plt.rcParams['legend.fontsize'] = 6

plt.rcParams['xtick.labelsize'] = 6

plt.rcParams['ytick.labelsize'] = 6


def read_result(path):
    """
    read the runtime in saved CSV result
    """
    with open(path, mode='r') as file:
        reader = csv.DictReader(file)
        key_list = reader.fieldnames
        csv_dist_list = list(reader)
    x_len = len(csv_dist_list)
    y_set = {}
    for key in key_list:
        y_set[key] = np.empty(x_len)
    for i in range(x_len):
        for key in key_list:
            y_set[key][i] = csv_dist_list[i][key]
    x = y_set[key_list[len(key_list) - 1]]
    return x, y_set


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])

    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}

    if not os.path.exists("draw"):
        os.makedirs("draw")

    # plot figures
    x_syn, runtime_syn = read_result("results/synthetic_runtime.csv")
    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("Aggregation Time(seconds)")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(x_syn)
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_syn, runtime_syn[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/runtime_syn.eps")
    plt.show()

    x_taxi, runtime_taxi = read_result("results/taxi_runtime.csv")
    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("Aggregation Time(seconds)")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(x_syn)
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_taxi, runtime_taxi[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/runtime_taxi.eps")
    plt.show()

    print("end")
