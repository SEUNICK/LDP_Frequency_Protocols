from fp.GRR import GRR
from fp.OLH import OLH
from fp.OUE import OUE
from fp.RLH import RLH
from fp.RUE import RUE
from fp.SS import SS
from fp.RWS import RWS
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


def get_ana_MSE(protocol_name, epsilon, d, n):
    prot = eval(protocol_name)(epsilon, d)
    return prot.get_ana_MSE(n)


def get_ana_MSE_with_d_range(protocol_name, epsilon, d_range, n=1):
    y = np.zeros(np.size(d_range))
    for i in range(np.size(d_range)):
        y[i] = get_ana_MSE(protocol_name, epsilon, d_range[i], n)
    return y


def get_ana_MSE_with_epsilon_range(protocol_name, epsilon_range, d, n=1):
    y = np.zeros(np.size(epsilon_range))
    for i in range(np.size(epsilon_range)):
        y[i] = get_ana_MSE(protocol_name, epsilon_range[i], d, n)
    return y


def write_csv(path, first_row, mes_list, range_list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        line = np.zeros(np.size(first_row), dtype='float64')
        for i in range(range_list.size):
            for j in range(first_row.size - 1):
                line[j] = mes_list[first_row[j]][i]
            line[first_row.size - 1] = range_list[i]
            writer.writerow(line)


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_epsilon = 4.0
    run_d_range = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])

    if not os.path.exists("draw"):
        os.makedirs("draw")

    # calculate the analytical n*MSE of LDP protocols
    ana_nMSE_d = {}
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        ana_nMSE_d[name] = get_ana_MSE_with_d_range(run_protocol_list[i], run_epsilon, run_d_range)

    # save the analytical n*MSE results for Table 3
    path = "results/nMSE_d_range_mse.csv"
    first_row = np.append(run_protocol_list, "d")
    write_csv(path, first_row, ana_nMSE_d, run_d_range)

    # separating protocols with similar results (OUE,OLH) (RUE,RLH) (SS,RWS)
    run_protocol_list1 = np.array(["GRR", "OUE", "RUE", "SS"])
    run_protocol_list2 = np.array(["OLH", "RLH", "RWS"])

    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}

    # plot figures
    plt.figure(figsize=(3.5, 2.5))
    plt.ylabel("Analytical n·MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list1.size):
        name = run_protocol_list1[i]
        plt.plot(run_d_range, ana_nMSE_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.tight_layout()
    a_bottom, a_top = plt.ylim()
    plt.savefig("draw/ana_nMSE_d_range1.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.5))
    plt.ylabel("Analytical n·MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list2.size):
        name = run_protocol_list2[i]
        plt.plot(run_d_range, ana_nMSE_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    plt.ylim(a_bottom, a_top)
    plt.tight_layout()
    plt.savefig("draw/ana_nMSE_d_range2.eps")
    plt.show()

    print("end")
