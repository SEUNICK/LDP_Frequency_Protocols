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
    read the average MSE in saved CSV result
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


def get_ana_MSE(protocol_name, epsilon, d, n):
    prot = eval(protocol_name)(epsilon, d)
    return prot.get_ana_MSE(n)


def get_ana_MSE_with_d_range(protocol_name, epsilon, d_range, n=1):
    y = np.zeros(np.size(d_range))
    for i in range(np.size(d_range)):
        y[i] = get_ana_MSE(protocol_name, epsilon, d_range[i], n)
    return y


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_epsilon = 4.0
    run_d_range = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])

    # calculate the analytical MSE of LDP protocols
    ana_MSE_syn_d = {}
    ana_MSE_taxi_d = {}
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        ana_MSE_syn_d[name] = get_ana_MSE_with_d_range(run_protocol_list[i], run_epsilon, run_d_range, 100000)
        ana_MSE_taxi_d[name] = get_ana_MSE_with_d_range(run_protocol_list[i], run_epsilon, run_d_range, 3582628)

    # separating protocols with similar results (OUE,OLH) (RUE,RLH) (SS,RWS)
    run_protocol_list1 = np.array(["GRR", "OUE", "RUE", "SS"])
    run_protocol_list2 = np.array(["OLH", "RLH", "RWS"])

    # empirical style
    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}

    # analytical style
    style_list2 = {"GRR": "C7x-", "OUE": "C8D-", "RUE": "C9v-", "OLH": "C8D-", "RLH": "C9v-", "SS": "mp-", "RWS": "mp-"}

    # read the average empirical MSE in saved CSV result
    x_syn, emp_MSE_syn_d = read_result("results/synthetic_d_range_mse.csv")
    x_taxi, emp_MSE_taxi_d = read_result("results/taxi_d_range_mse.csv")

    # plot figures
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list1.size):
        name = run_protocol_list1[i]
        plt.plot(run_d_range, emp_MSE_syn_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
        plt.plot(run_d_range, ana_MSE_syn_d[name], style_list2[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    h, l = ax.get_legend_handles_labels()
    ph = [plt.plot([], marker="", ls="")[0]] * 2
    handles = ph + h
    labels = ["Empirical:", "Analytical:"] + l
    leg = plt.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
    plt.tight_layout()
    a_bottom, a_top = plt.ylim()
    plt.savefig("draw/ana_emp_MSE_syn_d_1.eps")
    plt.show()

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list2.size):
        name = run_protocol_list2[i]
        plt.plot(run_d_range, emp_MSE_syn_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
        plt.plot(run_d_range, ana_MSE_syn_d[name], style_list2[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    h, l = ax.get_legend_handles_labels()
    ph = [plt.plot([], marker="", ls="")[0]] * 2
    handles = ph + h
    labels = ["Empirical:", "Analytical:"] + l
    leg = plt.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    plt.ylim(a_bottom, a_top)

    plt.tight_layout()
    plt.savefig("draw/ana_emp_MSE_syn_d_2.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list1.size):
        name = run_protocol_list1[i]
        plt.plot(run_d_range, emp_MSE_taxi_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
        plt.plot(run_d_range, ana_MSE_taxi_d[name], style_list2[name], linewidth=1, markerfacecolor='none',
                 markersize=5,
                 label=name)

    plt.tight_layout()
    a_bottom, a_top = plt.ylim()
    plt.savefig("draw/ana_emp_MSE_taxi_d_1.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(run_d_range)
    for i in range(run_protocol_list2.size):
        name = run_protocol_list2[i]
        plt.plot(run_d_range, emp_MSE_taxi_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
        plt.plot(run_d_range, ana_MSE_taxi_d[name], style_list2[name], linewidth=1, markerfacecolor='none',
                 markersize=5,
                 label=name)

    plt.ylim(a_bottom, a_top)
    plt.tight_layout()
    plt.savefig("draw/ana_emp_MSE_taxi_d_2.eps")
    plt.show()

    print("end")
