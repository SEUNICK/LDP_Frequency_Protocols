import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.pylab as lab
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


def draw_legend():
    fig = lab.figure()
    figlegend = lab.figure(figsize=(7, 0.5), dpi=300)
    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}
    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])
    ax = fig.add_subplot(111)
    for i in run_protocol_list:
        ax.plot(range(2), lab.randn(2), style_list[i], label=i, linewidth=1, markerfacecolor='none', markersize=5)

    figlegend.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], 'center', ncol=7, fontsize=6)
    fig.show()
    figlegend.show()
    figlegend.savefig('draw/legend.eps', bbox_inches='tight')


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])

    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}

    if not os.path.exists("draw"):
        os.makedirs("draw")

    # draw the legend for all 4 figures
    draw_legend()

    # read the average empirical MSE in saved CSV result
    x_syn_d, emp_MSE_syn_d = read_result("results/synthetic_d_range_mse.csv")
    x_syn_epsilon, emp_MSE_syn_epsilon = read_result("results/synthetic_epsilon_range_mse.csv")
    x_taxi_d, emp_MSE_taxi_d = read_result("results/taxi_d_range_mse.csv")
    x_taxi_epsilon, emp_MSE_taxi_epsilon = read_result("results/taxi_epsilon_range_mse.csv")

    # plot figures
    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(x_syn_d)
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_syn_d, emp_MSE_syn_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/emp_MSE_syn_d.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Privacy budget ε")
    plt.yscale("log")
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_syn_epsilon, emp_MSE_syn_epsilon[name], style_list[name], linewidth=1, markerfacecolor='none',
                 markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/emp_MSE_syn_epsilon.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Domain size d")
    plt.yscale("log")
    plt.xscale("log", base=2)
    plt.xticks(x_taxi_d)
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_taxi_d, emp_MSE_taxi_d[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/emp_MSE_taxi_d.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE")
    plt.xlabel("Privacy budget ε")
    plt.yscale("log")
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(x_taxi_epsilon, emp_MSE_taxi_epsilon[name], style_list[name], linewidth=1, markerfacecolor='none',
                 markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/emp_MSE_taxi_epsilon.eps")
    plt.show()

    print("end")
