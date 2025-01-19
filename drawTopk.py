import matplotlib.pyplot as plt
import numpy as np

# for PDF backend
plt.rcParams['pdf.fonttype'] = 42
# for PS backend
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 6

plt.rcParams['legend.fontsize'] = 6

plt.rcParams['xtick.labelsize'] = 6

plt.rcParams['ytick.labelsize'] = 6


def get_save_name(method_name, dataset_name, d, epsilon):
    save_name = method_name + "_" + dataset_name + "_" + str(d) + "_" + str(epsilon) + "_"
    return save_name


def get_k_mse_set(data_name, d, epsilon, k_range, repeat_time):
    k_mse_set = {}
    for protocol in run_protocol_list:
        k_mse_set[protocol] = np.zeros(k_range.size)
    for i in range(run_repeat_time):
        for protocol in run_protocol_list:
            save_name = get_save_name(protocol, data_name, d, epsilon)
            save_path = "results/" + protocol + "/" + save_name + str(i) + ".npz"
            npzfile = np.load(save_path)
            n = npzfile["data_list_sum"].sum()
            data_estimate = npzfile["estimate_data"] / n
            data_list_sum = npzfile["data_list_sum"] / n
            for j in range(k_range.size):
                k_mse_set[protocol][j] += get_k_mse(data_estimate, data_list_sum, k_range[j])
    for protocol in run_protocol_list:
        k_mse_set[protocol] = k_mse_set[protocol] / repeat_time
    return k_mse_set


def get_k_mse(x_est, x_theory, k):
    top_k_index = np.argpartition(x_theory, -k)[-k:]
    k_mse = get_mse(x_est[top_k_index], x_theory[top_k_index])
    return k_mse


def get_mse(x_est, x_theory):
    diff = x_est - x_theory
    diff2 = diff * diff
    return diff2.sum() / np.size(diff2)


if __name__ == '__main__':
    print("start")
    # experimental parameters
    run_d = 128
    run_epsilon = 4.0
    run_repeat_time = 100
    run_k_range = np.arange(2, 32, 2)

    run_protocol_list = np.array(["GRR", "OUE", "RUE", "OLH", "RLH", "SS", "RWS"])

    style_list = {"GRR": "C0+-", "OUE": "C1s-", "RUE": "C2^-", "OLH": "C31-", "RLH": "C42-", "SS": "C5*-",
                  "RWS": "C6o-"}

    # read each saved result of run_protocol to get the average MSE on top k values
    k_mse_set_syn = get_k_mse_set("synthetic", run_d, run_epsilon, run_k_range, run_repeat_time)
    k_mse_set_taxi = get_k_mse_set("taxi", run_d, run_epsilon, run_k_range, run_repeat_time)

    # plot figures
    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE)")
    plt.xlabel("k")
    plt.yscale("log")
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(run_k_range, k_mse_set_syn[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/topk_mse_syn.eps")
    plt.show()

    plt.figure(figsize=(3.5, 2.2))
    plt.ylabel("MSE)")
    plt.xlabel("k")
    plt.yscale("log")
    for i in range(run_protocol_list.size):
        name = run_protocol_list[i]
        plt.plot(run_k_range, k_mse_set_taxi[name], style_list[name], linewidth=1, markerfacecolor='none', markersize=5,
                 label=name)
    plt.tight_layout()
    plt.savefig("draw/topk_mse_taxi.eps")
    plt.show()

    print("end")
