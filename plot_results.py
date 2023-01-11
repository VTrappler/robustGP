import numpy as np
import matplotlib.pyplot as plt
import os

log_folder = "/home/victor/robustGP/logs/"


def get_design_logs(filename):
    design = np.genfromtxt(log_folder + f"{filename}_design.txt")
    logs = np.genfromtxt(log_folder + f"{filename}_log.txt", delimiter=",")
    return design, logs


def plots_logs(logs, axs, **kwargs):
    idx = ~np.isnan(logs[:, 1])
    axs[0].plot(logs[idx, 0], logs[idx, 1], **kwargs)
    axs[0].set_yscale("log")
    axs[1].plot(logs[idx, 0], logs[idx, 2], **kwargs)
    axs[1].set_yscale("log")


def get_experiments_files(exp_name):
    files = os.listdir(log_folder)
    exp_files = sorted(
        list(
            set(
                [
                    st.replace("_log.txt", "").replace("_design.txt", "")
                    for st in files
                    if st.startswith(exp_name)
                ]
            )
        )
    )
    return exp_files


exp_names = ["MC", "maxvar", "aIMSE"]
colors = {"MC": "k", "maxvar": "b", "aIMSE": "r"}

while True:
    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title("IMSE")
    axs[1].set_title("IMSE Delta")
    for exp in exp_names:
        exp_files = get_experiments_files(exp)
        for i, mc in enumerate(exp_files):
            if i == 0:
                label = exp
            else:
                label = None
            design, logs = get_design_logs(mc)
            plots_logs(logs, axs, color=colors[exp], alpha=0.5, label=label)
        print(f"{exp}, {i+1} replications")
        plt.legend()
    plt.show(block=False)
    plt.pause(60)
    plt.close()
