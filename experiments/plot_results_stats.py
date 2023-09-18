import numpy as np
import matplotlib.pyplot as plt
import os


plt.style.use("seaborn-v0_8")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # 'font.sans-serif' : ['Tahoma', 'DejaVu Sans','Lucida Grande', 'Verdana'],
        "image.cmap": "viridis",
        "figure.figsize": [8, 8],
        "savefig.dpi": 200,
    }
)


log_folder = "/home/vtrappler/robustGP/logs/"


def get_design_logs(filename):
    design = np.genfromtxt(log_folder + f"{filename}_design.txt")
    logs = np.genfromtxt(log_folder + f"{filename}_log.txt", delimiter=",")
    return design, logs


def plots_logs(logs, axs, **kwargs):
    idx = ~np.isnan(logs[:, 1])
    axs[0].plot(logs[idx, 0], logs[idx, 1], **kwargs)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("IMSE")
    axs[0].set_xlabel("iteration")

    axs[1].plot(logs[idx, 0], logs[idx, 2], **kwargs)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("IMSE")
    axs[1].set_xlabel("iteration")


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


exp_names = [
    "MC",
    "maxvar",
    "aIMSE",
]
colors = {
    "MC": "k",
    "maxvar": "b",
    "aIMSE": "r",
}

def pad_ragged(list_of_arr):
    max_len = 0
    for arr in list_of_arr:
        if len(arr) > max_len:
            max_len = len(arr)
    array = np.empty((len(list_of_arr), max_len))
    for i, arr in enumerate(list_of_arr):
        array[i, :] = np.pad(
            arr, (0, max_len - len(arr)), "constant", constant_values=(np.nan)
        )
    return array


results = {}

for exp in exp_names:
    exp_files = get_experiments_files(exp)
    results[exp] = {}
    IMSE = []
    IMSE_D = []
    for i, mc in enumerate(exp_files):
        print(mc)
        try:
            design, logs = get_design_logs(mc)
            idx = ~np.isnan(logs[:, 1])
            IMSE.append(logs[idx, 1])
            IMSE_D.append(logs[idx, 2])
        except FileNotFoundError:
            pass

    results[exp]["imse"] = pad_ragged(IMSE)
    results[exp]["imse_d"] = pad_ragged(IMSE_D)


fig, axs = plt.subplots(ncols=2, figsize=(8, 6))
axs[0].set_title(r"$\text{IMSE}_Z$")
axs[1].set_title(r"$\text{IMSE}_\Delta$")

for exp in exp_names:
    mn, sd = np.nanmean(results[exp]["imse"], 0), np.nanstd(results[exp]["imse"], 0)
    mn_d, sd_d = np.nanmean(results[exp]["imse_d"], 0), np.nanstd(results[exp]["imse_d"], 0)
    axs[0].plot(mn, color=colors[exp])
    axs[0].plot(mn - sd, color=colors[exp], ls=":")
    axs[0].plot(mn + sd, color=colors[exp], ls=":")
    axs[1].plot(mn_d, color=colors[exp])
    axs[1].plot(mn_d - sd_d, color=colors[exp], ls=":")
    axs[1].plot(mn_d + sd_d, color=colors[exp], ls=":")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

ymin1, ymax1 = axs[0].get_ylim()
ymin2, ymax2 = axs[1].get_ylim()
for ax in axs:
    ax.set_ylim([min([ymin1, ymin2]), max([ymax1, ymax2])])
plt.tight_layout()
plt.savefig(f"logs_fig_mn.png")
plt.close()
