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


def get_diagnostic(filename):
    diag = np.genfromtxt(log_folder + f"{filename}_diagnostic.txt", delimiter=",")
    return diag


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


def plots_diags(diag, axs, **kwargs):
    axs[0].plot(diag[:, 0], diag[:, 1], **kwargs)
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"norm diff $\Gamma$")
    axs[0].set_xlabel("iteration")

    axs[1].plot(diag[:, 0], diag[:, 2], **kwargs)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"norm diff $J^*$")
    axs[1].set_xlabel("iteration")

    axs[2].plot(diag[:, 0], diag[:, 3], **kwargs)
    axs[2].set_yscale("log")
    axs[2].set_ylabel(r"norm diff $\theta^*$")
    axs[2].set_xlabel("iteration")


def get_experiments_files(exp_name):
    files = os.listdir(log_folder)
    exp_files = sorted(
        list(
            set(
                [
                    st.replace("_log.txt", "")
                    .replace("_design.txt", "")
                    .replace("_diagnostic.txt", "")
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

# while True:
fig, axs = plt.subplots(ncols=2, figsize=(8, 6))
axs[0].set_title(r"$\text{IMSE}_Z$")
axs[1].set_title(r"$\text{IMSE}_\Delta$")
logs_dict = {}

for exp in exp_names:
    exp_files = get_experiments_files(exp)
    print(exp_files)
    logs_dict[exp] = []
    for i, mc in enumerate(exp_files):
        if i == 0:
            label = exp
        else:
            label = None
        try:
            design, logs = get_design_logs(mc)
            logs_dict[exp].append(logs)
            plots_logs(logs, axs, color=colors[exp], alpha=0.1, label=label)
        except FileNotFoundError:
            pass
    print(f"{exp}, {i+1} replications")
    # plots_logs(
    #     np.array(logs_dict[exp]).mean(0), axs, color=colors[exp], alpha=1, label=label
    # )
    plt.legend()
ymin1, ymax1 = axs[0].get_ylim()
ymin2, ymax2 = axs[1].get_ylim()
for ax in axs:
    ax.set_ylim([min([ymin1, ymin2]), max([ymax1, ymax2])])
plt.tight_layout()
plt.savefig(f"logs_fig.png")
plt.close()


fig, axs = plt.subplots(ncols=3, figsize=(10, 6))
axs[0].set_title(r"error in $\Gamma_\alpha$")
axs[1].set_title(r"$\|J^* - m^*\|^2$")
axs[2].set_title(r"$\|\theta^* - \theta_Z^*\|^2$")
diags_dict = {}
for exp in exp_names:
    exp_files = get_experiments_files(exp)
    diags_dict[exp] = []
    for i, exp_name in enumerate(exp_files):
        if i == 0:
            label = exp
        else:
            label = None
        try:
            diag = get_diagnostic(exp_name)
            diags_dict[exp].append(diag)
            plots_diags(diag, axs, color=colors[exp], alpha=0.1, label=label)
        except FileNotFoundError:
            pass
    print(f"{exp}, {i+1} replications")
    plots_diags(
        np.array(diags_dict[exp]).mean(0), axs, color=colors[exp], alpha=1, label=label
    )
    plt.legend()

ymin1, ymax1 = axs[0].get_ylim()
ymin2, ymax2 = axs[1].get_ylim()
for ax in axs:
    ax.set_ylim([min([ymin1, ymin2]), max([ymax1, ymax2])])
plt.tight_layout()
plt.savefig(f"diags_fig.png")
plt.close()

# plt.show()
# plt.pause(60)
# plt.close()
