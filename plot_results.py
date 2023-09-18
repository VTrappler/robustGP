import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

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


def get_design_logs(filename, log_folder):
    design = np.genfromtxt(os.path.join(log_folder, f"{filename}_design.txt"))
    logs = np.genfromtxt(os.path.join(log_folder, f"{filename}_log.txt"), delimiter=",")
    return design, logs


def get_diagnostic(filename, log_folder):
    diag = np.genfromtxt(
        os.path.join(log_folder, f"{filename}_diagnostic.txt"), delimiter=","
    )
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


def get_experiments_files(exp_name, log_folder):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make diagnostics on experiments")
    parser.add_argument("--log-path", type=str, help="path of the logs")
    parser.add_argument("--fig-path", type=str, help="path of the figs")
    parser.add_argument("--name", type=str, help="prefix for figures")
    parser.add_argument(
        "-e",
        "--exp-list",
        nargs="+",
        default=[],
        help="Name of experiments amongst MC, maxvar and/or aIMSE",
    )
    parser.add_argument("--diags", action="store_true")
    parser.set_defaults(diags=False)

    args = parser.parse_args()
    log_folder = args.log_path
    fig_path = args.fig_path
    name = args.name
    exp_names = args.exp_list
    print(f"{exp_names=}")
    print(f"{log_folder=}")

    # exp_names = [
    #     "MC",
    #     "maxvar",
    #     "aIMSE",
    # ]
    colors = {
        "MC": "C0",
        "maxvar": "C1",
        "aIMSE": "C2",
        "aIMSE_Delta": "C3",
    }

    # while True:
    fig, axs = plt.subplots(ncols=2, figsize=(8, 6))
    axs[0].set_title(r"$\text{IMSE}_Z$")
    axs[1].set_title(r"$\text{IMSE}_\Delta$")
    logs_dict = {}

    for exp in exp_names:
        exp_files = get_experiments_files(exp, log_folder)
        print(exp_files)
        logs_dict[exp] = []
        for i, mc in enumerate(exp_files):
            if i == 0:
                label = exp
            else:
                label = None
            try:
                design, logs = get_design_logs(mc, log_folder)
                logs_dict[exp].append(logs)
                plots_logs(logs, axs, color=colors[exp], alpha=0.5, label=label)
            except FileNotFoundError:
                pass
        print(f"{exp}, {i+1} replications")
        print([li.shape for li in logs_dict['aIMSE_Delta']])
        plots_logs(
            np.array(logs_dict[exp]).mean(0),
            axs,
            color=colors[exp],
            alpha=1,
            linestyle=":",
            label=f"avg {exp}",
        )
        plt.legend()
    ymin1, ymax1 = axs[0].get_ylim()
    ymin2, ymax2 = axs[1].get_ylim()
    for ax in axs:
        ax.set_ylim([min([ymin1, ymin2]), max([ymax1, ymax2])])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"{name}_logs.png"))
    plt.close()

    if args.diags:
        fig, axs = plt.subplots(ncols=3, figsize=(10, 6))
        axs[0].set_title(r"error in $\Gamma_\alpha$")
        axs[1].set_title(r"$\|J^* - m^*\|^2$")
        axs[2].set_title(r"$\|\theta^* - \theta_Z^*\|^2$")
        diags_dict = {}
        for exp in exp_names:
            exp_files = get_experiments_files(exp, log_folder)
            diags_dict[exp] = []
            for i, exp_name in enumerate(exp_files):
                if i == 0:
                    label = exp
                else:
                    label = None 
                try:
                    diag = get_diagnostic(exp_name, log_folder)
                    diags_dict[exp].append(diag)
                    plots_diags(diag, axs, color=colors[exp], alpha=0.1, label=label)
                except FileNotFoundError:
                    pass
            print(f"{exp}, {i+1} replications")
            plots_diags(
                np.array(diags_dict[exp]).mean(0),
                axs,
                color=colors[exp],
                alpha=1,
                label=label,
            )
            plt.legend()

        ymin1, ymax1 = axs[0].get_ylim()
        ymin2, ymax2 = axs[1].get_ylim()
        for ax in axs:
            ax.set_ylim([min([ymin1, ymin2]), max([ymax1, ymax2])])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_path, f"{name}_diags.png"))
        plt.close()

    # plt.show()
    # plt.pause(60)
    # plt.close()
