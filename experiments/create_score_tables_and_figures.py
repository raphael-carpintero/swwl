# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import (
    create_if_not_exists,
    load_scores_for_one_output,
    prefix_filenames,
    suffix_scores_filenames,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def create_table_scores_classification(
    datasets: list,
    kernels: list,
    hparams: list,
    config_params: dict,
    classifier: str = "svc",
    score: str = "test_acc",
):
    """
    This function loads the mean and standard deviation of the classification scores obtained for all pairs (dataset, kernel)
    and outputs two arrays gathering theses values. If some scores are missing, they are replaced by None.

    Args:
        datasets: The list of dataset names.
        kernels: The list of kernel names.
        hparams: The list of hyperparameter dicts for all kernels (must have the same size as 'kernels').
        config_params: The configuration dict.
        classifier: The name of the classifier.
        score: The metric to load ('test_acc' or 'test_mcc').

    Returns:
        means: A np.ndarray of shape (num_kernels, num_datasets) that gives the mean scores.
        stds: A np.ndarray of shape (num_kernels, num_datasets) that gives the standard deviations of scores.
    """
    means, stds = (
        np.full((len(kernels), len(datasets)), None),
        np.full((len(kernels), len(datasets)), None),
    )
    for i in range(len(kernels)):
        kernel, hparam = kernels[i], hparams[i]
        for j, dataset_name in enumerate(datasets):
            root_scores = prefix_filenames(
                config_params["results"]["scores"], dataset_name, kernel
            )
            suffix_scores = suffix_scores_filenames(
                dataset_name, classifier, kernel, hparam, seed=None
            )
            scores = load_scores_for_one_output(root_scores, suffix_scores, out=0)

            if scores is not None:
                means[i, j] = scores[f"mean_{score}"]
                stds[i, j] = scores[f"std_{score}"]
    return means, stds


def create_table_scores_regression(
    datasets: list,
    kernels: list,
    hparams: list,
    config_params: dict,
    regressor: str = "rgaspR",
    score: str = "test_rmse",
    steps_WL_iters_sqrt: dict = {},
    out_for_datasets: dict = {},
):
    """
    This function loads the mean and standard deviation of the regression scores obtained for all pairs (dataset, kernel)
    and outputs two arrays gathering theses values. If some scores are missing, they are replaced by None.

    Args:
        datasets: The list of dataset names.
        kernels: The list of kernel names.
        hparams: The list of hyperparameter dicts for all kernels (must have the same size as 'kernels').
        config_params: The configuration dict.
        regressor: The name of the regressor.
        score: The metric to load ('test_rmse' or 'test_q2').
        steps_WL_iters_sqrt: The dict associating the step of Wl iterations for each dataset (for the SWWL(sqrt(n)) kernel).
        out_for_datasets: The dict associating the number of the output to consider for each dataset.
    Returns:
        means: A np.ndarray of shape (num_kernels, num_datasets) that gives the mean scores.
        stds: A np.ndarray of shape (num_kernels, num_datasets) that gives the standard deviations of scores.
    """
    if regressor == "rgaspR":
        if score == "test_q2":
            idx_scores = 0
        elif score == "test_rmse":
            idx_scores = 1
        else:
            print("Not a valid score.")
            return None, None

    means, stds = (
        np.full((len(kernels), len(datasets)), None),
        np.full((len(kernels), len(datasets)), None),
    )
    for i in range(len(kernels)):
        kernel, hparam = kernels[i], hparams[i]
        for j, dataset_name in enumerate(datasets):
            hparam_copy = dict(hparam)
            if kernel in ["swwl", "wwl"] and hparam["step_wl"] == -1:
                hparam_copy["step_wl"] = steps_WL_iters_sqrt[dataset_name]
            if kernel in ["propag"] and dataset_name == "Rotor37_CM":
                hparam_copy["pk_w"] = [0.001]
            out = out_for_datasets[dataset_name]
            root_scores = prefix_filenames(
                config_params["results"]["scores"], dataset_name, kernel
            )
            suffix_scores = suffix_scores_filenames(
                dataset_name, regressor, kernel, hparam_copy, seed=None
            )
            # Depending on the regression (made in python or R), the format of the score files are different.
            if regressor == "rgaspR":
                filename_scores = os.path.join(
                    root_scores, "scores" + suffix_scores + f"_out{out}.npy"
                )
                if os.path.exists(filename_scores):
                    with open(filename_scores, "rb") as f:
                        scores = np.load(f)
                    print(scores.shape)
                    means[i, j] = np.mean(scores[:, idx_scores])
                    stds[i, j] = np.std(scores[:, idx_scores])
            else:
                scores = load_scores_for_one_output(root_scores, suffix_scores, out=0)
                if scores is not None:
                    means[i, j] = scores[f"mean_{score}"]
                    stds[i, j] = scores[f"std_{score}"]
    return means, stds


def plot_score_against_P(
    dataset_name: str,
    kernel: str,
    P_list: list,
    Q_list: list,
    H: int,
    T: int,
    config_params: dict,
    regressor: str = "rgaspR",
    score: str = "test_rmse",
    out_for_datasets: dict = {},
    horizontal_offsets: list = [],
):
    """
    This function plots (and saves the figure of) the evolution of the mean score with errorbars corresponding to the standard deviation depending on the number of projections.
    Several curves corresponding to different number of quantiles can be plotted simultaneously.
    This will fail if some (P,Q) couples are missing.

    Args:
        dataset_name: The name of the dataset.
        kernel: The name of the kernel.
        P_list: The list of the number of projections (x values for the plot).
        Q_list: The list of the number of quantiles (one curve for each number of quantiles).
        H: The number of continuous WL iterations.
        T: The step of WL iterations.
        config_params: The configuration dict.
        regressor: The name of the regressor.
        score: The metric to load ('test_rmse' or 'test_q2').
        steps_WL_iters_sqrt: The dict associating the step of Wl iterations for each dataset (for the SWWL(sqrt(n)) kernel).
        horizontal_offsets: This argument can be specified for aesthetics to add some horizontal offets to the curves (must be of the same length as Q_list).
    """

    if regressor == "rgaspR":
        if score == "test_q2":
            idx_scores = 0
        elif score == "test_rmse":
            idx_scores = 1
        else:
            print("Not a valid score.")
            return None, None

    P_arr = np.array(P_list)
    color_list = ["tab:blue", "tab:red", "tab:green", "k"]
    linestyles = ["-"]  # , ":.", "--.", "-"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    lins, marks = [], []
    for i, Q in enumerate(Q_list):
        means_for_Q, stds_for_Q = [], []
        for P in P_list:
            hparam = {
                "num_wl_iterations": [H],
                "num_projections": P,
                "num_quantiles": Q,
                "step_wl": T,
            }
            out = out_for_datasets[dataset_name]
            root_scores = prefix_filenames(
                config_params["results"]["scores"], dataset_name, kernel
            )
            suffix_scores = suffix_scores_filenames(
                dataset_name, regressor, kernel, hparam, seed=None
            )
            if regressor == "rgaspR":
                filename_scores = os.path.join(
                    root_scores, "scores" + suffix_scores + f"_out{out}.npy"
                )
                if os.path.exists(filename_scores):
                    with open(filename_scores, "rb") as f:
                        scores = np.load(f)
                    means_for_Q.append(np.mean(scores[:, idx_scores]))
                    stds_for_Q.append(np.std(scores[:, idx_scores]))
                else:
                    print(f"Error: missing scores for P={P}, Q={Q}.")
                    return
            else:
                scores = load_scores_for_one_output(root_scores, suffix_scores, out=0)
                if scores is not None:
                    means_for_Q.append(scores[f"mean_{score}"])
                    stds_for_Q.append(scores[f"std_{score}"])
                else:
                    print(f"Error: missing scores for P={P}, Q={Q}.")
                    return

        ax.errorbar(
            P_arr + horizontal_offsets[i],
            means_for_Q,
            yerr=stds_for_Q,
            color=color_list[i % len(color_list)],
            fmt=linestyles[i % len(linestyles)],
            linewidth=1.5,
            capsize=2,
            ls="",
        )
        (lin,) = ax.plot(
            P_arr + horizontal_offsets[i],
            means_for_Q,
            ls="-",
            lw=2,
            color=color_list[i % len(color_list)],
        )
        (mark,) = ax.plot(
            P_arr + horizontal_offsets[i],
            means_for_Q,
            marker="o",
            markersize=12,
            alpha=0.75,
            color=color_list[i % len(color_list)],
            markeredgecolor="k",
            markeredgewidth=1,
        )
        lins.append(lin)
        marks.append(mark)
    ax.set_ylabel(r"$\mathrm{RMSE}$", fontsize=26)
    ax.set_xlabel(
        r"$\mathrm{Number}$ $\mathrm{of}$ $\mathrm{projections}$ $P$", fontsize=26
    )
    ax.set_xticks(P_list)
    ax.locator_params(nbins=10)

    ax.legend(
        [(lins[i], marks[i]) for i in range(len(Q_list))],
        [rf"$Q = {Q}$" for Q in Q_list],
        fontsize=22,
        edgecolor="k",
        framealpha=1.0,
    )
    ax.tick_params(labelsize=22)
    ax.grid(True)
    create_if_not_exists(config_params["results"]["figures"])
    fig.savefig(
        os.path.join(
            config_params["results"]["figures"],
            f"PQ_{dataset_name}_H{H}_T{T}_{regressor}_{score}.png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=1000,
    )
    fig.savefig(
        os.path.join(
            config_params["results"]["figures"],
            f"PQ_{dataset_name}_H{H}_T{T}_{regressor}_{score}.eps",
        ),
        format="eps",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config_params = yaml.safe_load(f)

    # 1- Tables of scores for classification

    datasets = ["BZR", "COX2", "PROTEINS", "ENZYMES", "Cuneiform"]
    kernels = ["swwl", "wwl", "fgw", "sgml", "propag", "ghopper"]
    hparams = [
        {
            "num_wl_iterations": [0, 1, 2, 3],
            "num_projections": 20,
            "num_quantiles": 20,
            "step_wl": 1,
        },
        {"num_wl_iterations": [0, 1, 2, 3], "step_wl": 1},
        {"fgw_alphas": [0.0, 0.25, 0.5, 0.75, 1.0]},
        {"num_wl_iterations": [1, 2, 3]},
        {"pk_w": [], "pk_tmax": []},
        {},
    ]
    score = "test_acc"
    means, stds = create_table_scores_classification(
        datasets, kernels, hparams, config_params, classifier="svc", score=score
    )

    print("************************************")
    print("Datasets:", datasets)
    print("Kernels:", kernels)
    print(f"Mean {score} for classification: ")
    print(means)
    print(f"Std {score} for classification: ")
    print(stds)

    # 2- Tables of scores for regression

    datasets = [
        "Rotor37_CM",
        "Rotor37",
        "Rotor37_CM",
        "Tensile2d",
        "Tensile2d_CM",
        "AirfRANS",
        "AirfRANS_CM",
    ]
    kernels = ["swwl", "swwl", "wwl", "propag"]
    hparams = [
        {
            "num_wl_iterations": [3],
            "num_projections": 50,
            "num_quantiles": 100,
            "step_wl": 1,
        },
        {
            "num_wl_iterations": [3],
            "num_projections": 50,
            "num_quantiles": 100,
            "step_wl": -1,
        },
        {"num_wl_iterations": [3], "step_wl": 1},
        {"pk_w": [0.01], "pk_tmax": [1]},
    ]
    steps_WL_iters_sqrt = {
        "Rotor37": 173,
        "Rotor37_CM": 30,
        "Tensile2d": 100,
        "Tensile2d_CM": 30,
        "AirfRANS": 100,
        "AirfRANS_CM": 100,
    }
    out_for_datasets = {
        "Rotor37": 2,
        "Rotor37_CM": 2,
        "Tensile2d": 0,
        "Tensile2d_CM": 0,
        "AirfRANS": 0,
        "AirfRANS_CM": 0,
    }
    score = "test_rmse"
    means, stds = create_table_scores_regression(
        datasets,
        kernels,
        hparams,
        config_params,
        regressor="rgaspR",
        score=score,
        steps_WL_iters_sqrt=steps_WL_iters_sqrt,
        out_for_datasets=out_for_datasets,
    )

    print("\n************************************")
    print("Datasets:", datasets)
    print("Kernels:", kernels)
    print(f"Mean {score} for classification: ")
    print(means)
    print(f"Std {score} for classification: ")
    print(stds)

    # 3- Figure impact of P and Q for SWWL
    P_list = [2, 5, 10, 20, 30, 40, 50]
    Q_list = [10, 100, 500, 1000]
    horizontal_offsets = [0.0, 0.0, 0.35, 0.0]
    kernel = "swwl"
    H = 3
    T = 1
    dataset_name = "Rotor37"
    plot_score_against_P(
        dataset_name,
        kernel,
        P_list,
        Q_list,
        H,
        T,
        config_params,
        regressor="rgaspR",
        score="test_rmse",
        out_for_datasets=out_for_datasets,
        horizontal_offsets=horizontal_offsets,
    )
