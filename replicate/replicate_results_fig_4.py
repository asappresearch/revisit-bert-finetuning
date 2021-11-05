import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv("replicate_results.csv")

# load data for plots of test performance after debiasing
plot_values = {}
filetype = "test_last_log"
datasets = [
    "RTE",
    "MRPC",
    "STS-B",
    "CoLA",
]
test_metrics = {
    "RTE": (["test_acc"], 0),
    "MRPC": (["test_acc", "test_f1", "test_acc_and_f1"], 1),
    "STS-B": (["test_pearson", "test_spearmanr", "test_corr"], 1),
    "CoLA": (["test_mcc"], 0),
}

val_metrics = {
    "RTE": (["val_acc", "val_loss", "best_val_acc"], 0),
    "MRPC": (["val_acc", "val_f1", "val_acc_and_f1", "val_loss", "best_val_acc"], 1),
    "STS-B": (
        ["val_pearson", "val_spearmanr", "val_corr", "val_loss", "best_val_spearmanr"],
        1,
    ),
    "CoLA": (["val_mcc", "val_loss", "best_val_mcc"], 0),
}
for dataset in datasets:
    for method in ["not_debiased", "standard_debiased"]:
        test_metric = test_metrics[dataset]
        test_metric = test_metric[0][test_metric[1]]
        test_df = results[
            (results["dataset"] == dataset)
            & (results["method"] == method)
            & (results["filetype"] == filetype)
        ]
        test_df.sort_values("seed", inplace=True)
        test_accuracies = test_df[test_metric].to_numpy()

        val_metric = val_metrics[dataset]
        val_metric = val_metric[0][val_metric[1]]
        val_df = results[
            (results["dataset"] == dataset)
            & (results["method"] == method)
            & (results["filetype"] == "raw_log")
        ]
        val_df = val_df[val_df["i"] == val_df["i"].max()]
        val_df.sort_values("seed", inplace=True)
        val_accuracies = val_df[val_metric].to_numpy()

        if dataset not in plot_values:
            plot_values[dataset] = {}
        plot_values[dataset][method] = {
            "x": [i for i in range(1, 51)],
            "y": [],
            "std": [],
        }
        for i in plot_values[dataset][method]["x"]:
            simulations = []
            for k in range(1000):
                selection = np.random.randint(low=0, high=50, size=i)
                max_val_acc = 0  # val_accuracies[selection[0]]
                test_acc = 0  # test_accuracies[selection[0]]
                for j in selection:
                    if (
                        val_accuracies[j] > max_val_acc
                    ):  # or (max_val_acc != max_val_acc):
                        max_val_acc = val_accuracies[j]
                        test_acc = test_accuracies[j]
                if test_acc != test_acc:
                    test_acc = 0
                simulations.append(test_acc)

            plot_values[dataset][method]["y"].append(np.mean(simulations))
            plot_values[dataset][method]["std"].append(np.std(simulations))

# generate plot
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

for dataset, ax in zip(datasets, axs):
    test_metric = test_metrics[dataset]
    test_metric = test_metric[0][test_metric[1]]
    ax.set_title(dataset)
    ax.set_ylabel(test_metric)
    ax.set_xlabel("# of Random Trials")

    if dataset == "RTE":
        ylimits = [0.5, 0.75]
    elif dataset == "MRPC":
        ylimits = [0.8, 0.95]
    elif dataset == "STS-B":
        ylimits = [0.85, 0.9]
    elif dataset == "CoLA":
        ylimits = [0.4, 0.7]
    ax.set_ylim(ylimits)

    for method in ["not_debiased", "standard_debiased"]:
        if method == "not_debiased":
            label = "No Correction"
            color = "indianred"
            fillcolor = "lightcoral"
        elif method == "standard_debiased":
            label = "Correction"
            color = "cornflowerblue"
            fillcolor = "lightsteelblue"

        x = np.array(plot_values[dataset][method]["x"])
        y = np.array(plot_values[dataset][method]["y"])
        std = np.array(plot_values[dataset][method]["std"])
        ax.plot(x, y, label=label, color=color)
        ax.fill_between(x, y - std, y + std, color=fillcolor, alpha=0.5)
    ax.legend()
plt.savefig("figure_4_replicate.png")
