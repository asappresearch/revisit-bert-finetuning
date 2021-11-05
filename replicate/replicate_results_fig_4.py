import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from replicate_results_utils import get_test_accs, get_test_metrics, get_val_accs, debias

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

for dataset in datasets:
    for method in ["not_debiased", "standard_debiased"]:
        test_accuracies = get_test_accs(results, dataset, method, filetype)
        val_accuracies = get_val_accs(results, dataset, method, filetype)

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
                simulations.append(debias(val_accuracies, test_accuracies, size=i))
            plot_values[dataset][method]["y"].append(np.mean(simulations))
            plot_values[dataset][method]["std"].append(np.std(simulations))

# generate plot
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

test_metrics = get_test_metrics()

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
