import pandas as pd
import numpy as np
from replicate_results_utils import get_test_accs

results = pd.read_csv("replicate_results.csv")

# get results of table 1
plot_values = {}
filetype = "test_last_log"
datasets = [
    "RTE",
    "MRPC",
    "STS-B",
    "CoLA",
]

for dataset in datasets:
    for method in ["standard_debiased", "reinit_debiased"]:
        test_accuracies = get_test_accs(results, dataset, method, filetype)

        if dataset not in plot_values:
            plot_values[dataset] = {}
        plot_values[dataset][method] = {
            "x": [i for i in range(1, 51)],
            "y": [],
            "std": [],
        }

        if method == "reinit_debiased":
            method_str = "Re-init"
        if method == "standard_debiased":
            method_str = "Standard"
        acc_avg = np.mean(test_accuracies[:20]) * 100
        acc_std = np.std(test_accuracies[:20]) * 100
        print("Dataset: " + str(dataset) + " 3 Epochs, " + method_str + ", " + str(acc_avg) + " +- " + str(acc_std))
