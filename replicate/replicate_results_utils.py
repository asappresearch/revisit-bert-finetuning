import numpy as np


def get_test_metrics():
    test_metrics = {
        "RTE": (["test_acc"], 0),
        "MRPC": (["test_acc", "test_f1", "test_acc_and_f1"], 1),
        "STS-B": (["test_pearson", "test_spearmanr", "test_corr"], 1),
        "CoLA": (["test_mcc"], 0),
    }
    return test_metrics


def get_test_accs(results, dataset, method, filetype):
    test_metrics = get_test_metrics()
    test_metric = test_metrics[dataset]
    test_metric = test_metric[0][test_metric[1]]
    test_df = results[(results["dataset"] == dataset) & (results["method"] == method) & (results["filetype"] == filetype)]
    test_df.sort_values("seed", inplace=True)
    test_accuracies = test_df[test_metric].to_numpy()
    return test_accuracies


def get_val_metrics():
    val_metrics = {
        "RTE": (["val_acc", "val_loss", "best_val_acc"], 0),
        "MRPC": (["val_acc", "val_f1", "val_acc_and_f1", "val_loss", "best_val_acc"], 1),
        "STS-B": (["val_pearson", "val_spearmanr", "val_corr", "val_loss", "best_val_spearmanr"], 1),
        "CoLA": (["val_mcc", "val_loss", "best_val_mcc"], 0),
    }
    return val_metrics


def get_val_accs(results, dataset, method, filetype):
    val_metrics = get_val_metrics()
    val_metric = val_metrics[dataset]
    val_metric = val_metric[0][val_metric[1]]
    val_df = results[(results["dataset"] == dataset) & (results["method"] == method) & (results["filetype"] == "raw_log")]
    val_df = val_df[val_df["i"] == val_df["i"].max()]
    val_df.sort_values("seed", inplace=True)
    val_accuracies = val_df[val_metric].to_numpy()

    return val_accuracies


def debias(val_accuracies, test_accuracies, size, high=50):
    selection = np.random.randint(low=0, high=high, size=size)
    max_val_acc = 0
    test_acc = 0
    for j in selection:
        if (val_accuracies[j] > max_val_acc):
            max_val_acc = val_accuracies[j]
            test_acc = test_accuracies[j]
    if test_acc != test_acc:
        test_acc = 0
    return test_acc
