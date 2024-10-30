
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import commons
from commons import METHOD_TO_NICE_STR

def get_all_train_test_nll(n_hidden, regression_task, show_gap_data):
    ALL_DATASETS, _ = commons.getDataSetsAndFields(regression_task, show_gap_data)
    ALL_METHODS = commons.getAllMethods(n_hidden)

    all_results_train = np.zeros((len(ALL_METHODS), len(ALL_DATASETS)))
    all_results_test = np.zeros((len(ALL_METHODS), len(ALL_DATASETS)))

    for i, method_with_info in enumerate(ALL_METHODS):
        
        if method_with_info == 'layer_fix|1' or method_with_info == 'layer_fix':
            num_fix_layer = 1
            method = "layer_fix"
        elif method_with_info == 'layer_fix|2':
            num_fix_layer = 2
            method = "layer_fix"
        else:
            num_fix_layer = None
            method = method_with_info
        
        for j, dataset in enumerate(ALL_DATASETS):
            
            results, _, _ = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)        
            all_results_train[i, j]  = np.mean(results["train_nll"])
            all_results_test[i, j]  = np.mean(results["test_nll"])

    return all_results_train, all_results_test


def get_relative_nll(all_results):

    scaled_results = np.zeros_like(all_results)

    for dataset_id in range(all_results.shape[1]):
        min_value = np.min(all_results[:,dataset_id])
        max_value = np.max(all_results[:,dataset_id])
        scaled_results[:, dataset_id] = (all_results[:, dataset_id] - min_value) / (max_value - min_value)

    return scaled_results

def get_method_id(n_hidden, method_name):
    ALL_METHODS = commons.getAllMethods(n_hidden)
    return ALL_METHODS.index(method_name)


# method_name = "layer_fix|2"
# method_name = "row_fix"
# method_name = "sghmc"
# method_name = "map_new"
n_hidden = 2

all_train_nll = []
all_test_nll = []

for regression_task in [True, False]:
    for show_gap_data in [True, False]:
        all_results_train, all_results_test = get_all_train_test_nll(n_hidden, regression_task, show_gap_data)

        all_results_train = get_relative_nll(all_results_train)
        all_train_nll.append(all_results_train) # [method_id, :])

        all_results_test = get_relative_nll(all_results_test)
        all_test_nll.append(all_results_test) #[method_id, :])

# method_id = get_method_id(n_hidden, method_name)
        
all_train_nll = np.hstack(all_train_nll)
all_test_nll = np.hstack(all_test_nll)

print("****************************")

out = ""
for i, method in enumerate(commons.getAllMethods(n_hidden)):
    both_worst = (all_train_nll[i, :] == 1) & (all_test_nll[i, :] == 1)
    out += METHOD_TO_NICE_STR[method] + " & " + str(np.sum(both_worst)) + "\\\\" + "\n" 

print(out)

print("number of experiments (datasets) = ", all_test_nll.shape[1])

# print("all_train_nll = ", all_train_nll.shape)
# print("all_test_nll = ", all_test_nll.shape)

# X_LABEL = "train nll"
# Y_LABEL = "test nll"

# fig = plt.figure(figsize=(10, 5))
# fig.suptitle("dd")
# df = pd.DataFrame({X_LABEL: all_train_nll, Y_LABEL: all_test_nll})
# sns.regplot(x=X_LABEL, y=Y_LABEL, data=df, fit_reg=True, ci=95, n_boot=1000)


# plt.tight_layout()
# # plt.savefig(f'./plots/DDD.pdf')
# plt.show()