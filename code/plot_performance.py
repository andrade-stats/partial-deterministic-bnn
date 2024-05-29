
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import commons
from commons import METHOD_TO_NICE_STR

n_hidden = 2
prelearn = "cv"

REGRESSION_TASK = False

CRITERIA = "test_nll"

if prelearn == 'cv':
    if REGRESSION_TASK:
        ALL_DATASETS = ['boston', 'yacht', 'concrete', 'energy', 'california', 'protein']
    else:
        ALL_DATASETS = ['htru2', 'eeg', 'letter', 'miniboo']

    if n_hidden == 1:
        ALL_METHODS = ["sghmc", "abs_fix", "row_fix", "sharma_fix", "layer_fix"]
    elif n_hidden == 2:
        ALL_METHODS = ["sghmc", "layer_fix|1", "layer_fix|2"]
    
    all_results = np.zeros((len(ALL_METHODS), len(ALL_DATASETS)))
    
    for i, method_with_info in enumerate(ALL_METHODS):
        rmse_list = []
        acc_list = []
        nll_list = []
        
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
            
            results_map, _, _ = commons.fully_analyze(dataset, "map_new", n_hidden, num_fix_layer)
            results_sampling, _, _ = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)

            mean_map = np.mean(results_map[CRITERIA])
            mean_sampling = np.mean(results_sampling[CRITERIA])

            diff = (mean_map - mean_sampling) / mean_map 
            diff *= 100

            all_results[i, j] = diff

    
    columns = [METHOD_TO_NICE_STR[m] for m in ALL_METHODS]
    
    all_results_df = pd.DataFrame(all_results.T, columns=columns)
    print(all_results_df)


plt.ylim(-10.0, 10.0)
sns.boxplot(data=all_results_df).set(xlabel='method', ylabel='average improvement rate [%]')
plt.axhline(0.0, c='r', linestyle = "--")
# plt.savefig(f'./plots/boxplot_{prelearn}_{n_hidden}h{task}.png')
plt.show()

