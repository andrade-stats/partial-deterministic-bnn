
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import commons
from commons import METHOD_TO_NICE_STR

n_hidden = 2
prelearn = "cv"

REGRESSION_TASK = False
SHOW_GAP_DATA = False

# 'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw'

# CRITERIA = "mean_ess_raw"
# CRITERIA = "mean_rhat_raw"
CRITERIA = "test_nll"
# CRITERIA = "runtime"

METHOD_TO_PARAMETER_COUNT = {}
METHOD_TO_PARAMETER_COUNT["sghmc"] = "$h^2 + hd + 3h + 1$"
METHOD_TO_PARAMETER_COUNT["abs_fix"] = "$\\frac{1}{2}h^2 + \\frac{1}{2}hd + 3h + 1$"
METHOD_TO_PARAMETER_COUNT["row_fix"] = "$\\frac{1}{2}h^2 + \\frac{1}{2}hd + 3h + 1$"
METHOD_TO_PARAMETER_COUNT["sharma_fix"] = "$\\frac{1}{2}h^2 + \\frac{1}{2}hd + 3h + 1$"
METHOD_TO_PARAMETER_COUNT["layer_fix|1"] = "$h^2 + 2h + 1$"
METHOD_TO_PARAMETER_COUNT["layer_fix|2"] = "$h + 1$"

if prelearn == 'cv':
    
    if ("ess" in CRITERIA) or ("rhat" in CRITERIA):
        datasets, _ = commons.getDataSetsAndFields(regression_task = True, gap_data = True)
        ALL_DATASETS = datasets
        datasets, _ = commons.getDataSetsAndFields(regression_task = True, gap_data = False)
        ALL_DATASETS += datasets
        datasets, _ = commons.getDataSetsAndFields(regression_task = False, gap_data = True)
        ALL_DATASETS += datasets
        datasets, _ = commons.getDataSetsAndFields(regression_task = False, gap_data = False)
        ALL_DATASETS += datasets
    else:
        ALL_DATASETS, _ = commons.getDataSetsAndFields(REGRESSION_TASK, SHOW_GAP_DATA)
    
    ALL_METHODS = commons.getAllMethods(n_hidden)

    assert(ALL_METHODS[0] == "map_new")
    ALL_METHODS.pop(0)

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
            
            if ("ess" in CRITERIA) or ("rhat" in CRITERIA):
                _, _, analysis_sampling = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)
                all_results[i, j] = np.mean(analysis_sampling[CRITERIA])

            else:
                results_map, runtimes_map, _ = commons.fully_analyze(dataset, "map_new", n_hidden, num_fix_layer)
                results_sampling, runtimes_sampling, _ = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)

                results_map["runtime"] = runtimes_map
                results_sampling["runtime"] = runtimes_sampling
                
                mean_map = np.mean(results_map[CRITERIA])
                mean_sampling = np.mean(results_sampling[CRITERIA])

                if CRITERIA == "runtime":
                    if method_with_info == "sghmc":
                        diff = (mean_sampling - mean_map) / mean_map 
                    else:
                        # take into account the time to find the map solution
                        diff = mean_sampling / mean_map 
                else:
                    diff = (mean_map - mean_sampling) / mean_map 
                
                diff *= 100

                all_results[i, j] = diff

    
    columns = [METHOD_TO_NICE_STR[m] for m in ALL_METHODS]
    
    all_results_df = pd.DataFrame(all_results.T, columns=columns)
    print(all_results_df)


if CRITERIA == "runtime":
    output_runtimes = ""
    # increase of runtime relative to map's runtime
    for i, method_with_info in enumerate(ALL_METHODS):
        # output_runtimes +=  '''\midrule ''' + "\n"
        output_runtimes += METHOD_TO_NICE_STR[method_with_info] + " & "
        output_runtimes += METHOD_TO_PARAMETER_COUNT[method_with_info] + " & "
        output_runtimes += f" {int(round(np.mean(all_results[i]), -1))}\\%"   # ({int(round(np.std(all_results[i]), 0))}) "
        output_runtimes += " \\\\" + "\n"

    print(output_runtimes)

elif ("ess" in CRITERIA) or ("rhat" in CRITERIA):
    fig = plt.figure(figsize=(10, 5))

    title = commons.FIELD_TO_NICE_STR[CRITERIA]
    
    fig.suptitle(title)
    # plt.ylim(lower_limit, upper_limit)

    sns.boxplot(data=all_results_df).set(ylabel=commons.FIELD_TO_NICE_STR[CRITERIA])  # xlabel="methods", 
    
    plt.tight_layout()
    
    plt.savefig(f'./plots/boxplot_{prelearn}_{n_hidden}h_analysis_{CRITERIA}.pdf')
    # plt.show()

else:
    fig = plt.figure(figsize=(10, 5))

    if REGRESSION_TASK:
        title = "Regression"
    else:
        title = "Classification"

    if SHOW_GAP_DATA:
        title += " (gap splits)"
    else:
        title += " (random splits)"

    if REGRESSION_TASK:
        lower_limit = -10.0
        if SHOW_GAP_DATA:
            upper_limit = 20.0
        else:
            upper_limit = 10.0
    else:
        if SHOW_GAP_DATA:
            lower_limit = -20.0
            upper_limit = 20.0
        else:
            lower_limit = -20.0
            upper_limit = 10.0

    fig.suptitle(title)
    plt.ylim(lower_limit, upper_limit)

    sns.boxplot(data=all_results_df).set(ylabel="average improvement rate [%] of " + commons.FIELD_TO_NICE_STR[CRITERIA])  # xlabel="methods", 
    plt.axhline(0.0, c='r', linestyle = "--")

    plt.tight_layout()
    
    plt.savefig(f'./plots/boxplot_{prelearn}_{n_hidden}h_REGRESSION_{REGRESSION_TASK}_SHOW_GAP_DATA_{SHOW_GAP_DATA}.pdf')
    # plt.show()

