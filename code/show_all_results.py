
import numpy as np

import commons
from commons import METHOD_TO_NICE_STR
from commons import FIELD_TO_NICE_STR

# scp -r andrade@dgx2.hu-sm-ai.hiroshima-u.ac.jp:/raid/andrade/ResearchProjects/revised_sghmc/results/. results/.
# scp -r andrade@ada:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@student_mini:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@mini3:~/ResearchProjects/revised_sghmc/results/. results/.



    
def get_all_info(dataset, n_hidden, method_with_info):
    if n_hidden >= 2 and method_with_info.startswith("layer_fix") and "|" in method_with_info:
        method = method_with_info.split("|")[0]
        num_fix_layer = method_with_info.split("|")[1]
    else:
        method = method_with_info
        num_fix_layer = 1

    results, all_runtimes, analysis = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)
    all_runtimes = adjust_runtimes(dataset, n_hidden, num_fix_layer, method_with_info, all_runtimes)

    return results, all_runtimes, analysis


def adjust_runtimes(dataset, n_hidden, num_fix_layer, method_with_info, runtimes):
    if method_with_info == "map_new" or method_with_info == "sghmc":
        return runtimes
    else:
        _, runtimes_map, _ = commons.fully_analyze(dataset, "map_new", n_hidden, num_fix_layer)
        return runtimes + runtimes_map




def get_nice_runtime_str(runtimes_in_seconds):
    if np.all(np.isnan(runtimes_in_seconds)):
        return " - "
    else:
        runtimes_in_minutes = runtimes_in_seconds / 60.0
        return f" {round(np.mean(runtimes_in_minutes), 1)} ({round(np.std(runtimes_in_minutes), 1)}) "

    

# SHOW_ANALYSIS = "sampling_analysis"
# SHOW_ANALYSIS = "under_fitting_analysis"
SHOW_ANALYSIS = None # shows all results before the analysis sections

n_hidden = 2
REGRESSION_TASK = False
SHOW_GAP_DATA = True

if SHOW_ANALYSIS == "under_fitting_analysis":
    ALL_DATASETS, ALL_FIELDS = commons.getDataSetsAndFields(REGRESSION_TASK, SHOW_GAP_DATA)
    ALL_DATASETS = ["yacht", "energy"]
    ALL_METHODS =["sghmc", "sghmc_ext"]
else:
    ALL_DATASETS, ALL_FIELDS = commons.getDataSetsAndFields(REGRESSION_TASK, SHOW_GAP_DATA)
    ALL_METHODS = commons.getAllMethods(n_hidden)


PREDICION_AND_ANALYSIS_FIELDS = ALL_FIELDS + commons.SAMPLING_ANALYSIS_FIELDS

print("PREDICION_AND_ANALYSIS_FIELDS = ", PREDICION_AND_ANALYSIS_FIELDS)

FIELD_TYPE = {}
FIELD_TYPE["train_rmse"] = -1
FIELD_TYPE["train_nll"] = -1
FIELD_TYPE["test_rmse"] = -1
FIELD_TYPE["test_nll"] = -1
FIELD_TYPE["train_acc"] = 1
FIELD_TYPE["test_acc"] = 1

FIELD_TYPE["mean_ess_raw"] = 1
FIELD_TYPE["min_ess_raw"] = 1
FIELD_TYPE["mean_rhat_raw"] = -1
FIELD_TYPE["max_rhat_raw"] = -1


# "raw" means evaluated at the predicted values of X_test (otherwise evaluated on weights of the last layer)
SAMPLING_ANALYSIS_FIELDS = ['mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw']

output_runtimes =  '''\midrule ''' + "\n"
output_runtimes += "\\bfseries Method" + " & " + "\\bfseries 1 hidden layer " + " & " + "\\bfseries 2 hidden layers " + " \\\\" + "\n"


output_text =  '''\midrule ''' + "\n"

if SHOW_ANALYSIS == "sampling_analysis":
    assert(ALL_METHODS[0] == "map_new")
    ALL_METHODS.pop(0)
    output_text += "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in SAMPLING_ANALYSIS_FIELDS]) + " \\\\" + "\n"
else:
    output_text += "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in ALL_FIELDS]) + " \\\\" + "\n"


for dataset in ALL_DATASETS:
    
    output_runtimes +=  '''\midrule ''' + "\n"
    output_text +=  '''\midrule ''' + "\n"
    
    output_runtimes += '''\multicolumn{3}{c}{  ''' + commons.getNiceDataSetStr(dataset) + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    output_text += '''\multicolumn{5}{c}{  ''' + commons.getNiceDataSetStr(dataset) + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    
    field_to_results = {}
    for field in PREDICION_AND_ANALYSIS_FIELDS:
        field_to_results[field] = np.zeros(len(ALL_METHODS))

    # get best result of each column
    for method_id, method_with_info in enumerate(ALL_METHODS): 
        results, _, sampling_analysis =  get_all_info(dataset, n_hidden, method_with_info)
        results.update(sampling_analysis)
        # print("results = ", results.keys())
        # print("sampling_analysis = ", sampling_analysis)
        # assert(False)
        for field in PREDICION_AND_ANALYSIS_FIELDS:
            field_to_results[field][method_id] = np.mean(results[field])
    
    field_to_best_method_id = {}
    for field in PREDICION_AND_ANALYSIS_FIELDS:
        rounded_values = np.round(FIELD_TYPE[field] * field_to_results[field], decimals=3)
        max_value = np.max(rounded_values)
        max_ids = set(np.where(rounded_values == max_value)[0])
        field_to_best_method_id[field] = max_ids
    
    # print("field_to_best_method_id = ", field_to_best_method_id)
    
    # print out nicely
    for method_id, method_with_info in enumerate(ALL_METHODS): 

        print("method_with_info = ", method_with_info)
        
        if SHOW_ANALYSIS == "runtime":
            if method_with_info == "layer_fix|2":
                # last layer BNN
                _, runtimes_1, _ =  get_all_info(dataset, 1, "layer_fix")
            elif method_with_info == "layer_fix|1":
                runtimes_1 = np.nan
            else:
                _, runtimes_1, _ =  get_all_info(dataset, 1, method_with_info)

            _, runtimes_2, _ =  get_all_info(dataset, 2, method_with_info)

            output_runtimes += METHOD_TO_NICE_STR[method_with_info] + " & "
            output_runtimes += get_nice_runtime_str(runtimes_1) + " & " + get_nice_runtime_str(runtimes_2) + "\\\\" + "\n" 
        else:
            results, _, sampling_analysis =  get_all_info(dataset, n_hidden, method_with_info)
            
            if SHOW_ANALYSIS == "sampling_analysis":
                output_text += METHOD_TO_NICE_STR[method_with_info] + commons.get_result_str(sampling_analysis, SAMPLING_ANALYSIS_FIELDS, field_to_best_method_id, method_id)
            else:
                output_text += METHOD_TO_NICE_STR[method_with_info] + commons.get_result_str(results, ALL_FIELDS, field_to_best_method_id, method_id)


if SHOW_ANALYSIS == "runtime":
    print("-----------")
    print("RUNTIMES (in minutes):")
    print("-----------")
    print(output_runtimes)
    print("-----------")
else:
    print(output_text)