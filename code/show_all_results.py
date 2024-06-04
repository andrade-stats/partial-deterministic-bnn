
import numpy as np

import commons
from commons import METHOD_TO_NICE_STR
from commons import FIELD_TO_NICE_STR

# scp -r andrade@dgx2.hu-sm-ai.hiroshima-u.ac.jp:/raid/andrade/ResearchProjects/revised_sghmc/results/. results/.
# scp -r andrade@ada:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@student_mini:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@mini3:~/ResearchProjects/revised_sghmc/results/. results/.


def getNiceDataSetStr(dataset):
    return dataset.replace("_", "-")
    
def get_all_info(dataset, n_hidden, method_with_info):
    if n_hidden >= 2 and method_with_info.startswith("layer_fix") and "|" in method_with_info:
        method = method_with_info.split("|")[0]
        num_fix_layer = method_with_info.split("|")[1]
    else:
        method = method_with_info
        num_fix_layer = 1

    return commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)


def get_result_str(results, all_fields, field_to_best_method_id = None, method_id = None):

    output_one_method = ""

    for result_type in all_fields:
        mean = np.mean(results[result_type])
        std = np.std(results[result_type])
        output_one_method += " & "

        if (field_to_best_method_id is not None) and (method_id == field_to_best_method_id[result_type]):
            output_one_method += " \\textbf{" + str(round(mean, 3))  + "}" + f"({round(std, 3)}) "
        else:
            output_one_method += f" {round(mean, 3)}({round(std, 3)}) "

    output_one_method += "\\\\" + "\n"
    return output_one_method


SHOW_ANALYSIS = True

n_hidden = 2
REGRESSION_TASK = False
SHOW_GAP_DATA = True

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
output_runtimes += "\\bfseries Method" + " & " + "\\bfseries Runtime (in seconds)" + " \\\\" + "\n"


output_text =  '''\midrule ''' + "\n"

if SHOW_ANALYSIS:
    assert(ALL_METHODS[0] == "map_new")
    ALL_METHODS.pop(0)
    output_text += "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in SAMPLING_ANALYSIS_FIELDS]) + " \\\\" + "\n"
else:
    output_text += "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in ALL_FIELDS]) + " \\\\" + "\n"


for dataset in ALL_DATASETS:
    
    output_runtimes +=  '''\midrule ''' + "\n"
    output_text +=  '''\midrule ''' + "\n"
    
    output_runtimes += '''\multicolumn{2}{c}{  ''' + getNiceDataSetStr(dataset) + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    output_text += '''\multicolumn{5}{c}{  ''' + getNiceDataSetStr(dataset) + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    
    field_to_results = {}
    for field in PREDICION_AND_ANALYSIS_FIELDS:
        field_to_results[field] = np.zeros(len(ALL_METHODS))

    # get best result of each column
    for method_id, method_with_info in enumerate(ALL_METHODS): 
        results, runtimes, sampling_analysis =  get_all_info(dataset, n_hidden, method_with_info)
        results.update(sampling_analysis)
        # print("results = ", results.keys())
        # print("sampling_analysis = ", sampling_analysis)
        # assert(False)
        for field in PREDICION_AND_ANALYSIS_FIELDS:
            field_to_results[field][method_id] = np.mean(results[field])
    
    field_to_best_method_id = {}
    for field in PREDICION_AND_ANALYSIS_FIELDS:
        field_to_best_method_id[field] = np.argmax(FIELD_TYPE[field] * field_to_results[field])
    
    # print("field_to_best_method_id = ", field_to_best_method_id)
    
    # print out nicely
    for method_id, method_with_info in enumerate(ALL_METHODS): 
        results, runtimes, sampling_analysis =  get_all_info(dataset, n_hidden, method_with_info)
        
        output_runtimes += METHOD_TO_NICE_STR[method_with_info]
        output_runtimes += f" {round(np.mean(runtimes), 3)}({round(np.std(runtimes), 3)}) " + "\\\\" + "\n" 
        
        if SHOW_ANALYSIS:
            output_text += METHOD_TO_NICE_STR[method_with_info] + get_result_str(sampling_analysis, SAMPLING_ANALYSIS_FIELDS, field_to_best_method_id, method_id)
        else:
            output_text += METHOD_TO_NICE_STR[method_with_info] + get_result_str(results, ALL_FIELDS, field_to_best_method_id, method_id)


if SHOW_ANALYSIS:
    print("-----------")
    print(output_runtimes)
    print("-----------")

print(output_text)


