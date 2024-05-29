
import numpy as np

import commons
from commons import METHOD_TO_NICE_STR

# parser = ArgumentParser()
# # parser.add_argument('--dataset', type=str, default='energy')
# parser.add_argument('--diagnostics', type=str, choices=['y', 'n'], default='y')
# # parser.add_argument('--max_min', type=str, choices=['max', 'min'])
# parser.add_argument('--method', type=str, choices=['map', 'sghmc', 'abs_fix', 'layer_fix', 'row_fix'], default='map')
# parser.add_argument('--n_fixlay', type=int, default=1)
# parser.add_argument('--n_hidden', type=int, default=1)
# parser.add_argument('--n_units', type=int, default=100)
# parser.add_argument('--prelearn', type=str, choices=['opt', 'cv'], default='cv')
# args = parser.parse_args()

# scp -r andrade@ada:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@student_mini:~/ResearchProjects/revised_sghmc/results/. results/.
# scp -r danielandrade@mini3:~/ResearchProjects/revised_sghmc/results/. results/.


def get_result_str(results, all_fields):

    output_one_method = ""

    for result_type in all_fields:
        mean = np.mean(results[result_type])
        std = np.std(results[result_type])
        output_one_method += " & "
        output_one_method += f" {round(mean, 3)}({round(std, 3)}) "

    output_one_method += "\\\\" + "\n"
    return output_one_method


n_hidden = 2

REGRESSION_TASK = True

SHOW_GAP_DATA = True

if REGRESSION_TASK:
    ALL_FIELDS = ['train_rmse', 'train_nll', 'test_rmse', 'test_nll']
    
    if SHOW_GAP_DATA:
        # ALL_DATASETS = ["yacht_gap", "energy_gap", "concrete", "california", "protein"] 
        # ALL_DATASETS = ["boston_gap", "yacht_gap", "concrete_gap"] # "energy_gap",  "california", "protein"] 
        # ALL_DATASETS = ["yacht_gap", "energy_gap"]
        # "boston_gap", 
        ALL_DATASETS = ["boston_gap"]
    else:
        # ALL_DATASETS = ["boston", "yacht", "concrete", "energy", "california", "protein"]
        ALL_DATASETS = ["yacht"]
    
else:
    if SHOW_GAP_DATA:
        ALL_DATASETS = ["htru2_gap"] # "eeg", "letter", "miniboo"]
    else:
        ALL_DATASETS = ["htru2", "eeg", "letter", "miniboo"]
    
    ALL_FIELDS = ['train_acc', 'train_nll', 'test_acc', 'test_nll']


if n_hidden == 2:
    # ALL_METHODS = ["map_new", "sghmc", "abs_fix", "row_fix", "sharma_fix", "layer_fix|1", "layer_fix|2"]
    ALL_METHODS = ["map_new", "sghmc", "layer_fix|1", "layer_fix|2"]
elif n_hidden == 3:
    ALL_METHODS = ["map_new", "sghmc", "layer_fix|1", "layer_fix|2", "layer_fix|3"]
else:
    # ALL_METHODS = ["map_new", "sghmc", "abs_fix", "row_fix", "sharma_fix", "layer_fix"]
    ALL_METHODS = ["map_new", "sghmc", "sharma_fix","layer_fix"]
    # ALL_METHODS = ["map_new", "sghmc", "row_fix"]



FIELD_TO_NICE_STR = {}
FIELD_TO_NICE_STR["train_rmse"] = "RMSE (train)"
FIELD_TO_NICE_STR["train_nll"] = "NLL (train)"
FIELD_TO_NICE_STR["test_rmse"] = "RMSE (test)"
FIELD_TO_NICE_STR["test_nll"] = "NLL (test)"
FIELD_TO_NICE_STR["train_acc"] = "Accuracy (train)"
FIELD_TO_NICE_STR["test_acc"] = "Accuracy (test)"

FIELD_TO_NICE_STR["mean_ess_raw"] = "Mean ESS"
FIELD_TO_NICE_STR["min_ess_raw"] = "Min ESS"
FIELD_TO_NICE_STR["mean_rhat_raw"] = '''Mean $\hat{R}$'''
FIELD_TO_NICE_STR["max_rhat_raw"] = '''Max $\hat{R}$'''

# "raw" means evaluated at the predicted values of X_test (otherwise evaluated on weights of the last layer)
SAMPLING_ANALYSIS_FIELDS = ['mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw']

output_runtimes = "\\bfseries Method" + " & " + "\\bfseries Runtime (in seconds)" + " \\\\" + "\n"
output_runtimes +=  '''\midrule ''' + "\n"

output_analysis = "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in SAMPLING_ANALYSIS_FIELDS]) + " \\\\" + "\n"
output_analysis +=  '''\midrule ''' + "\n"

output_text = "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + FIELD_TO_NICE_STR[f] for f in ALL_FIELDS]) + " \\\\" + "\n"
output_text +=  '''\midrule ''' + "\n"

for dataset in ALL_DATASETS:
    
    output_runtimes += '''\multicolumn{2}{c}{  ''' + dataset + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    output_analysis += '''\multicolumn{5}{c}{  ''' + dataset + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
    output_text += '''\multicolumn{5}{c}{  ''' + dataset + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"

    
    for method_with_info in ALL_METHODS: 
        
        if n_hidden >= 2 and method_with_info.startswith("layer_fix") and "|" in method_with_info:
            method = method_with_info.split("|")[0]
            num_fix_layer = method_with_info.split("|")[1]
        else:
            method = method_with_info
            num_fix_layer = 1

        
        output_one_method = METHOD_TO_NICE_STR[method_with_info]

        results, runtimes, sampling_analysis = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)
        output_runtimes += METHOD_TO_NICE_STR[method_with_info]
        output_runtimes += f" {round(np.mean(runtimes), 3)}({round(np.std(runtimes), 3)}) " + "\\\\" + "\n" 
        
        output_text += METHOD_TO_NICE_STR[method_with_info] + get_result_str(results, ALL_FIELDS)

        if method_with_info != "map_new":
            output_analysis += METHOD_TO_NICE_STR[method_with_info] + get_result_str(sampling_analysis, SAMPLING_ANALYSIS_FIELDS)
        
    
print("-----------")
print(output_runtimes)

print("-----------")
print(output_analysis)

print("-----------")
print(output_text)


