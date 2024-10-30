
import numpy as np

import numpy as np
import pandas as pd

import commons
import os

import scipy.stats

def get_best_val_choice(dataset, id, method_with_info, n_hidden, n_units, max_min, pre_learn):
    
    method, num_fix_layer = commons.get_method_and_num_fix_layer(method_with_info, n_hidden)
    filename = commons.getFilename(id + 1, method, n_hidden, n_units, num_fix_layer, max_min)
    
    LOAD_PATH = f'./results/{method}/{dataset}/{pre_learn}'
    NEW_PATH = os.path.join(LOAD_PATH, filename)
    
    df = pd.read_csv(NEW_PATH)
    df.index = np.arange(len(df.index))
    
    all_val_nll = df.loc[:,"val_nll"].values
    best_idx = np.nanargmin(all_val_nll)

    return df, all_val_nll, best_idx

    
def get_result_combined_partial_determininstic_BNN(dataset, n_hidden, all_pb_bnn_methods):

    assert(n_hidden == 2)

    pre_learn = "cv"
    max_min = "max"

    print(f"-----{dataset}------")
    print("PD-BNN - SOLUTION:")

    if dataset.startswith("protein") or dataset.startswith("miniboo"):
        n_units = 200
    else:
        n_units = 100

    NR_FOLDS = commons.DATASET_TO_NR_FOLDS[dataset]

    if dataset in commons.REGRESSION_DATA_SETS:
        # regression
        pred_criteria = "rmse"
    else:
        # classification
        pred_criteria = "acc"
    
    all_selected_test_nll = np.zeros(NR_FOLDS)
    all_selected_test_pred_criteria = np.zeros(NR_FOLDS)
    all_selected_val_nll = np.zeros(NR_FOLDS)
    
    for id in range(NR_FOLDS):

        all_best_val_nll = {}

        for method_with_info in all_pb_bnn_methods:
            _, all_val_nll, best_idx = get_best_val_choice(dataset, id, method_with_info, n_hidden, n_units, max_min, pre_learn)

            all_best_val_nll[method_with_info] = all_val_nll[best_idx]

        best_method_with_info = min(all_best_val_nll, key=all_best_val_nll.get)

        df, _, best_idx = get_best_val_choice(dataset, id, best_method_with_info, n_hidden, n_units, max_min, pre_learn)

        all_selected_test_nll[id] = df.loc[best_idx, "test_nll"]
        all_selected_test_pred_criteria[id] = df.loc[best_idx, "test_" + pred_criteria]

        all_selected_val_nll[id] = df.loc[best_idx, "val_nll"]

    results = {}
    results["test_" + pred_criteria] = all_selected_test_pred_criteria
    results["test_nll"] = all_selected_test_nll
    results["val_nll"] = all_selected_val_nll
    # results["test_nll"] = all_selected_test_nll
    
    return results

ALL_PD_BNN_METHDOS = ["abs_fix", "row_fix", "sharma_fix", "layer_fix|1", "layer_fix|2"]

def get_results(dataset, n_hidden, method_with_info):
    if method_with_info == "pd-bnn-sub":
        return get_result_combined_partial_determininstic_BNN(dataset, n_hidden, all_pb_bnn_methods = ALL_PD_BNN_METHDOS)
    elif method_with_info == "pd-bnn-all":
        return get_result_combined_partial_determininstic_BNN(dataset, n_hidden, all_pb_bnn_methods = ["map_new", "sghmc"] + ALL_PD_BNN_METHDOS)
    else:
        method, num_fix_layer = commons.get_method_and_num_fix_layer(method_with_info, n_hidden)
        results, _, _ = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer)
        return results


def test_wilcoxon():
    n = 200

    my_x = np.random.uniform(size = n)
    my_y = np.random.uniform(size = n) - 0.1

    # Test Null-Hypotheses H_0: x <= y
    r = scipy.stats.wilcoxon(x = my_x, y= my_y, alternative = "greater")
    print(r)
    return 



def get_wilcoxon_test_result(all_test_nll, baseline, challenger):
    r = scipy.stats.wilcoxon(x = all_test_nll[baseline], y=all_test_nll[challenger], alternative = "greater")
    test_explanation_str = f"H_0: nll({commons.METHOD_TO_NICE_STR[baseline]}) <=  nll({commons.METHOD_TO_NICE_STR[challenger]})"
    return (test_explanation_str, r.pvalue)

def show_p_value(p_value):

    if p_value < 0.001:
        rounded_value = "$<$ 0.001"
    else:
        rounded_value = "%.3f" % p_value

    if p_value < 0.10:
        return " \\textbf{" +  rounded_value + "}"
    else:
        return rounded_value
    
def show_all_pd_bnn_results(REGRESSION_TASK, SHOW_GAP_DATA):
    n_hidden = 2

    ALL_DATASETS, ALL_FIELDS = commons.getDataSetsAndFields(REGRESSION_TASK, SHOW_GAP_DATA)
    ALL_METHODS = ["map_new", "sghmc", "pd-bnn-sub"] + ALL_PD_BNN_METHDOS

    ALL_FIELDS = ['val_nll', 'test_nll']
    
    print("PREDICION_AND_ANALYSIS_FIELDS = ", ALL_FIELDS)

    FIELD_TYPE = {}
    FIELD_TYPE["train_rmse"] = -1
    FIELD_TYPE["train_nll"] = -1
    FIELD_TYPE["test_rmse"] = -1
    FIELD_TYPE["test_nll"] = -1
    FIELD_TYPE["val_nll"] = -1
    FIELD_TYPE["train_acc"] = 1
    FIELD_TYPE["test_acc"] = 1

    output_text =  '''\midrule ''' + "\n"
    output_text += "\\bfseries Method" + " & " + " & ".join(["\\bfseries " + commons.FIELD_TO_NICE_STR[f] for f in ALL_FIELDS]) + " \\\\" + "\n"

    all_test_nll = {}
    for method_with_info in ALL_METHODS:
        all_test_nll[method_with_info] = []

    for dataset in ALL_DATASETS:
        
        output_text +=  '''\midrule ''' + "\n"
        output_text += '''\multicolumn{5}{c}{  ''' + commons.getNiceDataSetStr(dataset) + '''  } \\\\ ''' + "\n"  + '''\midrule ''' + "\n"
        
        field_to_results = {}
        for field in ALL_FIELDS:
            field_to_results[field] = np.zeros(len(ALL_METHODS))

        # get best result of each column
        for method_id, method_with_info in enumerate(ALL_METHODS): 

            results = get_results(dataset, n_hidden, method_with_info)

            all_test_nll[method_with_info].append(results["test_nll"])
            
            for field in ALL_FIELDS:
                field_to_results[field][method_id] = np.mean(results[field])
        
        field_to_best_method_id = {}
        for field in ALL_FIELDS:
            field_to_best_method_id[field] = set([np.argmax(FIELD_TYPE[field] * field_to_results[field])])
        
        # print("field_to_best_method_id = ", field_to_best_method_id)
        
        # print out nicely
        for method_id, method_with_info in enumerate(ALL_METHODS): 

            print("method_with_info = ", method_with_info)
            results = get_results(dataset, n_hidden, method_with_info)
            output_text += commons.METHOD_TO_NICE_STR[method_with_info] + commons.get_result_str(results, ALL_FIELDS, field_to_best_method_id, method_id)

    for method_with_info in ALL_METHODS:
        all_test_nll[method_with_info] = np.hstack(all_test_nll[method_with_info])

             
    # print("all_test_nll = ", all_test_nll)
    # print(output_text)

    # r = scipy.stats.wilcoxon(x = all_test_nll["map_new"], y=all_test_nll["sghmc"])
    # r = scipy.stats.wilcoxon(x = all_test_nll["map_new"], y=all_test_nll["pd-bnn-sub"])

    # print("orig = ", all_test_nll["sghmc"])

    # rnd_a = np.copy(all_test_nll["sghmc"])
    # np.random.shuffle(rnd_a)
    # r = scipy.stats.wilcoxon(x = all_test_nll["sghmc"], y=rnd_a)
    
    # r = scipy.stats.wilcoxon(x = all_test_nll["sghmc"], y=all_test_nll["pd-bnn-sub"])
    # r = scipy.stats.wilcoxon(x =all_test_nll["pd-bnn-sub"], y=all_test_nll["sghmc"])


    print("number of experiments = ", all_test_nll["map_new"].shape[0])

    # Test Null-Hypotheses H_0: x <= y
    r = scipy.stats.wilcoxon(x = all_test_nll["map_new"], y=all_test_nll["sghmc"], alternative = "greater")
    print("Null-Hypotheses H_0: nll(MAP) <=  nll(sghmc) , pvalue = ", r.pvalue)

    r = scipy.stats.wilcoxon(x = all_test_nll["map_new"], y=all_test_nll["pd-bnn-sub"], alternative = "greater")
    print("Null-Hypotheses H_0: nll(MAP) <=  nll(pd-bnn-sub) , pvalue = ", r.pvalue)

    r = scipy.stats.wilcoxon(x = all_test_nll["sghmc"], y=all_test_nll["pd-bnn-sub"], alternative = "greater")
    print("Null-Hypotheses H_0: nll(sghmc) <=  nll(pd-bnn-sub) , pvalue = ", r.pvalue)

    all_wilcoxon_results = []
    all_wilcoxon_results.append(get_wilcoxon_test_result(all_test_nll, "map_new", "sghmc"))
    all_wilcoxon_results.append(get_wilcoxon_test_result(all_test_nll, "map_new", "pd-bnn-sub"))
    all_wilcoxon_results.append(get_wilcoxon_test_result(all_test_nll, "sghmc", "pd-bnn-sub"))
    
    out_txt = ""
    for method in ALL_PD_BNN_METHDOS:
        out_txt += commons.METHOD_TO_NICE_STR[method] + " & "
        # out_txt += show_p_value(get_wilcoxon_test_result(all_test_nll, "map_new", method)[1])  + " & "
        # out_txt += show_p_value(get_wilcoxon_test_result(all_test_nll, "sghmc", method)[1]) + ''' \\\\ ''' + "\n"
        
        if method != "layer_fix|2":
            out_txt += show_p_value(get_wilcoxon_test_result(all_test_nll, "layer_fix|2", method)[1])  + ''' \\\\ ''' + "\n"
        

    print("")

    # print(''' $H_0$: nll(map) $\leq$ nll(full-random) ''' + " & " + show_p_value(get_wilcoxon_test_result(all_test_nll, "map_new", "sghmc")[1]) + ''' \\\\ ''' + "\n")

    # print(all_wilcoxon_results)
    print("out_txt: ")
    print(out_txt)


# run with python ./code/sghmc.py --dataset yacht --method sghmc_ext2 --id=-1
# get results with:
# scp -r danielandrade@mini3:"~/ResearchProjects/revised_sghmc/results/*" .

# test_wilcoxon()
# assert(False)

dataset = "yacht"
method = "map_new"
n_hidden = 2

# results, all_runtimes, analysis = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer = None)
# print("train_nll = ", np.mean(results["train_nll"]))

# results = get_result_combined_partial_determininstic_BNN(dataset, n_hidden)
# print("test_nll = ", np.mean(results["test_nll"]))
# print("val_nll = ", np.mean(results["val_nll"]))

# results, _, _ = commons.fully_analyze(dataset, method, n_hidden, num_fix_layer = None)
# print("test_nll = ", np.mean(results["test_nll"]))
# print("val_nll = ", np.mean(results["val_nll"]))

# show_all_pd_bnn_results(REGRESSION_TASK = True, SHOW_GAP_DATA = False)
# show_all_pd_bnn_results(REGRESSION_TASK = True, SHOW_GAP_DATA = True)
# show_all_pd_bnn_results(REGRESSION_TASK = False, SHOW_GAP_DATA = False)
show_all_pd_bnn_results(REGRESSION_TASK = False, SHOW_GAP_DATA = True)
