
import numpy as np
import pandas as pd
import os

import GPUtil
import torch

LR = [0.01]
PRIOR_VAR = [1000.0, 100.0, 50.0, 10.0, 1.0]
NOISE_VAR = [1.0, 0.5, 0.1, 0.01]

REGRESSION_DATA_SETS = ["boston", "yacht", "concrete", "energy", "california", "protein"] + ["boston_gap", "yacht_gap", "concrete_gap", "energy_gap", "california_gap", "protein_gap"]

METHOD_TO_NICE_STR = {}
METHOD_TO_NICE_STR["map"] = "map"
METHOD_TO_NICE_STR["map_new"] = "map-new"
METHOD_TO_NICE_STR["sghmc"] = "sghmc"
METHOD_TO_NICE_STR["layer_fix|1"] = "layer-fix|1"
METHOD_TO_NICE_STR["layer_fix|2"] = "layer-fix|2"
METHOD_TO_NICE_STR["layer_fix"] = "layer-fix"
METHOD_TO_NICE_STR["sharma_fix"] = "sharma"
METHOD_TO_NICE_STR["abs_fix"] = "abs-fix"
METHOD_TO_NICE_STR["row_fix"] = "row-fix"

def getFilename(id, method, n_hidden, n_units, num_fix_layer = None, max_min = None):

    if method == 'abs_fix' or method == 'row_fix':
        return f'{max_min}_{method}_{n_hidden}h{n_units}_id{id}.csv'
    elif method == 'layer_fix':
        return f'layer_fix{num_fix_layer}_{n_hidden}h{n_units}_id{id}.csv'
    else:
        return f'{method}_{n_hidden}h{n_units}_id{id}.csv'
    

DATASET_TO_NR_FOLDS = {}
DATASET_TO_NR_FOLDS["boston_gap"] = 13
DATASET_TO_NR_FOLDS["yacht_gap"] = 6
DATASET_TO_NR_FOLDS["concrete_gap"] = 8
DATASET_TO_NR_FOLDS["energy_gap"] = 8
DATASET_TO_NR_FOLDS["california_gap"] = 8
DATASET_TO_NR_FOLDS["protein_gap"] = 9
DATASET_TO_NR_FOLDS["htru2_gap"] = 8
DATASET_TO_NR_FOLDS["eeg_gap"] = 14
DATASET_TO_NR_FOLDS["letter_gap"] = 16

DATASET_TO_NR_FOLDS["boston"] = 10
DATASET_TO_NR_FOLDS["yacht"] = 10
DATASET_TO_NR_FOLDS["concrete"] = 10
DATASET_TO_NR_FOLDS["energy"] = 10
DATASET_TO_NR_FOLDS["california"] = 10
DATASET_TO_NR_FOLDS["protein"] = 5
DATASET_TO_NR_FOLDS["htru2"] = 10
DATASET_TO_NR_FOLDS["eeg"] = 10
DATASET_TO_NR_FOLDS["letter"] = 10
DATASET_TO_NR_FOLDS["miniboo"] = 5

def get_all_fold_ids(dataset, foldId_specifier):
    if dataset.endswith("_gap"):
        assert(foldId_specifier == -1)
        assert(dataset in DATASET_TO_NR_FOLDS)
        return np.arange(DATASET_TO_NR_FOLDS[dataset]) + 1
    elif foldId_specifier == -1:
        if (dataset == 'miniboo') or (dataset == 'protein'):
            return np.arange(5) + 1
        else:
            return np.arange(10) + 1
    else:
        assert(foldId_specifier >= 1 and foldId_specifier <= 10)
        return np.asarray([foldId_specifier])


def fully_analyze(dataset, method, n_hidden, num_fix_layer):

    pre_learn = "cv"
    max_min = "max"

    print(f"-----{dataset}------")
    print(f"{method} - SOLUTION:")

    if dataset.startswith("protein") or dataset.startswith("miniboo"):
        n_units = 200
    else:
        n_units = 100

    NR_FOLDS = DATASET_TO_NR_FOLDS[dataset]

    if dataset in REGRESSION_DATA_SETS:
        # regression
        pred_criteria = "rmse"
    else:
        # classification
        pred_criteria = "acc"
    
     # "raw" means evaluated at the predicted values of X_test (otherwise evaluated on weights of the last layer)
    SAMPLING_ANALYSIS_FIELDS = ['mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw']

    all_minimum_train_pred_criteria = np.zeros(NR_FOLDS)
    all_minimum_train_nll = np.zeros(NR_FOLDS)
    all_selected_lr = np.zeros(NR_FOLDS)
    all_selected_test_nll = np.zeros(NR_FOLDS)
    all_selected_test_pred_criteria = np.zeros(NR_FOLDS)
    all_runtimes = np.zeros(NR_FOLDS)

    analysis = {}
    for field in SAMPLING_ANALYSIS_FIELDS:
        analysis[field] = np.zeros(NR_FOLDS)

    for id in range(NR_FOLDS):
        filename = getFilename(id + 1, method, n_hidden, n_units, num_fix_layer, max_min)

        SAVE_PATH = f'./results/{method}/{dataset}/{pre_learn}'
        NEW_PATH = os.path.join(SAVE_PATH, filename)
        
        df = pd.read_csv(NEW_PATH)

        # print("NEW_PATH = ", NEW_PATH)
        # print("df = ")
        # print(df)
        
        df = df.loc[df["lr"] == 0.01]
        df.index = np.arange(len(df.index))
        
        all_minimum_train_pred_criteria[id] = np.nanmin(df.loc[:,"train_" + pred_criteria].values)
        all_minimum_train_nll[id] = np.nanmin(df.loc[:,"train_nll"].values)

        all_val_nll = df.loc[:,"val_nll"].values
        best_idx = np.nanargmin(all_val_nll)

        all_selected_lr[id] = df.loc[best_idx, "lr"]
        all_selected_test_nll[id] = df.loc[best_idx, "test_nll"]
        all_selected_test_pred_criteria[id] = df.loc[best_idx, "test_" + pred_criteria]

        all_runtimes[id] = df.loc[best_idx, "elapsed_t"]

        if method != "map_new":
            for field in SAMPLING_ANALYSIS_FIELDS:
                analysis[field][id] = df.loc[best_idx, field]

    print("best setting = ")
    print("all_minimum_train_nll = ", all_minimum_train_nll)
    print("all_selected_lr = ", all_selected_lr)
    print("all_selected_test_nll = ", all_selected_test_nll)
    
    print("avg min train_nll = ", np.mean(all_minimum_train_nll))
    print("avg test_nll = ", np.mean(all_selected_test_nll))

    results = {}
    results["train_" + pred_criteria] = all_minimum_train_pred_criteria
    results["test_" + pred_criteria] = all_selected_test_pred_criteria
    results["train_nll"] = all_minimum_train_nll
    results["test_nll"] = all_selected_test_nll
    
    
    return results, all_runtimes, analysis



def get_most_freemem_gpu():
    max_mem_ = 0
    max_id_ = 0
    i = 0 
    for g in GPUtil.getGPUs():
        if g.memoryFree > max_mem_ :
            max_mem_ = g.memoryFree
            max_id_ = i
        i += 1
    return(max_id_)



