import argparse
import csv
import math
import os
import shutil
import time
from distutils.util import strtobool
from itertools import product

import numpy as np
import torch

from utilities.util import (ensure_dir, ess_rhat_func, from_torch_to_numpy,
                            load_uci_data, round_func, search_best, show_data_ratios)

from optbnn.bnn.likelihoods import LikGaussian
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.metrics import uncertainty as uncertainty_metrics
from optbnn.sgmcmc_bayes_net.regression_net import RegressionNet
from optbnn.utils import util

from commons import LR, PRIOR_VAR, NOISE_VAR

import commons
import multiprocessing
from functools import partial
from types import SimpleNamespace

from extended_models import getModel


def sghmc(id, MAP_PATH, tmp_file_identifier, lr, prior_var, noise_var, **specs):

    s = SimpleNamespace(**specs)

    device = torch.device(s.device)

    if s.device == "cuda":
        N_GPU = s.n_gpu_use
        torch.cuda.set_device(N_GPU)
        print("USE GPU NR ", torch.cuda.current_device())
    else:
        N_GPU = None

    hidden_dims = [s.n_units] * s.n_hidden

    SEED = 123
    assert(s.train_eval == 'train')

    print(f"Loading split {id} of {s.dataset} dataset")
    
    # データセットを読み込み (正規化してない)
    X_train, y_train = load_uci_data(usage='train', id=id, name=s.dataset, device=device)
    X_val, y_val = load_uci_data(usage='val', id=id, name=s.dataset, device=device)
    X_test, y_test = load_uci_data(usage='test', id=id, name=s.dataset, device=device)
    
    show_data_ratios(y_train, y_val, y_test)

    # numpyに変換
    X_train, y_train = from_torch_to_numpy(X_train, y_train)
    X_val, y_val = from_torch_to_numpy(X_val, y_val)
    X_test, y_test = from_torch_to_numpy(X_test, y_test)

    # 次元数
    input_dim, output_dim = int(X_train.shape[-1]), 1

    # SGHMCサンプラーの設定
    sampling_configs = {
        "batch_size": 32,
        "num_samples": 40, 
        "n_discarded": 10,
        "num_burn_in_steps": s.NUM_BURN_IN_KEEP_EVERY,
        "keep_every": s.NUM_BURN_IN_KEEP_EVERY,
        "lr": 1e-2,
        "num_chains": 4,
        "mdecay": 1e-2,
        "print_every_n_samples": 5
    }
    
    sampling_configs['lr'] = lr
    
    util.set_seed(SEED)

    net = getModel(MAP_PATH, id, task = "regression", input_dim = input_dim, output_dim = output_dim, hidden_dims = hidden_dims, s = s)

    likelihood = LikGaussian(noise_var)

    # 標準ガウス事前分布の初期化
    if s.pre_learn == 'opt':
        ckpt_path = f'./results/prelearn_pvar/{s.dataset}/ckpts/it-200_nvar{noise_var}_{s.n_hidden}h{s.n_units}_id{id}.ckpt'
        prior = OptimGaussianPrior(ckpt_path, device)
    elif s.pre_learn == 'cv':  
        prior_std = math.sqrt(prior_var)
        prior = FixedGaussianPrior(std=prior_std, device = device) 
        
    bayes_net = RegressionNet(net, likelihood, prior, tmp_file_identifier, n_gpu=N_GPU, normalize_input=True, normalize_output=True)
    
    # サンプリング開始
    s_time = time.perf_counter()

    bayes_net.sample_multi_chains(X_train, y_train, **sampling_configs)
    e_time = time.perf_counter()
    elapsed_t = e_time - s_time # 経過時間
    
    # 第2層の重みを保存
    sampled_models = bayes_net.sampled_weights
    
    # print("sampled_models = ", sampled_models)
    # assert(False)
    print("len(sampled_models) = ", len(sampled_models))
    assert(len(sampled_models) == 4*30) # there should be 30 samples from each left that we are actually using for evaluation

    sampled_lastW = [t[-2] for t in sampled_models]
    sampled_last_W = np.hstack(sampled_lastW).T

    pred_mean, pred_var = bayes_net.predict(X_train)
    train_rmse = uncertainty_metrics.rmse(pred_mean, y_train)
    train_nll = uncertainty_metrics.gaussian_nll(y_train, pred_mean, pred_var)

    pred_mean, pred_var = bayes_net.predict(X_val)
    val_rmse = uncertainty_metrics.rmse(pred_mean, y_val)
    val_nll = uncertainty_metrics.gaussian_nll(y_val, pred_mean, pred_var)

    pred_mean, pred_var, _, raw_preds = bayes_net.predict(X_test, True, True)
    test_rmse = uncertainty_metrics.rmse(pred_mean, y_test)
    test_nll = uncertainty_metrics.gaussian_nll(y_test, pred_mean, pred_var)
    
    print("run evaluation of ESS and R-hat")
    # モデルの最後の層の重みでESSとRハットを計算
    mean_ess, min_ess, mean_rhat, max_rhat  = ess_rhat_func(sampled_last_W, sampling_configs['num_chains'])
    # 予測分布でESSとRハットを計算
    mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw = ess_rhat_func(raw_preds, sampling_configs['num_chains'])  

    print(f"> RMSE = {test_rmse:.4f} | NLL = {test_nll:.4f}")
    if s.pre_learn == 'cv':
        
        list_param = [lr, prior_var, noise_var, train_rmse, val_rmse,  test_rmse, train_nll, val_nll, test_nll,
                        mean_ess, min_ess, mean_rhat, max_rhat, mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw, int(elapsed_t)]
    
    elif s.pre_learn == 'opt':
        if s.train_eval == 'train':
            list_param = [lr, noise_var, train_rmse, val_rmse, test_rmse, train_nll, val_nll, int(test_nll)]
        elif s.train_eval == 'eval':
            list_param = [lr, noise_var, train_rmse, test_rmse, train_nll, test_nll, 
                          mean_ess, min_ess, mean_rhat, max_rhat, mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw, int(elapsed_t)]
    
    return round_func(list_param)


def main(id, **specs):

    s = SimpleNamespace(**specs)

    #! ############################################################################################################
    SAVE_PATH = f'./results/{s.method}/{s.dataset}/{s.pre_learn}'
    MAP_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}/map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'

    if s.method == 'sghmc':
        filename = f'sghmc_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'abs_fix':
        filename = f'{s.max_min}_abs_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'layer_fix':
        filename = f'layer_fix{s.num_fix_layer}_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'row_fix':
        filename = f'{s.max_min}_row_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'sharma_fix':
        filename = f'sharma_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    else:
        assert(False)

    PARAM_PATH = os.path.join(SAVE_PATH, filename)
        
    ensure_dir(dir=SAVE_PATH)
    #! ############################################################################################################


    tmp_file_identifier = f"tmp_{s.dataset}_{id}_{s.max_min}_{s.method}_{s.n_hidden}_{s.n_units}_{s.num_fix_layer}_{s.pre_learn}_{s.train_eval}"
    print("tmp_file_identifier = ", tmp_file_identifier)

    total_list = []
    
    if s.pre_learn == 'cv':
        
        header = ['lr', 'prior_var', 'noise_var', 'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll',
                    'mean_ess', 'min_ess', 'mean_rhat', 'max_rhat', 'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw','elapsed_t']
        for lr, prior_var, noise_var in product(LR, PRIOR_VAR, NOISE_VAR):
            print(f'lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}')
            list_param = sghmc(id, MAP_PATH, tmp_file_identifier, lr=lr, prior_var=prior_var, noise_var=noise_var, **specs)
            total_list.append(list_param)
    
    elif s.pre_learn == 'opt':
        if s.train_eval == 'train':
            header = ['lr', 'noise_var', 'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll']
            for lr, noise_var in product(LR, NOISE_VAR):
                print(f'lr = {lr}, noise_var = {noise_var}')
                list_param = sghmc(id, MAP_PATH, tmp_file_identifier, lr=lr, prior_var=None, noise_var=noise_var, **specs)
                total_list.append(list_param)
        elif s.train_eval == 'eval':
            header = ['lr', 'noise_var', 'train_rmse', 'test_rmse', 'train_nll', 'test_nll',
                      'mean_ess', 'min_ess', 'mean_rhat', 'max_rhat', 'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw', 'elapsed_t']
            lr, noise_var = search_best(PARAM_PATH)
            print(f'lr = {lr}, noise_var = {noise_var}')
            list_param = sghmc(id, MAP_PATH, tmp_file_identifier, lr=lr, prior_var=None, noise_var=noise_var, **specs)
            total_list.append(list_param)
            
    
    with open(os.path.join(SAVE_PATH, filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_list)

    path = f'./{tmp_file_identifier}'
    if os.path.isdir(path):
        shutil.rmtree(path)

    return




# Example:
# python ./code/sghmc.py --method=row_fix --max_min=max --dataset boston_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1
# python ./code/sghmc.py --method=sghmc --dataset boston_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--act_fn', type=str, choices=['tanh', 'relu'], default='relu')
    parser.add_argument('--dataset', type=str, default='yacht') # name of dataset
    parser.add_argument('--fix_bias', type=str, choices=['t', 'f'], default='f')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--init_param', type=str, choices=['norm', 'map'], default='norm')
    parser.add_argument('--max_min', type=str, choices=['max', 'min'], default='max')
    parser.add_argument('--method', type=str, choices=['sghmc', 'fix', 'abs_fix', 'layer_fix', 'row_fix', 'sharma_fix'], default='sghmc')
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--num_fix_layer', type=int, default=1)
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv')
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()

    act_fn = args.act_fn
    dataset = args.dataset
    fix_bias = bool(strtobool(args.fix_bias))
    foldId_specifier = args.id
    init_param = args.init_param
    max_min = args.max_min
    method = args.method
    n_hidden = args.n_hidden
    num_fix_layer = args.num_fix_layer
    pre_learn = args.pre_learn # 事前学習モデルの使用
    train_eval = args.train_eval

    if dataset.startswith("protein"):
        n_units = 200
        NUM_BURN_IN_KEEP_EVERY = 5000
    else:
        n_units = 100
        NUM_BURN_IN_KEEP_EVERY = 2000

    assert(act_fn == "relu")

    print("foldId_specifier = ", foldId_specifier)
    all_fold_ids = commons.get_all_fold_ids(dataset, foldId_specifier)

    print("all_fold_ids = ", all_fold_ids)
    
    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None

    with multiprocessing.Pool(processes=all_fold_ids.shape[0]) as pool:
        pool.map(partial(main, dataset=dataset, train_eval=train_eval, n_hidden=n_hidden, n_units=n_units, act_fn = act_fn, device = args.device, pre_learn = pre_learn, fix_bias = fix_bias, init_param = init_param, max_min = max_min, method = method, num_fix_layer = num_fix_layer, NUM_BURN_IN_KEEP_EVERY = NUM_BURN_IN_KEEP_EVERY,  n_gpu_use = n_gpu_use), all_fold_ids)

    print('実験終了!!')