import csv
import os
import time
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import product
from math import sqrt
import math

import torch
import torch.nn as nn
import torch.optim as optim
from model.mlp import MLP
from utilities.util import (ensure_dir, gaussian_nll, load_uci_data, log_joint,
                            search_best)

from commons import LR, PRIOR_VAR, NOISE_VAR

from optbnn.bnn.priors import OptimGaussianPrior

from utilities.util import DataNormalizer, show_data_ratios

import commons
import multiprocessing
from functools import partial
from types import SimpleNamespace

SEED = 123

def cal_rmse_nll(id, model, list_param, noise_var, normalizer, dataset, device):
    list_param = list_param.copy()
    loss = nn.MSELoss()
    
    usage_list = ['train', 'val', 'test']
    rmse_list = []
    nll_list = []
            
    for usage in usage_list:
        X, y = load_uci_data(usage=usage, id=id, name=dataset, device=device)
        X = normalizer.zscore_normalize_x(X)
        pred = model(X)
        
        pred = normalizer.zscore_unnormalize_y(pred)
        
        # adjust noise_var appropriately
        noise_var_adjusted = noise_var * (normalizer.y_std ** 2)
        
        rmse = round(sqrt(loss(pred, y).item()), 5)
        rmse_list.append(rmse)
        nll = round(gaussian_nll(y, pred, noise_var_adjusted), 5)
        nll_list.append(nll)

    list_param += rmse_list + nll_list
    
    return list_param



def map(SAVE_PATH, id, lr, prior_var, noise_var, **specs):

    s = SimpleNamespace(**specs)
    
    assert(s.train_eval == 'train')
    
    device = torch.device(s.device)
    
    if s.device == "cuda":
        torch.cuda.set_device(s.n_gpu_use)
        print("USE GPU NR ", torch.cuda.current_device())

    torch.manual_seed(SEED)

    if s.pre_learn == 'cv':
        list_param = [lr, prior_var, noise_var]
    elif s.pre_learn == 'opt':
        list_param = [lr, noise_var]

    #? データの読み込み
    X_train, y_train = load_uci_data(usage='train', id=id, name=s.dataset, device=device)
    X_val, y_val = load_uci_data(usage='val', id=id, name=s.dataset, device=device)

    _, y_test = load_uci_data(usage='test', id=id, name=s.dataset, device=device)
    show_data_ratios(y_train, y_val, y_test)

    normalizer = DataNormalizer(X_train, y_train)
    X_train, y_train = normalizer.X_train, normalizer.y_train

    data_set = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=s.batch, shuffle=True, pin_memory=False)
    input_dim, output_dim = X_train.shape[1], 1

    #? モデルの読み込み
    model = MLP(input_dim, output_dim, [s.n_units]*s.n_hidden, s.act_fn, init_param='norm', task='regression')
    
    #! パラメータの初期化
    model.reset_parameters()
    
    model.to(device)
    # #? 最適化手法
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = noise_var / prior_var)
    
    neg_logjoint_list = []
    s_time = time.perf_counter()
    for i in range(s.epoch):
        
        nll_value = 0.
        for _, mini_batch in enumerate(dataloader):
            X_batch, y_batch = mini_batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            assert(isinstance(X_batch, torch.FloatTensor) or isinstance(X_batch, torch.cuda.FloatTensor))
            
            optimizer.zero_grad()
            assert(s.pre_learn == "cv")
            ave_neg_logjoint_value = -log_joint(model, X_batch, y_batch, noise_var = 1.0, prior_var = prior_var, device = device, task='regression', prior="no_prior") / y_batch.size()[0]
            neg_logjoint_list.append(ave_neg_logjoint_value.item())

            if math.isinf(neg_logjoint_list[-1]):
                break

            ave_neg_logjoint_value.backward()
            optimizer.step()
        
        if math.isinf(neg_logjoint_list[-1]):
            break
        
        if i % 100 == 0:
            print('epoch : ', i)
            print("neg_logjoin : ", neg_logjoint_list[-1])

        nll_value += ave_neg_logjoint_value.detach() * y_batch.size()[0]
    e_time = time.perf_counter()
    elapsed_t = int(e_time - s_time)
    nll_value /= len(data_set)
    neg_logjoint_list.append(nll_value)
    
    full_list_param = cal_rmse_nll(id, model, list_param, noise_var, normalizer, s.dataset, device)
    
    print("full_list_param = ", full_list_param)

    full_list_param.append(elapsed_t)
    

    if s.pre_learn == 'cv':
        model_name = f'lr{lr}_pvar{prior_var}_nvar{noise_var}_{s.n_hidden}h{s.n_units}_id{id}.pth'
    else:
        model_name = f'lr{lr}_nvar{noise_var}_{s.n_hidden}h{s.n_units}_id{id}.pth'

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))

    return full_list_param


def main(id, **specs):

    s = SimpleNamespace(**specs)

    #!###########################################################################################
    filename = f'map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'
    SAVE_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}'
    MAP_PATH = os.path.join(SAVE_PATH, filename)

    if s.train_eval == 'eval':
        SAVE_PATH = os.path.join(SAVE_PATH, 'eval')

    # ディレクトリの存在確認、もしなければ作成する
    ensure_dir(SAVE_PATH)
    #!###########################################################################################


    total_list = []
    
    if s.pre_learn == 'cv':
        if s.train_eval == 'train':
            header = ['lr', 'prior_var', 'noise_var', 'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll', 'elapsed_t']
            for lr, prior_var, noise_var in product(LR, PRIOR_VAR, NOISE_VAR):
                print(f'lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}')
                list_param = map(SAVE_PATH, id, lr, prior_var, noise_var, **specs)
                total_list.append(list_param)
        elif s.train_eval == 'eval':
            header = ['lr', 'prior_var', 'noise_var', 'train_rmse', 'test_rmse', 'train_nll', 'test_nll', 'elapsed_t']
            lr, prior_var, noise_var = search_best(MAP_PATH)
            print(f'lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}')
            list_param = map(SAVE_PATH, id, lr, prior_var, noise_var, **specs)
            total_list.append(list_param)
        elif s.train_eval == 'modelsave':
            lr, prior_var, noise_var = search_best(MAP_PATH)
            list_param = map(SAVE_PATH, id, lr, prior_var=prior_var, noise_var=noise_var, **specs)
            return
    elif s.pre_learn == 'opt':
        if s.train_eval == 'train':
            header = ['lr', 'noise_var', 'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll']
            
            for lr, noise_var in product(LR, NOISE_VAR):
                print(f'lr = {lr}, noise_var = {noise_var}')
                list_param = map(SAVE_PATH, id, lr=lr, prior_var=None, noise_var=noise_var, **specs)
                total_list.append(list_param)
        elif s.train_eval == 'eval':
            header = ['lr', 'noise_var', 'train_rmse', 'test_rmse', 'train_nll', 'test_nll', 'elapsed_t']
            lr, noise_var = search_best(path=MAP_PATH)
            list_param = map(SAVE_PATH, id, lr=lr, prior_var=None, noise_var=noise_var, **specs)
            total_list.append(list_param)
        elif s.train_eval == 'modelsave':
            lr, noise_var = search_best(MAP_PATH)
            list_param = map(SAVE_PATH, id, lr=lr, prior_var=None, noise_var=noise_var, **specs)
            return 
        
    with open(os.path.join(SAVE_PATH, filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_list)

    return


# Example:
# python ./code/map.py --dataset boston_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yacht') # name of dataset
    parser.add_argument('--act_fn', type=str, choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--id', type=int, default=1) # fold id
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval', 'modelsave'], default='train')
    parser.add_argument('--n_units', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')

    # specifies whether to use cross-valiation (cv) or OptBNN (opt) from "All you need is a good functional prior for Bayesian Deep Learning"
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv') 

    args = parser.parse_args()

    batch = 32
    dataset = args.dataset
    act_fn = args.act_fn
    foldId_specifier = args.id
    train_eval = args.train_eval
    n_units = args.n_units
    n_hidden = args.n_hidden
    pre_learn = args.pre_learn

    if dataset.startswith("protein"):
        n_units = 200
    else:
        assert(n_units == 100)


    if dataset.startswith("protein"):
        epoch = 500
    elif dataset.startswith("california"):
        epoch = 1000
    else:
        epoch = 2000

    print("foldId_specifier = ", foldId_specifier)
    all_fold_ids = commons.get_all_fold_ids(dataset, foldId_specifier)

    print("all_fold_ids = ", all_fold_ids)

    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None

    with multiprocessing.Pool(processes=all_fold_ids.shape[0]) as pool:
        pool.map(partial(main, dataset=dataset, train_eval=train_eval, n_hidden=n_hidden, n_units=n_units, act_fn = act_fn, device = args.device, pre_learn = pre_learn, epoch = epoch, batch = batch,  n_gpu_use = n_gpu_use), all_fold_ids)

    print('実験終了!!')








