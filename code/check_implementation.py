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

from utilities.util import (ensure_dir, gaussian_nll, load_uci_data, log_joint,
                            search_best)

from commons import LR, PRIOR_VAR, NOISE_VAR


from utilities.util import DataNormalizer, show_data_ratios

import commons
from types import SimpleNamespace

from extended_models import getModel

SEED = 123


def show_layer(net):
    for idx, (name, layer) in enumerate(net.named_children()):
        if idx == 0:
            assert(name == "layers") # contains all hidden layers
            for i, (param_name, param) in enumerate(layer.named_parameters()):
                print("param_name = ", param_name)
                print("param = ", param)
    
    return



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



def map(id, lr, prior_var, noise_var, **specs):

    s = SimpleNamespace(**specs)
    
    assert(s.train_eval == 'train')
    
    MAP_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}/map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'

    device = torch.device(s.device)
    
    if s.device == "cuda":
        torch.cuda.set_device(s.n_gpu_use)
        print("USE GPU NR ", torch.cuda.current_device())

    torch.manual_seed(SEED)

    assert(s.pre_learn == 'cv')
    list_param = [lr, prior_var, noise_var]
    
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
    model = getModel(MAP_PATH, id, task = "regression", input_dim = input_dim, output_dim = output_dim, hidden_dims = [s.n_units]*s.n_hidden, s = s)

    #! パラメータの初期化
    model.reset_parameters()
    
    model.to(device)
    # #? 最適化手法
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = noise_var / prior_var)
    
    print("model = ", model)
    show_layer(model)
    

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
    
    print("model = ", model)
    show_layer(model)

    assert(False)

    print("full_list_param = ", full_list_param)

    return full_list_param


def main(**specs):

    s = SimpleNamespace(**specs)

    id = 1
    total_list = []
    
    if s.pre_learn == 'cv':
        if s.train_eval == 'train':
            header = ['lr', 'prior_var', 'noise_var', 'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll', 'elapsed_t']
            for lr, prior_var, noise_var in product(LR, PRIOR_VAR, NOISE_VAR):
                print(f'lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}')
                list_param = map(id, lr, prior_var, noise_var, **specs)
                total_list.append(list_param)
        else:
            assert(False)
    else:
        assert(False)
    

    return


# Example:
# python ./code/map.py --dataset boston_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1 --num_fix_layer

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yacht') # name of dataset
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--method', type=str, choices=['sghmc', 'fix', 'abs_fix', 'layer_fix', 'row_fix', 'sharma_fix'], default='sghmc')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--num_fix_layer', type=int, default=1)

    args = parser.parse_args()

    batch = 32
    dataset = args.dataset
    act_fn = "relu"
    train_eval = "train"
    n_hidden = args.n_hidden
    pre_learn = "cv"
    max_min = "max"

    if dataset.startswith("protein"):
        n_units = 200
    else:
        n_units = 100
        

    if dataset.startswith("protein"):
        epoch = 500
    elif dataset.startswith("california"):
        epoch = 1000
    else:
        epoch = 2000

    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None

    main(method = args.method, dataset=dataset, train_eval=train_eval, n_hidden=n_hidden, n_units=n_units, act_fn = act_fn, device = args.device, pre_learn = pre_learn, epoch = epoch, batch = batch,  n_gpu_use = n_gpu_use, max_min = max_min, num_fix_layer = args.num_fix_layer)

    print('実験終了!!')
