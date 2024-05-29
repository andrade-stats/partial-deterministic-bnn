import csv
import os
import time
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import product

import numpy as np
import torch
import torch.optim as optim
from model.mlp import MLP
from utilities.util import ensure_dir, load_uci_data, log_joint, search_best

from commons import LR, PRIOR_VAR

from optbnn.bnn.priors import OptimGaussianPrior
from optbnn.metrics.metrics_tensor import accuracy, nll

from utilities.util import normalize_x_only, show_data_ratios

import commons
import multiprocessing
from functools import partial
from types import SimpleNamespace

SEED = 123

# モデルの精度(accuracy)と負の対数尤度(negative log likelihood)を計算する関数。
def get_acc_nll(model, dataloader):
    accs = [] 
    nlls = []
    
    model.eval()
    
    with torch.no_grad():
        for _, (data, target) in enumerate(dataloader):
            log_prob = model.predict(data)
            
            acc_ = accuracy(log_prob, target)
            accs.append(acc_)
            
            nll_ = nll(log_prob, target)
            nlls.append(nll_.item())
    
    acc_mean = round(np.mean(accs), 5)
    nll_mean = round(np.mean(nlls), 5)
    
    return acc_mean, nll_mean



def map(id, SAVE_PATH, lr, prior_var, **specs):

    s = SimpleNamespace(**specs)


    device = torch.device(s.device)
    
    if s.device == "cuda":
        torch.cuda.set_device(s.n_gpu_use)
        print("USE GPU NR ", torch.cuda.current_device())
    
    torch.manual_seed(SEED)

    #? データの読み込み
    train_X, train_y = load_uci_data(usage='train', id=id, name=s.dataset, device=device)
    val_X, val_y = load_uci_data(usage='val', id=id, name=s.dataset, device=device)
    test_X, test_y = load_uci_data(usage='test', id=id, name=s.dataset, device=device)
    
    #? int型に変換
    train_y = train_y.to(dtype=torch.int64).reshape(-1)
    val_y = val_y.to(dtype=torch.int64).reshape(-1)
    test_y = test_y.to(dtype=torch.int64).reshape(-1)

    show_data_ratios(train_y, val_y, test_y)
    
    train_X = normalize_x_only(train_X)

    data_set = torch.utils.data.TensorDataset(train_X, train_y)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=s.batch, shuffle=True, pin_memory=False)
    input_dim, output_dim = train_X.shape[1], torch.unique(train_y).shape[0]
    
    model = MLP(input_dim, output_dim, s.hidden_dims, s.act_fn, init_param='norm', task='classification')
    model.to(device)
    #! パラメータの初期化
    model.reset_parameters()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = 1.0 / prior_var)
    
    neg_logjoint_list = []
    s_time = time.perf_counter()
    for i in range(s.epoch):
        
        nll_value = 0.
        for _, mini_batch in enumerate(dataloader):
            X_batch, y_batch = mini_batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            assert(s.pre_learn == "cv")
            ave_neg_logjoint_value = -log_joint(model, X_batch, y_batch, None, prior_var, device, task='classification', prior="no_prior") / y_batch.size()[0]
            ave_neg_logjoint_value.backward()
            optimizer.step()
            
            nll_value += ave_neg_logjoint_value.detach() * y_batch.size()[0]
        nll_value /= len(data_set)
        neg_logjoint_list.append(nll_value)

        if i % 50 == 0:
            print('epoch : ', i)
            print("neg_logjoin : ", neg_logjoint_list[-1])
    
    e_time = time.perf_counter() # 経過時間
    
    train_dataset = torch.utils.data.TensorDataset(normalize_x_only(train_X), train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=s.batch, shuffle=True, pin_memory=False)
    val_dataset = torch.utils.data.TensorDataset(normalize_x_only(val_X), val_y)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=s.batch, shuffle=False, pin_memory=False)
    test_dataset = torch.utils.data.TensorDataset(normalize_x_only(test_X), test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=s.batch, shuffle=False, pin_memory=False)
    train_acc, train_nll = get_acc_nll(model, train_dataloader)
    val_acc, val_nll = get_acc_nll(model, val_dataloader)
    test_acc, test_nll = get_acc_nll(model, test_dataloader)
    
    print("train_acc, train_nll = ", (train_acc, train_nll))
    print("val_acc, val_nll = ", (val_acc, val_nll))

    elapsed_t = int(e_time - s_time)
    print(f'経過時間(in seconds) = {elapsed_t}') 
    
    if s.pre_learn == 'cv':
        full_list_param = [lr, prior_var, train_acc, val_acc, test_acc, train_nll, val_nll, test_nll, elapsed_t]
    else:
        full_list_param = [lr, train_acc, val_acc, test_acc, train_nll, val_nll, test_nll, elapsed_t]

   
    if s.pre_learn == 'cv':
        model_name = f'lr{lr}_pvar{prior_var}_{s.n_hidden}h{s.n_units}_id{id}.pth'
    else:
        model_name = f'lr{lr}_{s.n_hidden}h{s.n_units}_id{id}.pth'

    torch.save(model.state_dict(), os.path.join(SAVE_PATH, model_name))
    
    return full_list_param


def main(id, **specs):

    s = SimpleNamespace(**specs)

    #! ####################################################################
    SAVE_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}' 
    MAP_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}/map_new_{s.n_hidden}h{s.n_units}_id{id}.csv' 
    filename = f'map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'

    if s.train_eval == 'eval':
        SAVE_PATH = os.path.join(SAVE_PATH, 'eval')

    ensure_dir(SAVE_PATH)
    #! ####################################################################

    total_list = []
    
    assert(s.train_eval == 'train')

    if s.pre_learn == 'cv':
        if s.train_eval == 'train':
            header = ['lr', 'prior_var', 'train_acc', 'val_acc', 'test_acc', 'train_nll', 'val_nll', 'test_nll', 'elapsed_t']
            for lr, prior_var in product(LR, PRIOR_VAR):
                print(f'lr = {lr}, prior_var = {prior_var}')
                list_param = map(id, SAVE_PATH, lr, prior_var, **specs)
                total_list.append(list_param)
        elif s.train_eval == 'eval':
            header = ['lr', 'prior_var', 'train_acc', 'test_acc', 'train_nll', 'test_nll', 'elapsed_t']
            lr, prior_var = search_best(MAP_PATH)
            list_param = map(id, SAVE_PATH, lr, prior_var, specs)
            total_list.append(list_param)
        elif s.train_eval == 'modelsave':
            lr, prior_var = search_best(MAP_PATH)
            list_param = map(id, SAVE_PATH, lr, prior_var, specs)
            return
    elif s.pre_learn == 'opt':
        if s.train_eval == 'train':
            header = ['lr', 'train_acc', 'val_acc', 'test_acc', 'train_nll', 'val_nll', 'test_nll']
            for lr in LR:
                print(f'lr = {lr}')
                list_param = map(id, SAVE_PATH, lr, None, specs)
                total_list.append(list_param)
        elif s.train_eval == 'eval':
            header = ['lr', 'train_acc', 'test_acc', 'train_nll', 'test_nll', 'elapsed_t']
            lr = search_best(MAP_PATH)[0]
            print(f'lr = {lr}')
            list_param = map(id, SAVE_PATH, lr, None, specs)
            total_list.append(list_param)
        elif s.train_eval == 'modelsave':
            lr = search_best(MAP_PATH)[0]
            print(f'lr = {lr}')
            list_param = map(id, SAVE_PATH, lr, None, specs)
            return

    #? csvファイルで保存
    with open(os.path.join(SAVE_PATH, filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_list)
    
    return


# Example:
# python ./code/classification_map.py --dataset htru2_gap --n_hidden=1 --train_eval train --id=-1 

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='eeg') # name of dataset
    parser.add_argument('--id', type=int, default=1) # fold id
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval', 'modelsave'], default='train')
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--n_units', type=int, default=100)
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--one_by_one', type=str, choices=['yes', 'no'], default='no')

    args = parser.parse_args()

    batch = 64
    dataset = args.dataset

    foldId_specifier = args.id
    train_eval = args.train_eval
    n_hidden = args.n_hidden
    n_units = args.n_units

    if dataset.startswith("miniboo"):
        n_units = 200
        epoch = 100
    else:
        epoch = 300
        assert(n_units == 100)

    pre_learn = args.pre_learn
    hidden_dims = [n_units]*n_hidden

    act_fn = 'relu'
    
    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None
    
    print("foldId_specifier = ", foldId_specifier)
    all_fold_ids = commons.get_all_fold_ids(dataset, foldId_specifier)

    print("all_fold_ids = ", all_fold_ids)
    
    if args.one_by_one == "no":
        with multiprocessing.Pool(processes=all_fold_ids.shape[0]) as pool:
            pool.map(partial(main, dataset=dataset, train_eval=train_eval, n_hidden=n_hidden, n_units=n_units, hidden_dims=hidden_dims, act_fn = act_fn, device = args.device, pre_learn = pre_learn, epoch = epoch, batch = batch, n_gpu_use = n_gpu_use), all_fold_ids)
    else:
        print("WARNING RUN ONE BY ONE (MIGHT BE SLOW)")
        for id in all_fold_ids:
            main(id, dataset=dataset, train_eval=train_eval, n_hidden=n_hidden, n_units=n_units, hidden_dims=hidden_dims, act_fn = act_fn, device = args.device, pre_learn = pre_learn, epoch = epoch, batch = batch, n_gpu_use = n_gpu_use)

    print('実験終了!!')
