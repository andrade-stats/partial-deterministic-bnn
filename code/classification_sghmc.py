import csv
import math
import os
import shutil
import time
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import product

import numpy as np
import torch
import torch.utils.data as data_utils


from utilities.util import (ensure_dir, ess_rhat_func, load_uci_data,
                            search_best)

from optbnn.bnn.likelihoods import LikCategorical
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.metrics.metrics_tensor import accuracy, nll
from optbnn.sgmcmc_bayes_net.classification_net import ClassificationNet
from optbnn.utils import util

from utilities.util import normalize_x_only, show_data_ratios

from extended_models import getModel

from commons import LR, PRIOR_VAR

import commons
import multiprocessing
from functools import partial
from types import SimpleNamespace


def get_acc_nll(bayes_net, dataloader):
    accs = []
    nlls = []
    with torch.no_grad():
        for _, (data, target) in enumerate(dataloader):
            log_pred_mean = bayes_net.log_predict(
                data, num_samples=40)

            acc_ = accuracy(log_pred_mean, target)
            nll_ = nll(log_pred_mean, target)
            accs.append(acc_)
            nlls.append(nll_.item())
    
    return round(np.mean(accs), 5), round(np.mean(nlls), 5)
    




def sghmc(id, MAP_PATH, tmp_file_identifier, lr, prior_var, **specs):

    s = SimpleNamespace(**specs)

    device = torch.device(s.device)

    if s.device == "cuda":
        N_GPU = s.n_gpu_use
        torch.cuda.set_device(N_GPU)
        print("USE GPU NR ", torch.cuda.current_device())
    else:
        N_GPU = None

    train_X, train_y = load_uci_data('train', id, s.dataset, device)
    train_y = train_y.to(dtype=torch.int64).reshape(-1)
    val_X, val_y = load_uci_data('val', id, s.dataset, device)
    val_y = val_y.to(dtype=torch.int64).reshape(-1)
    test_X, test_y = load_uci_data('test', id, s.dataset, device)
    test_y = test_y.to(dtype=torch.int64).reshape(-1)

    show_data_ratios(train_y, val_y, test_y)

    input_dim = train_X.shape[1]
    output_dim = torch.unique(train_y).shape[0]
    hidden_dims = [s.n_units] * s.n_hidden

    sampler_batch_size = 64
    sampling_configs = {
        "batch_size": sampler_batch_size, # Mini-batch size
        "num_samples": 40,                # Total number of samples for each chain 60から40に変更
        "n_discarded": 10,                # Number of the first samples to be discared for each chain
        "num_burn_in_steps": s.NUM_BURN_IN_KEEP_EVERY,         # Number of burn-in steps
        "keep_every": s.NUM_BURN_IN_KEEP_EVERY,                # Thinning interval
        "lr": 1e-2,                       # Step size
        "num_chains": 4,                  # Number of chains
        "mdecay": 1e-2,                   # Momentum coefficient
        "print_every_n_samples": 5
    }


    util.set_seed(123)
    
    assert(s.train_eval == 'train')

    print(f"Loading split {id} of {s.dataset} dataset")

    sampling_configs['lr'] = lr
    
    train_data_loader = data_utils.DataLoader(
                data_utils.TensorDataset(normalize_x_only(train_X), train_y),
                batch_size=sampler_batch_size, shuffle=True)

    val_data_loader = data_utils.DataLoader(
                data_utils.TensorDataset(normalize_x_only(val_X), val_y),
                batch_size=sampler_batch_size, shuffle=False)
    
    test_data_loader = data_utils.DataLoader(
                data_utils.TensorDataset(normalize_x_only(test_X), test_y),
                batch_size=sampler_batch_size, shuffle=False)
    
    # Setup the likelihood
    likelihood = LikCategorical()
    
    if s.pre_learn == 'opt':
        # Load the optimized prior
        ckpt_path = f'./results/prelearn_pvar/{s.dataset}/ckpts/it-200_{s.n_hidden}h{s.n_units}_id{id}.ckpt'
        print(f"Loading prior: {ckpt_path}")
        prior = OptimGaussianPrior(ckpt_path, device)
    elif s.pre_learn == 'cv':
        prior_std = math.sqrt(prior_var)
        prior = FixedGaussianPrior(std=prior_std, device = device)
    
    
    net = getModel(MAP_PATH, id, task = "classification", input_dim = input_dim, output_dim = output_dim, hidden_dims = hidden_dims, s = s)

    # Initialize the Bayesian net
    bayes_net = ClassificationNet(net, likelihood, prior, tmp_file_identifier, n_gpu=N_GPU)
    
    s_time = time.perf_counter() 
    # Start sampling using SGHMC sampler
    bayes_net.sample_multi_chains(data_loader=train_data_loader, **sampling_configs)
    e_time = time.perf_counter() 
    elapsed_t = int(e_time - s_time) # time in seconds
    
    # # Make predictions using the posterior
    sampled_weights_loader = bayes_net._load_all_sampled_weights()
    sampled_weights = []
    for idx, weights in enumerate(sampled_weights_loader):
        sampled_weights.append(weights)

    sampled_lastW = [(t['output_layer.W'].to('cpu').detach().numpy().copy().flatten()) for t in sampled_weights]
    sampled_lastW = np.vstack(sampled_lastW)
    train_acc, train_nll = get_acc_nll(bayes_net=bayes_net, dataloader=train_data_loader)
    val_acc, val_nll = get_acc_nll(bayes_net=bayes_net, dataloader=val_data_loader)
    test_acc, test_nll = get_acc_nll(bayes_net=bayes_net, dataloader=test_data_loader)
    

    print("Run evaluations of ESS and R-hat")
    _, log_predictions = bayes_net.log_predict(normalize_x_only(test_X), return_individual_predictions=True)
    max_log_predictions, _ = torch.max(log_predictions, dim=2)
    max_predictions = torch.exp(max_log_predictions)

    print("test_X.shape = ", test_X.shape)
    print("log_predictions.shape = ", log_predictions.shape)
    print("max_predictions.shape = ", max_predictions.shape)
    print("sampled_lastW.shape = ", sampled_lastW.shape)
    print("nr of samples = ", sampled_lastW.shape[0])

    all_class_predicted_probabilities = torch.exp(torch.flatten(log_predictions, start_dim=1)) # result shape = (number of MCMC samples, number of classes * number of test samples)
    print("all_class_predicted_probabilities.shape = ", all_class_predicted_probabilities.shape)
    
    mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw = ess_rhat_func(all_class_predicted_probabilities.to('cpu').detach().numpy().copy(), sampling_configs['num_chains']) 
    mean_ess, min_ess, mean_rhat, max_rhat  = ess_rhat_func(sampled_lastW, sampling_configs['num_chains'])
        

    if s.pre_learn == 'opt':
        list_param = [lr, train_acc, val_acc, test_acc, train_nll, val_nll, test_nll,
                        mean_ess, min_ess, mean_rhat, max_rhat, mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw, elapsed_t]
    elif s.pre_learn == 'cv':
        list_param = [lr, prior_var, train_acc, val_acc, test_acc, train_nll, val_nll, test_nll,
                        mean_ess, min_ess, mean_rhat, max_rhat, mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw, elapsed_t]
    
    
    print("----" * 20)
    return list_param


def main(id, **specs):

    s = SimpleNamespace(**specs)

    #! ##################################################################################################################################
    SAVE_PATH = f'./results/{s.method}/{s.dataset}/{s.pre_learn}'
    MAP_PATH = f'./results/map_new/{s.dataset}/{s.pre_learn}/map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'

    if s.method == 'sghmc' or s.method == 'sharma_fix':
        filename = f'{s.method}_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'layer_fix':
        filename = f'{s.method}{s.num_fix_layer}_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'abs_fix' or s.method == 'row_fix':
        filename = f'{s.max_min}_{s.method}_{s.n_hidden}h{s.n_units}_id{id}.csv'
    else:
        assert(False)

    ensure_dir(dir=SAVE_PATH)
    #! ##################################################################################################################################

    tmp_file_identifier = f"tmp_{s.dataset}_{id}_{s.max_min}_{s.method}_{s.n_hidden}_{s.n_units}_{s.num_fix_layer}_{s.pre_learn}_{s.train_eval}"
    print("tmp_file_identifier = ", tmp_file_identifier)


    total_list = []
    if s.pre_learn == 'cv':
        
        header = ['lr', 'prior_var', 'train_acc', 'val_acc', 'test_acc', 'train_nll', 'val_nll', 'test_nll',
                    'mean_ess', 'min_ess', 'mean_rhat', 'max_rhat', 'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw','elapsed_t']

        for lr, prior_var in product(LR, PRIOR_VAR):
            print(f'lr = {lr}, prior_var = {prior_var}')
            list_param = sghmc(id, MAP_PATH, tmp_file_identifier, lr, prior_var, **specs)
            total_list.append(list_param)
        
    elif s.pre_learn == 'opt':
        header = ['lr', 'train_acc', 'val_acc', 'test_acc', 'train_nll', 'val_nll', 'test_nll',
                    'mean_ess', 'min_ess', 'mean_rhat', 'max_rhat', 'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw', 'elapsed_t']
        for lr in LR:
            print(f'lr = {lr}')
            list_param = sghmc(id, MAP_PATH, tmp_file_identifier, lr, prior_var=None, **specs)
            total_list.append(list_param)

    save_dir = os.path.join(SAVE_PATH, filename)

    with open(save_dir, 'w') as f:
       writer = csv.writer(f)
       writer.writerow(header)
       writer.writerows(total_list)
    
    path = f'./{tmp_file_identifier}'
    if os.path.isdir(path):
        shutil.rmtree(path)

    return

# Example:
# python ./code/classification_sghmc.py --method=row_fix --max_min=max --dataset htru2_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1
# python ./code/classification_sghmc.py --method=sghmc --dataset htru2_gap --n_hidden=1 --pre_learn=cv --train_eval=train --id=-1

if __name__ == '__main__':

    # Setup directories
    parser = ArgumentParser()
    parser.add_argument('--act_fn', type=str, choices=['tanh', 'relu'], default='relu')
    parser.add_argument('--dataset', type=str, default='eeg') # name of dataset
    parser.add_argument('--fix_bias', type=str, choices=['t', 'f'], default='f')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--max_min', type=str, choices=['max', 'min'], default='max')
    parser.add_argument('--method', type=str, choices=['sghmc', 'layer_fix', 'abs_fix', 'row_fix', 'sharma_fix'], default='sghmc')
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--num_fix_layer', type=int, default=1)
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv') # 事前学習した事前分布の使用の有無
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--one_by_one', type=str, choices=['yes', 'no'], default='no')
    
    args = parser.parse_args()

    if args.dataset.startswith("miniboo"):
        n_units = 200
        NUM_BURN_IN_KEEP_EVERY = 5000
    else:
        n_units = 100
        NUM_BURN_IN_KEEP_EVERY = 2000


    foldId_specifier = args.id

    assert(args.act_fn == "relu")

    print("foldId_specifier = ", foldId_specifier)
    all_fold_ids = commons.get_all_fold_ids(args.dataset, foldId_specifier)

    print("all_fold_ids = ", all_fold_ids)

    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None

    if args.one_by_one == "no":
        with multiprocessing.Pool(processes=all_fold_ids.shape[0]) as pool:
            pool.map(partial(main, dataset=args.dataset, train_eval=args.train_eval, n_hidden=args.n_hidden, n_units=n_units, act_fn = args.act_fn, device = args.device, pre_learn = args.pre_learn, fix_bias = bool(strtobool(args.fix_bias)), n_gpu_use = n_gpu_use, max_min = args.max_min, method = args.method, num_fix_layer = args.num_fix_layer, NUM_BURN_IN_KEEP_EVERY = NUM_BURN_IN_KEEP_EVERY), all_fold_ids)
    else:
        print("WARNING RUN ONE BY ONE (MIGHT BE SLOW)")
        for id in all_fold_ids:
            main(id, dataset=args.dataset, train_eval=args.train_eval, n_hidden=args.n_hidden, n_units=n_units, act_fn = args.act_fn, device = args.device, pre_learn = args.pre_learn, fix_bias = bool(strtobool(args.fix_bias)), n_gpu_use = n_gpu_use, max_min = args.max_min, method = args.method, num_fix_layer = args.num_fix_layer, NUM_BURN_IN_KEEP_EVERY = NUM_BURN_IN_KEEP_EVERY)
    
    print('実験終了!!')
    
    


