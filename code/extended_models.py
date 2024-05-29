from model.abs_fix_mlp import ABS_FIX_MLP
from model.layer_fix_mlp import Layer_Fix_MLP
from model.mlp import MLP
from model.row_fix_mlp import Row_Fix_MLP
from model.sharma_fix_mlp import Sharma_Fix_Linear_MLP

from utilities.util import search_best

import torch
import math


def get_fixed_weights(map_net, s):
    fc1_W = map_net.state_dict()['layers.0.W'].detach().clone()

    if s.n_hidden == 1:
        fixed_W_list = [fc1_W]
    else:
        assert(s.n_hidden == 2)
        fc2_W = map_net.state_dict()['layers.linear_1.W'].detach().clone()
        fixed_W_list = [fc1_W, fc2_W]

    return fixed_W_list




def move_to_cuda(net, s):
    if s.device == "cuda":
        current_device = torch.cuda.current_device()
        return net.to(current_device)
    else:  
        return net


def getModel(MAP_PATH, id, task, input_dim, output_dim, hidden_dims, s):

    assert(task == "classification" or task == "regression")
    
    num_fix_ratio = 0.5
    assert(s.act_fn == "relu")

    if s.method != 'sghmc':
        if task == "regression":
            if s.pre_learn == 'cv':
                map_lr, map_pvar, map_nvar = search_best(MAP_PATH)
                model_path = f'./results/map_new/{s.dataset}/{s.pre_learn}/lr{map_lr}_pvar{map_pvar}_nvar{map_nvar}_{s.n_hidden}h{s.n_units}_id{id}.pth' 
            elif s.pre_learn == 'opt':
                map_lr, map_nvar = search_best(MAP_PATH)
                model_path = f'./results/map_new/{s.dataset}/opt/lr{map_lr}_nvar{map_nvar}_{s.n_hidden}h{s.n_units}_id{id}.pth'
        else:
            if s.pre_learn == 'opt':
                map_lr = search_best(MAP_PATH)[0]
                model_path = f'./results/map_new/{s.dataset}/{s.pre_learn}/lr{map_lr}_{s.n_hidden}h{s.n_units}_id{id}.pth'
            elif s.pre_learn == 'cv':
                map_lr, map_pvar = search_best(MAP_PATH)
                model_path = f'./results/map_new/{s.dataset}/cv/lr{map_lr}_pvar{map_pvar}_{s.n_hidden}h{s.n_units}_id{id}.pth'


    if s.method == 'sghmc':
        net = MLP(input_dim, output_dim, hidden_dims, s.act_fn, init_param='norm', task=task)
    elif s.method == 'abs_fix':
        # CHECKED
        map_net = MLP(input_dim, output_dim, hidden_dims, 'relu', init_param='norm', task=task)
        map_net.load_state_dict(torch.load(model_path))
        map_net = move_to_cuda(map_net, s)
        fixed_W_list = get_fixed_weights(map_net, s)
        net = ABS_FIX_MLP(input_dim, output_dim, hidden_dims, s.act_fn, 'norm', task, fixed_W_list, num_fix_ratio, s.max_min) 
    elif s.method == 'row_fix':
        # CHECKED
        map_net = MLP(input_dim, output_dim, hidden_dims, 'relu', init_param='norm', task=task)
        map_net.load_state_dict(torch.load(model_path))
        map_net = move_to_cuda(map_net, s)
        fixed_W_list = get_fixed_weights(map_net, s)
        
        if s.max_min == 'max':
            descending = True
        elif s.max_min =='min':
            descending = False
        else:
            assert(False)
        
        net = Row_Fix_MLP(input_dim, output_dim, hidden_dims, 'relu', 'norm', task, fixed_W_list, num_fix_ratio, descending)

    elif s.method == "sharma_fix":
        # CHECKED
        map_net = MLP(input_dim, output_dim, hidden_dims, 'relu', init_param='norm', task=task)
        map_net.load_state_dict(torch.load(model_path))
        map_net = move_to_cuda(map_net, s)
        fixed_W_list = get_fixed_weights(map_net, s)
        net = Sharma_Fix_Linear_MLP(input_dim, output_dim, hidden_dims, 'relu', 'norm',  task, fixed_W_list, num_fix_ratio)
    elif s.method == 'layer_fix':
        # CHECKED
        net = Layer_Fix_MLP(input_dim, output_dim, hidden_dims, s.act_fn, task=task, n_fix_layer=s.num_fix_layer)
        net.load_state_dict(torch.load(model_path))
        net = move_to_cuda(net, s)

        for idx, (name, layer) in enumerate(net.named_children()):
            if idx == 0:
                assert(name == "layers") # contains all hidden layers
                for i, (param_name, param) in enumerate(layer.named_parameters()):
                    if s.num_fix_layer * 2 > i: # Wとbが交互に出てくるため*2をすることで、層ごとに処理を行う
                        param.requires_grad = False
    else:
        assert(False)

    
    net = move_to_cuda(net, s)

    # debug output
    # for idx, (name, layer) in enumerate(net.named_children()):
    #         if idx == 0:
    #             assert(name == "layers") # contains all hidden layers
    #             for i, (param_name, param) in enumerate(layer.named_parameters()):
    #                 print(f"{param_name} = {param.device}")
    # assert(False)

    return net


