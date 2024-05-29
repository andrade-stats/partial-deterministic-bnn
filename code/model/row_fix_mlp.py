import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.mlp import MLP


class Row_Fix_Linear(nn.Module):
    def __init__(self, n_in, n_out, fc1_W = None, num_fix_ratio = None, descending = None):
        super(Row_Fix_Linear, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        # Initialize the parameters
        if fc1_W is None:
            self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        else:
            self.num_fix = int(self.n_out * num_fix_ratio)
            self.W = nn.Parameter(torch.zeros(self.n_in, self.num_fix), True)

        self.b = nn.Parameter(torch.zeros(self.n_out), True)
        
        #! 事前学習したMAPのモデル
        self.fc1_W = fc1_W
        if fc1_W is not None:
            self.full_fc1_W = torch.ones_like(self.fc1_W) #! 計算に使うWの入れ物
            print("self.fc1_W.device = ", self.fc1_W.device)
            
            print("self.fc1_W.get_device() = ", self.fc1_W.get_device())
            self.full_fc1_W = self.full_fc1_W.to(self.fc1_W.get_device()) # move to same device as fc1_W

            norm_W = torch.norm(fc1_W, dim=0) #! L2正則化 n_units=10の時, norm_W.shape = (10)
            
            _, indices = torch.sort(norm_W, descending=descending)
            indices = indices[0:self.num_fix]
            self.full_fc1_W[:, indices] = self.fc1_W[:, indices]
            
            self.sampling_indices = (self.full_fc1_W == 1.0)
            # print("self.required_true_indices = ", self.required_true_indices)
            # print("self.required_true_indices,shape = ", self.required_true_indices.shape)
            # assert(False)

        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.
        init.normal_(self.W, 0, std)
        init.constant_(self.b, 0)

    def forward(self, X):
        if self.fc1_W is not None:
            # print("----")
            # print("self.sampling_indices = ", self.sampling_indices)
            # print("self.full_fc1_W = ", self.full_fc1_W)
            # print("self.W = ", self.W)
            # assert(False)
            new_W = self.full_fc1_W.detach()
            new_W[self.sampling_indices] = torch.flatten(self.W)
        else:
            new_W = self.W

        new_W = new_W / math.sqrt(self.n_in)
        b = self.b
        
        return torch.mm(X, new_W) + b


class Row_Fix_MLP(MLP):

    def __init__(self, input_dim, output_dim, hidden_dims, act_fn, init_param, task, fixed_W_list, num_fix_ratio, descending):
        super().__init__(input_dim, output_dim, hidden_dims, act_fn, init_param, task)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.task = task
        options = {'tanh': torch.tanh, 'relu': F.relu}
        self.act_fn = options[act_fn]

        self.layers = nn.ModuleList([Row_Fix_Linear(input_dim, hidden_dims[0], fixed_W_list[0], num_fix_ratio, descending)])
        self.norm_layers = nn.ModuleList([nn.Identity()])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                f'linear_{i}', Row_Fix_Linear(hidden_dims[i-1], hidden_dims[i], fixed_W_list[i], num_fix_ratio, descending)
            )
            self.norm_layers.add_module(
                f'norm_{i}', nn.Identity()
            )
        self.output_layer = Row_Fix_Linear(hidden_dims[-1], output_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Row_Fix_Linear):
                m.reset_parameters()
