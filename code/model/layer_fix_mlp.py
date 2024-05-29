import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.mlp import MLP


#! 特定の層を固定するためのLinear
class Layer_Fix_Linear(nn.Module):
    def __init__(self, n_in, n_out, fix_flag):
        super(Layer_Fix_Linear, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out        
        self.fix_flag = fix_flag
        
        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        self.b = nn.Parameter(torch.zeros(self.n_out), True)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.fix_flag:
            init.normal_(self.W, 0, std)
            init.constant_(self.b, 0)

    def forward(self, X):  
        W = self.W
        W = W / math.sqrt(self.n_in)
        b = self.b
        return torch.mm(X, W) + b




class Layer_Fix_MLP(MLP):
    def __init__(self, input_dim, output_dim, hidden_dims, act_fn, task, n_fix_layer):
        super().__init__(input_dim, output_dim, hidden_dims, act_fn, task, n_fix_layer)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.task = task
        
        self.n_fix_layer = n_fix_layer
        
        options = {'tanh': torch.tanh, 'relu': F.relu}
        self.act_fn = options[act_fn]
        
        self.layers = nn.ModuleList([Layer_Fix_Linear(input_dim, hidden_dims[0], fix_flag=True)])
        self.norm_layers = nn.ModuleList([nn.Identity()])
        for i in range(1, len(hidden_dims)):
            fix_flag = False
            if n_fix_layer > i:
                fix_flag = True
            self.layers.add_module(
                f'linear_{i}', Layer_Fix_Linear(hidden_dims[i-1], hidden_dims[i], fix_flag=fix_flag)
            )
            self.norm_layers.add_module(
                f'norm_{i}', nn.Identity()
            )
        self.output_layer = Layer_Fix_Linear(hidden_dims[-1], output_dim, fix_flag=False)

 
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Layer_Fix_Linear):
                m.reset_parameters()