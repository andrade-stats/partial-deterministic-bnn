import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.mlp import MLP

torch.manual_seed(123)
import pandas as pd
import math

# sample only weights that have the largest absolute value of MAP solution
class Sharma_Fix_Linear(nn.Module):
    def __init__(self, n_in, n_out, fc1_W = None, num_fix_ratio = 0.0):
        super(Sharma_Fix_Linear, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        
        # Initialize the parameters
        if fc1_W is None:
            self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        else:
            assert(num_fix_ratio == 0.5)
            abs_weights = torch.abs(fc1_W)
            abs_med = torch.median(torch.flatten(abs_weights))
            self.sampling_indices = (abs_weights > abs_med)
            nr_random_weights = torch.sum(self.sampling_indices)
            
            # print("nr_random_weights = ", nr_random_weights)
            # print("torch.flatten(abs_weights).shape[0] = ", torch.flatten(abs_weights).shape[0])
            assert(math.fabs(nr_random_weights - torch.flatten(abs_weights).shape[0] / 2) < 5)  # nr_random_weights should be rougly half of all parameters W
            
            self.W = nn.Parameter(torch.zeros(nr_random_weights), True)

            assert(isinstance(self.W, torch.FloatTensor) or isinstance(self.W, torch.cuda.FloatTensor))
            
        self.b = nn.Parameter(torch.zeros(self.n_out), True)
        
        #! 事前学習したMAPのモデル
        self.fc1_W = fc1_W
     
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.
        init.normal_(self.W, 0, std)
        init.constant_(self.b, 0)

    def forward(self, X):
        if self.fc1_W is not None:
            new_W = self.fc1_W.detach()
            new_W[self.sampling_indices] = self.W
        else:
            new_W = self.W
        
        new_W = new_W / math.sqrt(self.n_in)
        b = self.b
        
        return torch.mm(X, new_W) + b


class Sharma_Fix_Linear_MLP(MLP):
    
    def __init__(self, input_dim, output_dim, hidden_dims, act_fn, init_param, task, fixed_W_list, num_fix_ratio):
        super().__init__(input_dim, output_dim, hidden_dims, act_fn, init_param, task)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.task = task
        options = {'tanh': torch.tanh, 'relu': F.relu}
        self.act_fn = options[act_fn]

        self.layers = nn.ModuleList([Sharma_Fix_Linear(input_dim, hidden_dims[0], fixed_W_list[0], num_fix_ratio)])
        self.norm_layers = nn.ModuleList([nn.Identity()])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                f'linear_{i}', Sharma_Fix_Linear(hidden_dims[i-1], hidden_dims[i], fixed_W_list[i], num_fix_ratio)
            )
            self.norm_layers.add_module(
                f'norm_{i}', nn.Identity()
            )
        self.output_layer = Sharma_Fix_Linear(hidden_dims[-1], output_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Sharma_Fix_Linear):
                m.reset_parameters()
