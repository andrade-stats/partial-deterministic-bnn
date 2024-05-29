import math

import torch
import torch.nn as nn
import torch.nn.init as init
from model.mlp import MLP


#? for absmap
class ABS_FIX_LINEAR(nn.Module):
    def __init__(self, n_in, n_out, pre_W = None, num_fix_ratio=0, max_min=None):
        super(ABS_FIX_LINEAR, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.pre_W = pre_W
        self.max_min = max_min
        
        if self.pre_W is None:
            self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True) 
        else:
            self.num_fix = int(self.n_in * num_fix_ratio)
            self.W = nn.Parameter(torch.zeros(self.n_in-self.num_fix, self.n_out), True)
            abs_W = torch.abs(self.pre_W)
            if max_min == 'max':
                descending = False
            elif max_min == 'min':
                descending = True
            else:
                assert(False)

            _, sorted_indices = torch.sort(abs_W, dim=0, descending=descending)

            random_param_indices = sorted_indices[0:(self.n_in-self.num_fix), :]

            # print("random_param_indices = ", random_param_indices.shape)
            # print("self.pre_W = ", self.pre_W.t())

            for column_id in range(self.n_out):
                self.pre_W[random_param_indices[:, column_id], column_id] = 1.0
            
            # print("self.pre_W = ", self.pre_W.t())
            # assert(False)

            self.sampling_indices = (self.pre_W == 1.0)

                
        self.b = nn.Parameter(torch.zeros(self.n_out), True)
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        init.normal_(self.W, 0, std)
        init.constant_(self.b, 0)

    def forward(self, X):
        if self.pre_W is not None:
            new_W = self.pre_W.detach()
            new_W[self.sampling_indices] = torch.flatten(self.W)
        else:
            new_W = self.W
        
        new_W = new_W / math.sqrt(self.n_in)
        b = self.b
        
        return torch.mm(X, new_W) + b


class ABS_FIX_MLP(MLP):

    def __init__(self, input_dim, output_dim, hidden_dims, act_fn, init_param, task, fixed_W_list, num_fix_ratio, max_min):
        """絶対値が最大となる位置と値を固定するMLPの初期化。
        
        """
        super().__init__(input_dim, output_dim, hidden_dims, act_fn, init_param, task)

        self.layers = nn.ModuleList([ABS_FIX_LINEAR(input_dim, hidden_dims[0], pre_W=fixed_W_list[0], num_fix_ratio = num_fix_ratio, max_min = max_min)])
        self.norm_layers = nn.ModuleList([nn.Identity()])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                f'linear_{i}', ABS_FIX_LINEAR(hidden_dims[i-1], hidden_dims[i], pre_W=fixed_W_list[i], num_fix_ratio = num_fix_ratio, max_min = max_min)
            )
            self.norm_layers.add_module(
                f'norm_{i}', nn.Identity()
            )
        self.output_layer = ABS_FIX_LINEAR(hidden_dims[-1], output_dim)

    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ABS_FIX_LINEAR):
                m.reset_parameters()
                


        

