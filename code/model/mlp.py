import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, n_in, n_out, init_param):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(Linear, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.init_param = init_param

        # Initialize the parameters
        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        self.b = nn.Parameter(torch.zeros(self.n_out), True)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if self.init_param == 'norm':
            init.normal_(self.W, 0, std)
            init.constant_(self.b, 0)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W
        W = W / math.sqrt(self.n_in)
        b = self.b
        return torch.mm(X, W) + b


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, act_fn, init_param, task):
        """
        多層パーセプトロン(MLP)の初期化。

        Args:
            input_dim: int, 入力データのサイズ。
            output_dim: int, 出力データのサイズ。
            hidden_dims: intのリスト, 隠れ層のサイズを含むリスト。
            act_fn: str, ネットワークで使用する活性化関数の名前。
            init_param: str, 線形層の初期パラメータ。
            task: str, タスクのタイプ, 'regression'または'classification'である必要ある。
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.init_param = init_param
        self.task = task
        
        # 活性化関数のオプション
        options = {'tanh': torch.tanh, 'relu': F.relu}
        self.act_fn = options[act_fn]

        # レイヤーの定義
        self.layers = nn.ModuleList([Linear(input_dim, hidden_dims[0], init_param)])
        self.norm_layers = nn.ModuleList([nn.Identity()])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                f'linear_{i}', Linear(hidden_dims[i-1], hidden_dims[i], init_param)
            )
            self.norm_layers.add_module(
                f'norm_{i}', nn.Identity()
            )
        self.output_layer = Linear(hidden_dims[-1], output_dim, init_param)

    def reset_parameters(self):
        """すべての線形層のパラメータをリセット"""
        for m in self.modules():
            if isinstance(m, Linear):
                m.reset_parameters()

    def forward(self, X, log_softmax=False):
        """
        入力データを与えて順方向の処理を行う。
        
        Args:
            X: torch.tensor, [batch_size, input_dim], 入力データ。
            log_softmax: bool, log softmax値を返すかどうかを示す。
                

        Returns:
            torch.tensor, [batch_size, output_dim], 出力データ。
        """
        X = X.reshape(-1, self.input_dim)
        for linear_layer, norm_layer in zip(list(self.layers), list(self.norm_layers)):
            X = self.act_fn(norm_layer(linear_layer(X)))
            
        X = self.output_layer(X)
        
        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X


    def predict(self, X):
        """
        入力データを与えて予測を行います。

        Args:
            x: torch tensor, shape [batch_size, input_dim]

        Returns:
            torch tensor, shape [batch_size, num_classes], 各クラスの予測確率
        """
        self.eval()
        if self.task == "classification":
            return self.forward(X, log_softmax=True) # log_prob        
        else:
            return self.forward(X, log_softmax=False)




