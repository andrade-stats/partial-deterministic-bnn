import errno
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
from arviz.stats.diagnostics import ess, rhat

from optbnn.bnn.likelihoods import LikCategorical
from optbnn.utils.normalization import zscore_normalization, zscore_unnormalization

# lossの計算
def log_likelihood_func(model, X, y, noise_std, device, task='regression'):
    mu = model(X)
    if task == 'regression':
        y = y.view(-1, 1)
        std = torch.ones(y.size()[0], device=device) * noise_std
        std = std.view(-1, 1)
        assert mu.size() == std.size() 
        assert mu.size() == y.size()
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(y).sum()
    elif task == 'classification':
        loss = torch.nn.NLLLoss(reduction='sum')
        return -loss(mu, y)
    

def gaussian_nll(y, mu, var):
    """
    正規分布における負の対数尤度(Negative Log-Likelihood, NLL)を計算する。
    
    Args:
        y: torch.Tensor, 真の値。
        mu: torch.Tensor, 予測値。
        var: float, 予測値の分散。
        
    Returns: 
        nll: float, 正規分布の負の対数尤度の値
    """
    y = y.view(-1, 1)
    assert y.shape == mu.shape
    std = math.sqrt(var)
    std = torch.ones_like(y) * std
    assert y.shape == std.shape
    dist = torch.distributions.Normal(mu, std)
    nll = - dist.log_prob(y).mean()
    
    return nll.item()
    

def log_joint(model, X, y, noise_var, prior_var, device, task, prior):        
    # yの対数尤度
    if task == 'regression':
        noise_std = math.sqrt(noise_var)
        logpdf_w = log_likelihood_func(model, X, y, noise_std, device, task)
    elif task == 'classification':
        fx = model.forward(X, log_softmax=True)    
        likelihood = LikCategorical()
        logpdf_w = likelihood.loglik(fx=fx, y=y)
    
    if prior == "no_prior":
        log_theta_prior_in = 0.0
    else:
        if prior is None:
            prior_std = math.sqrt(prior_var)
            prior_dist_in = torch.distributions.Normal(0, prior_std) #! mean, standard deviation
            log_theta_prior_in = 0.0
            for p in model.parameters():
                log_theta_prior_in += prior_dist_in.log_prob(p).sum() 
        else:
            log_theta_prior_in = prior.logp(model)
        
    return logpdf_w + log_theta_prior_in


# tensorをnp.arrayに変換
def from_torch_to_numpy(X, y):
    X = X.to('cpu').detach().numpy().copy()
    y = y.to('cpu').detach().numpy().copy()
    return X, y


def search_best(path):
    df = pd.read_csv(path)
    header = df.columns
    
    if ('prior_var' not in header) and ('noise_var' not in header):
        extraction_ls = ['lr', 'val_nll']
    elif ('noise_var' not in header):
        extraction_ls = ['lr', 'prior_var', 'val_nll']
    elif ('prior_var' not in header):
        extraction_ls = ['lr', 'noise_var', 'val_nll']
    else:
        extraction_ls = ['lr', 'prior_var', 'noise_var', 'val_nll']
    
    # print("extraction_ls = ", extraction_ls)
    
    new_df = df.loc[:, extraction_ls] # 特定の列を抽出

    values = new_df.values
    # print("values = ", values)
    idx = np.nanargmin(values[:, -1]) # 最小のval_nllのインデックス
    best_param = values[idx, 0:-1]
    
    return best_param


def round_func(list, number=5):
    return [round(x, number) for x in list]


def ensure_dir(dir):
    """
    指定されたディレクトリが存在するか確認し、存在しない場合は新しく作成する。
    
    Args: 
        dir: str, 確認するディクレトリのパス。
        
    Returns:
        None
    """
    if not os.path.isdir(dir): # 指定されたディレクトリが存在しない場合
        try:
            os.makedirs(dir) # 新しいディレクトリを作成
        except OSError as ex:
            if ex.errno == errno.EEXIST and os.path.isdir(dir):
                pass # ディレクトリがすでに存在する場合は何もしない
            else:
                raise


def ess_rhat_func(samples, n_chains):
    """
    サンプルの有効サイズ(Effective Sample Size; ESS)とGelman-RubinのRハット統計量を計算する関数。
    
    Args: 
        samples: サンプルの行列
        n_chains: チェーンの数
        
    Returns:
        ess_mean: ESSの平均値
        ess_min: ESSの最小値
        rhat_mean: Rハットの平均値
        rhat_max: Rハットの最大値
    """
    
    # 不要な警告を無視
    warnings.simplefilter('ignore')
    
    # サンプル行列のサイズを取得
    n_total_samples = samples.shape[0]
    n_vars = samples.shape[1]
    
    # 各チェーンごとのサンプルに再形成
    samples_each_chain = samples.reshape(n_chains, n_total_samples//n_chains, n_vars)
    
    # 各次元ごとのESSとRハットを計算
    n_ess_each_dim = np.zeros(n_vars)
    rhat_each_dim = np.zeros(n_vars)
    for i in range(n_vars):
        n_ess_each_dim[i] = ess(samples_each_chain[:, :, i])
        rhat_each_dim[i] = rhat(samples_each_chain[:, :, i], method="rank") # "split" is the ordinary Gelman-Rubin R-Hat, "rank" is the new one proposed bei Aki Vehtari
    
    # ESSとRハットの統計量を計算
    ess_mean = round(np.nanmean(n_ess_each_dim), 5)
    ess_min = round(np.nanmin(n_ess_each_dim), 5)
    rhat_mean = round(np.nanmean(rhat_each_dim), 5)
    rhat_max = round(np.nanmax(rhat_each_dim), 5)
    
    return ess_mean, ess_min, rhat_mean, rhat_max


def load_uci_data(usage, id, name, device):
    """
    UCIデータセットをロードする回数。
    
    Args:
        usage: str, 使用目的を指定('train', 'val', 'test')
        id:, int, データセットのインデックス
        name: str, データセットの名前
        device: torch.device, データを配置するデバイス
        
    Returns:
        X: torch.Tensor, 特徴量データ
        y: torch.Tensor, ターゲットデータ
    """
    
    all_usage_list = ['train', 'val', 'test']
    
    if not(usage in all_usage_list):
        raise ValueError("Invalid usage name")

    if "gap" in name:
        data_path = f'./gap_data/{name}/{id}/'
    else:
        data_path = f'./data/{name}/{id}/'

    X = pd.read_csv(data_path + f'{usage}_X.csv', header=None)
    y = pd.read_csv(data_path + f'{usage}_y.csv', header=None)

    X = torch.tensor(np.array(X).astype('f'), device=device)
    y = torch.tensor(np.array(y).astype('f'), device=device)
    
    return X, y



class DataNormalizer():

    def __init__(self, X_train, y_train = None):
        self.X_train, self.X_mean, self.X_std = zscore_normalization(X_train)

        if y_train is not None:
            self.y_train, self.y_mean, self.y_std = zscore_normalization(y_train)

    def zscore_unnormalize_x(self, x):
        return zscore_unnormalization(x, self.X_mean, self.X_std)
    
    def zscore_unnormalize_y(self, y):
        return zscore_unnormalization(y, self.y_mean, self.y_std)
    
    def zscore_normalize_x(self, x):
        normalized_values, _, _ = zscore_normalization(x, self.X_mean, self.X_std)
        return normalized_values
    
    def zscore_normalize_y(self, y):
        normalized_values, _, _ = zscore_normalization(y, self.y_mean, self.y_std)
        return normalized_values
    


def normalize_x_only(data):
    normalizer = DataNormalizer(data)
    return normalizer.X_train


def show_data_ratios(train_y, val_y, test_y):
    n = train_y.shape[0] + val_y.shape[0] + test_y.shape[0]
    print("n = ", n)
    train_val_ratio = ((train_y.shape[0] + val_y.shape[0]) / n) * 100
    print("train_val ratio = ", train_val_ratio)
    assert(train_val_ratio > 88 and train_val_ratio < 92)

    val_ratio_to_train_val =  (val_y.shape[0] / (train_y.shape[0] + val_y.shape[0])) * 100
    print("val val_ratio_to_train_val = ", val_ratio_to_train_val)
    assert(val_ratio_to_train_val > 18 and val_ratio_to_train_val < 22)
    return
