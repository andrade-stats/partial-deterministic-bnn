import math
import os
from argparse import ArgumentParser
from distutils.util import strtobool

import numpy as np
import torch
from utilities.util import ensure_dir, from_torch_to_numpy, load_uci_data

from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.gp import kernels, mean_functions, priors
from optbnn.gp.models.gpr import GPR
from optbnn.prior_mappers.wasserstein_mapper import (MapperWasserstein,
                                                     weights_init)
from optbnn.utils import util
from optbnn.utils.exp_utils import get_input_range
from optbnn.utils.normalization import normalize_data
from optbnn.utils.rand_generators import MeasureSetGenerator

SEED = 123

parser = ArgumentParser()
parser.add_argument('--act_fn', type=str, choices=['tanh', 'relu'], default='relu')
parser.add_argument('--dataset', type=str, default='yacht')
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--light_expt', type=str, choices=['y', 'n'], default='n')
parser.add_argument('--n_hidden', type=int, default=1)
parser.add_argument('--n_units', type=int, default=100)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()

act_fn = args.act_fn
dataset = args.dataset
id = args.id
light_expt = bool(strtobool(args.light_expt))
n_hidden = args.n_hidden
n_units = args.n_units

if dataset == 'protein':
    n_units = 200
else:
    assert(n_units == 100)

device = torch.device(args.device)

if args.device == "cuda":
    USE_GPU_GP = True
    N_GPU = 1
else:
    USE_GPU_GP = False
    N_GPU = 0

# Optimize the prior
num_iters = 200 # ワッサースタイン最適化のイタレーションの数 
lr = 0.05 # 学習率
n_samples = 128 # ミニバッチサイズ

class NewMapperWasserstein(MapperWasserstein):
    def __init__(self, gp, bnn, data_generator, out_dir,
                input_dim=1, output_dim=1, n_data=256, 
                wasserstein_steps=..., wasserstein_lr=0.01, 
                wasserstein_thres=0.01, logger=None, n_gpu=0, 
                gpu_gp=False, lipschitz_constraint_type="gp"):
        super().__init__(gp, bnn, data_generator, out_dir, 
                        input_dim, output_dim, n_data, wasserstein_steps, 
                        wasserstein_lr, wasserstein_thres, logger, n_gpu, 
                        gpu_gp, lipschitz_constraint_type)
        
    def optimize(self, num_iters, n_samples=128, lr=1e-2,
                 save_ckpt_every=50, print_every=10, debug=False, ckpt_fname=None):
        wdist_hist = []

        # print("self.device = ", self.device)
        # assert(False)

        wasserstein_steps = self.wasserstein_steps
        prior_optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr=lr)

        # Prior loop
        for it in range(1, num_iters+1):
            # Draw X
            X = self.data_generator.get(self.n_data)
            X = X.to(self.device)
            if not self.gpu_gp:
                X = X.to("cpu")

            # Draw functions from GP
            gp_samples = self.gp.sample_functions(
                X.double(), n_samples).detach().float().to(self.device)
            if self.output_dim > 1:
                gp_samples = gp_samples.squeeze()

            if not self.gpu_gp:
                X = X.to(self.device)

            # Draw functions from BNN
            nnet_samples = self.bnn.sample_functions(
                X, n_samples).float().to(self.device)
            if self.output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            ## Initialisation of lipschitz_f
            self.wasserstein.lipschitz_f.apply(weights_init)

            # Optimisation of lipschitz_f
            self.wasserstein.wasserstein_optimisation(X, 
                n_samples, n_steps=wasserstein_steps[1],
                threshold=self.wasserstein_threshold, debug=debug)
            prior_optimizer.zero_grad()


            wdist = self.wasserstein.calculate(nnet_samples, gp_samples)
            wdist.backward()
            prior_optimizer.step()

            wdist_hist.append(float(wdist))
            if (it % print_every == 0) or it == 1:
                self.print_info(">>> Iteration # {:3d}: "
                                "Wasserstein Dist {:.4f}".format(
                                    it, float(wdist)))

            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, ckpt_fname)
                torch.save(self.bnn.state_dict(), path)

        # Save accumulated list of intermediate wasserstein values
        if debug:
            values = np.array(self.wasserstein.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "wsr_intermediate_values.log")
            np.savetxt(path, values, fmt='%.6e')
            self.print_info('Saved intermediate wasserstein values in: ' + path)

        return wdist_hist


def optimize_prior(noise_var, saved_dir):
    util.set_seed(SEED)
    # saved_dir = os.path.join(saved_dir, dataset)
    print(f"<<<noise_var : {noise_var}>>> Loading split {id} of {dataset} dataset")
    # データセットの読み込み
    X_train, y_train = load_uci_data('train', id, dataset, device)
    X_val, y_val = load_uci_data('val', id, dataset, device)
    X_test, y_test = load_uci_data('test', id, dataset, device)
    
    X_train = torch.vstack([X_train, X_val])
    y_train = torch.vstack([y_train, y_val])
    
    # tensorからnumpyに変更
    X_train, y_train = from_torch_to_numpy(X_train, y_train)
    X_test, y_test = from_torch_to_numpy(X_test, y_test)
    
    X_train_, y_train_, X_test_, _, _, _ = normalize_data(
        X_train, y_train, X_test, y_test)
    x_min, x_max = get_input_range(X_train_, X_test_)
    input_dim, output_dim = int(X_train.shape[-1]), 1
    
    # Initialize the measurement set generator
    rand_generator = MeasureSetGenerator(X_train_, x_min, x_max, 0.7)
    
    # Initialize the mean and covariance function of the target hierarchical GP prior
    mean = mean_functions.Zero()
    
    lengthscale = math.sqrt(2 * input_dim)
    variance = 1.
    kernel = kernels.RBF(input_dim=input_dim,
                        lengthscales=torch.tensor([lengthscale], dtype=torch.double),
                        variance=torch.tensor([variance], dtype=torch.double), ARD=True)
    
    # Place hyper-priors on lengthscales and variances
    kernel.lengthscales.prior = priors.LogNormal(
                torch.ones([input_dim]) * math.log(lengthscale),
                torch.ones([input_dim]) * 1.)
    kernel.variance.prior = priors.LogNormal(
            torch.ones([1]) * 0.1,
            torch.ones([1]) * 1.)   
    
    # Initialize the GP model
    gp = GPR(X=torch.from_numpy(X_train_), Y=torch.from_numpy(y_train_).reshape([-1, 1]),
            kern=kernel, mean_function=mean)
    gp.likelihood.variance.set(noise_var)    

    # Initialize tunable MLP prior
    hidden_dims = [n_units] * n_hidden
    mlp_reparam = GaussianMLPReparameterization(input_dim, output_dim,
        hidden_dims, act_fn, scaled_variance=True)
    
    ckpt_fname = f'it-{num_iters}_nvar{noise_var}_{n_hidden}h{n_units}_id{id}.ckpt'
    mapper = NewMapperWasserstein(gp, mlp_reparam, rand_generator, out_dir=saved_dir,
                            output_dim=output_dim, n_data=100,
                            wasserstein_steps=(0, 200),
                            wasserstein_lr=0.02,
                            logger=None, wasserstein_thres=0.1,
                            n_gpu=N_GPU, gpu_gp=USE_GPU_GP)
    
    w_hist = mapper.optimize(num_iters=num_iters, n_samples=n_samples,
                            lr=lr, print_every=10, save_ckpt_every=num_iters, debug=True, ckpt_fname=ckpt_fname)
    print("----" * 20)


def main():
    if light_expt:
        NOISE_VAR = [1.0]
    else:
        NOISE_VAR = [1.0, 0.5, 0.1, 0.01]
    save_dir = './results/prelearn_pvar'
    SAVED_DIR = os.path.join(save_dir, dataset)
    
    ensure_dir(SAVED_DIR)
    
    for noise_var in NOISE_VAR:
        optimize_prior(noise_var, SAVED_DIR)


if __name__ == '__main__':
    main()
