import math
import os
from argparse import ArgumentParser
from distutils.util import strtobool

import numpy as np
import torch
import torch.utils.data as data_utils
from utilities.util import from_torch_to_numpy, load_uci_data

from optbnn.bnn.reparam_nets import GaussianMLPReparameterization
from optbnn.gp import kernels, mean_functions, priors
from optbnn.gp.models.gpr import GPR
from optbnn.prior_mappers.wasserstein_mapper import (WassersteinDistance,
                                                     weights_init)
from optbnn.utils import util
from optbnn.utils.rand_generators import ClassificationGenerator

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['htru2', 'eeg', 'letter', 'miniboo'], default='miniboo')
parser.add_argument('--id', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=1)
parser.add_argument('--n_units', type=int, default=100)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()

dataset = args.dataset
id = args.id
n_hidden = args.n_hidden
n_units = args.n_units

if dataset == 'miniboo':
    n_units = 200
else:
    assert(n_units == 100)

class NewMapperWasserstein(object):
    def __init__(self, gp, bnn, data_generator, out_dir,
                 input_dim=1, output_dim=1, n_data=256,
                 wasserstein_steps=(200, 200), wasserstein_lr=0.01, wasserstein_thres=0.01, 
                 logger=None, n_gpu=0, gpu_gp=False, lipschitz_constraint_type="gp"):
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_dir = out_dir
        self.device, device_ids = util.prepare_device(n_gpu)
        
        self.gpu_gp = gpu_gp

        assert lipschitz_constraint_type in ["gp", "lp"]
        self.lipschitz_constraint_type = lipschitz_constraint_type

        if type(wasserstein_steps) != list and type(wasserstein_steps) != tuple:
            wasserstein_steps = (wasserstein_steps, wasserstein_steps)
        self.wasserstein_steps = wasserstein_steps
        self.wasserstein_threshold = wasserstein_thres

        # Move models to configured device
        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        # Initialize the module of wasserstance distance
        self.wasserstein = WassersteinDistance(
            self.bnn, self.gp,
            self.n_data, output_dim=self.output_dim,
            wasserstein_lr=wasserstein_lr, device=self.device,
            gpu_gp=self.gpu_gp,
            lipschitz_constraint_type=self.lipschitz_constraint_type)

        # Setup logger
        self.print_info = print if logger is None else logger.info

        print("self.device = ", self.device)
        assert(False)

        # Setup checkpoint directory
        # self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        # util.ensure_dir(self.ckpt_dir)

    def optimize(self, num_iters, n_samples=128, lr=1e-2,
                 save_ckpt_every=50, print_every=10, debug=False):
        wdist_hist = []

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
                path = os.path.join(self.out_dir, f'it-{it}_{n_hidden}h{n_units}_id{id}.ckpt')
                torch.save(self.bnn.state_dict(), path)

        # Save accumulated list of intermediate wasserstein values
        if debug:
            values = np.array(self.wasserstein.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "wsr_intermediate_values.log")

        return wdist_hist

util.set_seed(123)
OUT_DIR = f'./results/prelearn_pvar/{dataset}'

device = torch.device(args.device)

if args.device == "cuda":
    USE_GPU_GP = True
    N_GPU = 1
else:
    USE_GPU_GP = False
    N_GPU = 0

X, y = load_uci_data('train', id, name=dataset, device=device)
X, y = from_torch_to_numpy(X, y)
X = X.astype(np.float32)
y = y.reshape([-1]).astype(np.int64)
print(X.shape, y.shape)

# Setup directories
ckpt_dir = os.path.join(OUT_DIR, "ckpts")
util.ensure_dir(ckpt_dir)

#! Configure hyper-parameters
input_dim = X.shape[1]
output_dim = np.unique(y).shape[0]

hidden_dims = [n_units] * n_hidden

activation_fn = "relu"

mapper_batch_size = 256        # The size of the measurement set. The measurement points are sampled from the training data.
mapper_n_samples = 128         # The size of mini batch used in Wasserstein optimization
mapper_n_lipschitz_iters = 200 # The number of Lipschitz function iterations per prior iteration
mapper_n_prior_iters = 200     # The number of prior iterations
mapper_lipschitz_lr = 0.02     # The learning rate for the opimization of the Lipschitz function (inner loop)
mapper_prior_lr = 0.05         # The learning rate for the optimization of the prior (outer loop)

# Initialize data loader for the mapper
data_loader = data_utils.DataLoader(
                    data_utils.TensorDataset(torch.from_numpy(X),
                                             torch.from_numpy(y)),
                    batch_size=mapper_batch_size, shuffle=True)

# Setup the measurement set generator
# We draw measurement points from the training data
rand_generator = ClassificationGenerator(data_loader)

# Specify the target GP prior
lengthscale = math.sqrt(2. * input_dim)
variance = 8.

X_, y_ = rand_generator.get(return_label=True)
kernel = kernels.RBF(
    input_dim,
    lengthscales=torch.tensor([lengthscale], dtype=torch.double),
    variance=torch.tensor([variance], dtype=torch.double),
    ARD=True)

kernel.lengthscales.prior = priors.LogNormal(
        torch.ones([input_dim]) * math.log(lengthscale),
        torch.ones([input_dim]) * 1.)

kernel.variance.prior = priors.LogNormal(
    torch.ones([1]) * math.log(variance),
    torch.ones([1]) * 0.3)

gp = GPR(X_.reshape([mapper_batch_size, -1]).double(),
                    util.to_one_hot(y_, num_classes=30).double(),
                    kern=kernel, mean_function=mean_functions.Zero())

# Initialize the Gaussian prior to optimize
mlp_reparam = GaussianMLPReparameterization(input_dim, output_dim,
    hidden_dims, activation_fn, scaled_variance=True)

# Initialize the mapper
# saved_dir = os.path.join(OUT_DIR, "hierarchical")
mapper = NewMapperWasserstein(gp, mlp_reparam, rand_generator, out_dir=ckpt_dir,
                           output_dim=output_dim,
                           n_data=mapper_batch_size,
                           wasserstein_steps=(0, mapper_n_lipschitz_iters),
                           wasserstein_lr=mapper_lipschitz_lr,
                           wasserstein_thres=0.1,
                           n_gpu=N_GPU, gpu_gp=USE_GPU_GP)

# Start optimization
print("Start optimizing prior")
w_hist = mapper.optimize(num_iters=mapper_n_prior_iters, n_samples=mapper_n_samples,
                lr=mapper_prior_lr, print_every=5,
                save_ckpt_every=200, debug=True)
print("----" * 20)