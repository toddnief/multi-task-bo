import torch
import math

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode

from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# training constants
BO_LOOPS = 2
X_DIM = 3
N_TASKS = 2

# define the data
X = torch.rand(10, X_DIM).double()

for i in range(N_TASKS):
    Y = 1 - torch.norm(X - 0.5, dim=-1, keepdim=True)
    Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
    XX = torch.column_stack((X, torch.ones(len(X))*i)).double() # add feature column to train_X

    # create train_Y or stack it if train_Y already exists
    if i == 0:
        train_X = XX
        train_Y = standardize(Y).double()
    else:
        train_X = torch.cat((train_X, XX))
        train_Y = torch.cat((train_Y, standardize(Y))).double()

# define the feature extractor for the deep kernel
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.data_dim = data_dim

        self.add_module('linear1', torch.nn.Linear(self.data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

data_dim = train_X.size(-1)
feature_extractor = LargeFeatureExtractor(data_dim).double()

# define the deep kernel
class DeepMaternKernel(MaternKernel):
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2, diag=False, **params):
        x1 = self.feature_extractor(x1[:-1])
        x2 = self.feature_extractor(x2[:-1])
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )
    
deep_kernel = DeepMaternKernel(feature_extractor)

for _ in range(BO_LOOPS):
    # define the GP using the deep kernel
    gp = MultiTaskGP(train_X, train_Y, task_feature=-1, covar_module=deep_kernel)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # fit the GP model
    fit_gpytorch_mll(mll)

    # check to make sure weights of feature extractor are updating
    # for name, param in deep_kernel.feature_extractor.named_parameters():
    #     print(param)
    #     break

    # use the acquisition function to pick a point
    UCB = UpperConfidenceBound(gp, beta=0.1)
    bounds = torch.stack([torch.zeros(X_DIM), torch.ones(X_DIM)])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    print(candidate)  # tensor([0.4887, 0.5063])

    # generate arbitrary y value
    f_X = torch.tensor([candidate.sum()]).unsqueeze(0)

    # print updated parameters to make sure this is working
    train_X = torch.cat((train_X, candidate))
    train_Y = torch.cat((train_Y, f_X))