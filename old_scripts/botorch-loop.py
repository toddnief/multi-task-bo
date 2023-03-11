from math import floor
from scipy.linalg import cholesky
from scipy.io import loadmat
import numpy as np

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf

SMOKE_TEST = True
SMOKE_TRAIN = 100
SMOKE_CANDIDATES = 1000
SMOKE_BO_LOOPS = 20

data = torch.Tensor(loadmat('elevators.mat')['data']).double()
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y1 = standardize(data[:, -1])

# generate correlated second task using Cholesky decomposition
covar = np.array([[1.0, .8], 
                  [.8, 1.0]])

lower_chol = torch.from_numpy(cholesky(covar).T)
yy = torch.randn(len(y1)) # generate random normal values
y2 = (lower_chol @ torch.vstack((y1,yy)))[1] # extract correlated targets

train_n = int(floor(0.8 * len(X))) if not SMOKE_TEST else SMOKE_TRAIN
candidates_n = len(X) - int(floor(0.8 * len(X))) if not SMOKE_TEST else SMOKE_CANDIDATES

train_x = X[:train_n, :].contiguous()
train_y1 = y1[:train_n].contiguous().unsqueeze(1)
train_y2 = y2[:train_n].contiguous().unsqueeze(1)

test_x = X[candidates_n:, :].contiguous() if not SMOKE_TEST else X[-train_n:, :].contiguous()
test_y1 = y1[candidates_n:].contiguous().unsqueeze(1) if not SMOKE_TEST else y1[-train_n:].contiguous().unsqueeze(1)
test_y2 = y2[candidates_n:].contiguous().unsqueeze(1) if not SMOKE_TEST else y2[-train_n:].contiguous().unsqueeze(1)

if torch.cuda.is_available():
    train_x, train_y1, train_y2, test_x, test_y1, test_y2 = train_x.cuda(), train_y1.cuda(), train_y2.cuda(), test_x.cuda(), test_y1.cuda(), test_y2.cuda()

# Deep Kernel Learning code
data_dim = train_x.size(-1)

# Define neural network to do feature extraction/deep kernel learning
# Use same architecture suggested in Wilson paper
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

feature_extractor = LargeFeatureExtractor()

# Define Gaussian Process model that includes feature extraction step
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

            self.mean_module1 = gpytorch.means.ConstantMean()
            self.covar_module1 = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor

            self.num_outputs = 1 # required for BoTorch UCB
            self.batch_shape = [1] # required for BoTorch MES

            # Scaler for extracted features
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # Pass input data through feature extractor/DKL layers
            self.projected_x = self.feature_extractor(x)
            self.projected_x = self.scale_to_bounds(self.projected_x)

            # Define the mean and covariance on the extracted features
            mean_x1 = self.mean_module1(self.projected_x)
            covar_x1 = self.covar_module1(self.projected_x)

            mean_x2 = self.mean_module1(self.projected_x)
            covar_x2 = self.covar_module1(self.projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x1, covar_x1), gpytorch.distributions.MultivariateNormal(mean_x2, covar_x2)
        
# Define GP model using a Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y1, likelihood).double()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 2 if SMOKE_TEST else 60

# Put model in training mode
model.train()
likelihood.train()

# Use Adam as optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module1.parameters()},
    {'params': model.mean_module1.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# Define loss — use Marginal Log Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train(model, train_x, train_y, train_network=True):
    for _ in range(training_iterations):
        optimizer.zero_grad() # zero out gradients
        output = model.forward(train_x) # use model.forward to avoid opaque GPyTorch caching error
        loss = -mll(output, train_y)

        # Only train the DKL network if specified
        # Otherwise, only update GP parameters
        if train_network:
            loss.mean().backward()
        else:
            grad = torch.autograd.grad(loss.mean(),model.projected_x) # only calculate gradient for the GP layer
        optimizer.step()

    return model

breakpoint()

# Actually train the model
model = train(model, train_x, train_y1)

bo_loops = SMOKE_BO_LOOPS if SMOKE_TEST else 30

best = -float('inf')
true_best = max(test_y1)

for loop in range(1, bo_loops + 1): # adjust count to start from 1 so the training "outside the loop" counts
    # Use trained mean and covar module from the trained model to create a BoTorch GP to use in BO loop
    # This is a hacky workaround to get the BO loop to work using BoTorch
    scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    projected_x = scale_to_bounds(model.feature_extractor(train_x)).detach()
    gp = SingleTaskGP(projected_x, train_y1, covar_module=model.covar_module, mean_module=model.mean_module)

    # Use the UCB acquisition function
    acq = UpperConfidenceBound(gp, beta=0.1)

    # Can't use information-based acquisition functions with DKL/GP setup unfortunately
    # acq = qMaxValueEntropy(model, test_x)
    # NotImplementedError: Batched GP models (e.g., fantasized models) are not yet supported by `qMaxValueEntropy`

    # for multi-task, return several points and do a grid search to find the best option
    # or just grid search for not that many points

    # select point from candidate points
    candidate_idx = acq(model.feature_extractor(test_x).unsqueeze(1)).argmax()
    candidate = test_y1[candidate_idx]
    best = max(best, candidate)

    print(f"### Loop {loop} ###")
    print(f"Best So Far: {best.numpy()[0]}")
    print(f"Candidate Value: {candidate.numpy()[0]}")
    print(f"Regret: {(true_best - best).numpy()[0]}")

    # remove selected candidate from candidates and add to training set
    train_x = torch.cat((train_x, test_x[candidate_idx].unsqueeze(0)))
    train_y1 = torch.cat((train_y1, test_y1[candidate_idx].unsqueeze(0)))
    train_y2 = torch.cat((train_y2, test_y2[candidate_idx].unsqueeze(0)))

    test_x = torch.cat((test_x[:candidate_idx], test_x[candidate_idx + 1:]))
    test_y1 = torch.cat((test_y1[:candidate_idx], test_y1[candidate_idx + 1:]))
    test_y2 = torch.cat((test_y2[:candidate_idx], test_y2[candidate_idx + 1:]))

    # retrain DKL layer with new training points every 5 BO loops
    train_network = loop % 5 == 0
    if train_network:
        print("Retraining kernel...")
    train(model, train_x, train_y1, train_network)

# need to rewrite this
import plotly.express as px

data = {"DKL": regrets_dkl, "No DKL": regrets_no_dkl}

# fig = px.scatter(data, x=range(len(regrets_dkl)), y=data, height=600, width=800, title="Simple Regret at Each BO Iteration")
fig = px.line(data, x=range(len(regrets_dkl)), y=data, height=600, width=800, title="Simple Regret at Each BO Iteration", markers=True)
fig.update_layout(xaxis_title="BO Iteration", yaxis_title="Simple Regret", title_x=.5, legend_title="Strategy")
fig.update_yaxes(rangemode="tozero")
fig.write_image("plots/uncorrelated.png")
fig.show()