from math import floor
from scipy.linalg import cholesky
from scipy.io import loadmat
import numpy as np

import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.optim import optimize_acqf

SMOKE_TEST = True
SMOKE_TRAIN = 100
SMOKE_CANDIDATES = 1000
SMOKE_BO_LOOPS = 15

data = torch.Tensor(loadmat('elevators.mat')['data']).double()
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y1 = standardize(data[:, -1]) # mean 0, unit variance

# generate correlated second task using Cholesky decomposition
covar = np.array([[1.0, -.2], 
                  [-.2, 1.0]])

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

# # Define Gaussian Process model that includes feature extraction step
# class GPRegressionModel(gpytorch.models.ExactGP):
#         def __init__(self, train_x, train_y, likelihood, feature_extractor):
#             super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

#             self.mean_module1 = gpytorch.means.ConstantMean()
#             self.covar_module1 = gpytorch.kernels.GridInterpolationKernel(
#                 gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
#                 num_dims=2, grid_size=100
#             )
#             self.mean_module2 = gpytorch.means.ConstantMean()
#             self.covar_module2 = gpytorch.kernels.GridInterpolationKernel(
#                 gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
#                 num_dims=2, grid_size=100
#             )

#             self.feature_extractor = feature_extractor

#             self.num_outputs = 1 # required for BoTorch UCB
#             self.batch_shape = [1] # required for BoTorch MES

#             # Scaler for extracted features
#             self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

#         def forward(self, x):
#             # Pass input data through feature extractor/DKL layers
#             self.projected_x = self.feature_extractor(x)
#             self.projected_x = self.scale_to_bounds(self.projected_x)

#             # Define the mean and covariance on the extracted features
#             mean_x1 = self.mean_module1(self.projected_x)
#             covar_x1 = self.covar_module1(self.projected_x)

#             mean_x2 = self.mean_module1(self.projected_x)
#             covar_x2 = self.covar_module1(self.projected_x)
#             return gpytorch.distributions.MultivariateNormal(mean_x1, covar_x1), gpytorch.distributions.MultivariateNormal(mean_x2, covar_x2)      

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor

        self.num_outputs = 1 # required for BoTorch UCB
        self.batch_shape = [1] # required for BoTorch MES

        # Scaler for extracted features
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        # Pass input data through feature extractor/DKL layers
        self.projected_x = self.feature_extractor(x)
        self.projected_x = self.scale_to_bounds(self.projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y, likelihood)

# Define Gaussian Process model that includes feature extraction step
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, feature_extractor):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

            self.mean_module1 = gpytorch.means.ConstantMean()
            self.covar_module1 = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.mean_module2 = gpytorch.means.ConstantMean()
            self.covar_module2 = gpytorch.kernels.GridInterpolationKernel(
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
model = GPRegressionModel(train_x, train_y1, likelihood, feature_extractor).double()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 2 if SMOKE_TEST else 20

# Put model in training mode
model.train()
likelihood.train()

# Use Adam as optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.mean_module1.parameters()},
    {'params': model.covar_module1.parameters()},
    {'params': model.mean_module2.parameters()},
    {'params': model.covar_module2.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# Define loss — use Marginal Log Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train(model, train_x, train_y1, train_y2, train_network=True):
    for _ in range(training_iterations):
        optimizer.zero_grad() # zero out gradients
        output1, output2 = model.forward(train_x) # use model.forward to avoid opaque GPyTorch caching error
        loss = -mll(output1, train_y1) - mll(output2, train_y2)

        # Only train the DKL network if specified
        # Otherwise, only update GP parameters
        if train_network:
            loss.mean().backward()
        else:
            grad = torch.autograd.grad(loss.mean(),model.projected_x) # only calculate gradient for the GP layer
        optimizer.step()

    return model

# Actually train the model
model = train(model, train_x, train_y1, train_y2)

bo_loops = SMOKE_BO_LOOPS if SMOKE_TEST else 30

# initialize "best" points
best1 = -float('inf')
best2 = -float('inf')
best1_no_dkl = -float('inf')
best_multi = -float('inf')
best_multi_no_dkl = -float('inf')
true_best1 = max(test_y1)
true_best2 = max(test_y2)
true_best_multi = -float('inf')

for i in range(len(test_y1)):
    curr = test_y1[i] + test_y2[i]
    true_best_multi = max(curr, true_best_multi)

# set up GPs without DKL
gp1_no_dkl = SingleTaskGP(train_x, train_y1)
gp2_no_dkl = SingleTaskGP(train_x, train_y2)
mll1_no_dkl = ExactMarginalLogLikelihood(gp1_no_dkl.likelihood, gp1_no_dkl)
mll2_no_dkl = ExactMarginalLogLikelihood(gp2_no_dkl.likelihood, gp2_no_dkl)

regrets_dkl = []
regrets_no_dkl = []

for loop in range(1, bo_loops + 1): # adjust count to start from 1 so the training "outside the loop" counts
    # Use trained mean and covar module from the trained model to create a BoTorch GP to use in BO loop
    # This is a hacky workaround to get the BO loop to work using BoTorch
    # GPs are fit in the training step
    scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    projected_x = scale_to_bounds(model.feature_extractor(train_x)).detach()
    gp1 = SingleTaskGP(projected_x, train_y1, covar_module=model.covar_module1, mean_module=model.mean_module1)
    gp2 = SingleTaskGP(projected_x, train_y2, covar_module=model.covar_module2, mean_module=model.mean_module2)

    # fit the non DKL GPs
    fit_gpytorch_mll(mll1_no_dkl)
    fit_gpytorch_mll(mll2_no_dkl)

    # Use the UCB acquisition function
    acq1 = UpperConfidenceBound(gp1, beta=0.1)
    acq2 = UpperConfidenceBound(gp2, beta=0.1)

    # Set up acquisition functions for GPs that don't use the DKL mean and covariance
    acq1_no_dkl = UpperConfidenceBound(gp1_no_dkl, beta=.1)
    acq2_no_dkl = UpperConfidenceBound(gp2_no_dkl, beta=.1)

    # Can't use information-based acquisition functions with DKL/GP setup unfortunately
    # acq = qMaxValueEntropy(model, test_x)
    # NotImplementedError: Batched GP models (e.g., fantasized models) are not yet supported by `qMaxValueEntropy`

    # Evaluate points for DKL
    candidate1_idx = acq1(model.feature_extractor(test_x).unsqueeze(1)).argmax()
    candidate1 = test_y1[candidate1_idx]

    task1_results = acq1(model.feature_extractor(test_x).unsqueeze(1))
    task2_results = acq2(model.feature_extractor(test_x).unsqueeze(1))

    # Evaluate points for no DKL
    task1_results_no_dkl = acq1_no_dkl(test_x.unsqueeze(1))
    task2_results_no_dkl = acq2_no_dkl(test_x.unsqueeze(1))

    task1_idx_no_dkl = task1_results_no_dkl.argmax()
    task1_value_no_dkl = test_y1[task1_idx_no_dkl]
    best1_no_dkl = max(task1_value_no_dkl, best1_no_dkl)

    # Evaluate points for single-task
    best1 = max(best1, candidate1)
    regret1 = (true_best1 - best1).item()
    regret1_no_dkl = (true_best1 - best1_no_dkl).item()

    # check all candidates to find the best combination
    best_can = 0
    best_guess_can = -float('inf')
    best_can_no_dkl = 0
    best_guess_can_no_dkl = -float('inf')

    for can in range(len(task1_results)):
        curr = task1_results[can] + task2_results[can]
        curr_no_dkl = task1_results_no_dkl[can] + task2_results_no_dkl[can]
        if curr > best_guess_can:
            best_guess_can = curr
            best_can = can
        if curr_no_dkl > best_guess_can_no_dkl:
            best_guess_can_no_dkl = curr
            best_can_no_dkl = can

    # Evaluate the true values for the candidate
    real_multi_value = test_y1[best_can] + test_y2[best_can]
    best_multi = max(best_multi, real_multi_value)
    multi_regret = true_best_multi - best_multi

    real_multi_value_no_dkl = test_y1[best_can_no_dkl] + test_y2[best_can_no_dkl]
    best_multi_no_dkl = max(best_multi_no_dkl, real_multi_value_no_dkl)
    multi_regret_no_dkl = true_best_multi - best_multi_no_dkl

    # Save regret values
    regrets_dkl.append(multi_regret.item())
    regrets_no_dkl.append(multi_regret_no_dkl.item())

    # Print status updates for single-task optimization
    # best_idx1 = candidate1_idx

    print(f"### Loop {loop} ###")
    print(f"Best So Far: {best1.item()}")
    print(f"Candidate Value: {candidate1.item()}")
    print(f"Regret: {regret1}")
    print(f"Regret (No DKL): {regret1_no_dkl}")

    # Print status updates for multi-task optimization
    # print(f"### Loop {loop} ###")
    # print(f"Best So Far: {best_multi.item()}")
    # print(f"Candidate Value: {real_multi_value.item()}")
    # print(f"Regret (DKL): {multi_regret.item()}")
    # print(f"Regret (No DKL): {multi_regret_no_dkl.item()}")

    # Code to break if the optimal point has been found
    # if not regret1:
    #     break

    # Remove selected candidate from candidates and add to training set
    train_x = torch.cat((train_x, test_x[best_can].unsqueeze(0)))
    train_y1 = torch.cat((train_y1, test_y1[best_can].unsqueeze(0)))
    train_y2 = torch.cat((train_y2, test_y2[best_can].unsqueeze(0)))

    test_x = torch.cat((test_x[:best_can], test_x[best_can + 1:]))
    test_y1 = torch.cat((test_y1[:best_can], test_y1[best_can + 1:]))
    test_y2 = torch.cat((test_y2[:best_can], test_y2[best_can + 1:]))

    # tk Make sure the model is actually getting trained here
    # print the gradients or something

    # retrain DKL layer with new training points every 5 BO loops
    train_network = loop % 5 == 0
    if loop == bo_loops:
        train_network = False # Don't retrain after the final BO loop
    if train_network:
        print("Retraining kernel...")
    
    model = train(model, train_x, train_y1, train_y2, train_network)

# Plot regret
import plotly.express as px
plotname = "uncorrelated.png"

data = {"DKL": regrets_dkl, "No DKL": regrets_no_dkl}

fig = px.line(data, x=range(len(regrets_dkl)), y=data, height=600, width=800, title="Simple Regret at Each BO Iteration", markers=True)
fig.update_layout(xaxis_title="BO Iteration", yaxis_title="Simple Regret", title_x=.5, legend_title="Strategy")
fig.update_yaxes(rangemode="tozero")
fig.write_image(f"plots/{plotname}")
fig.show()