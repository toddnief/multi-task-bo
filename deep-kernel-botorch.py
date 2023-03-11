import math
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

from scipy.linalg import cholesky
from scipy.io import loadmat

import torch
from torch.nn import Softmax
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, GridInterpolationKernel, ScaleKernel, RBFKernel
from gpytorch.functions import MaternCovariance, RBFCovariance
from gpytorch.settings import trace_mode

from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedPosteriorTransform

# general setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
DEEP_KERNEL = True
DEEP_KERNEL_DIM = 5
CONTROL_SEED = True
if CONTROL_SEED:
    pl.seed_everything(42)

# training constants
SMOKE_TEST = False
BO_LOOPS = 5 if SMOKE_TEST else 20
TRAINING_LOOPS = 5 if SMOKE_TEST else 20

# problem size constants for random data generation
N_TASKS = 2
TRAIN_SIZE = 100
TEST_SIZE = 1000

# choose which experiment to run
EXPERIMENT = "MNIST" # choices: random, elevators, MNIST, CIFAR

if EXPERIMENT == "MNIST" or EXPERIMENT == "CIFAR":
    BO_LOOPS = 5 if SMOKE_TEST else 50

def stack_data(x, y, n_tasks=N_TASKS):
    # stack Y values and X values for feeding into multi-task GP
    for i in range(n_tasks):
        XX = torch.column_stack((x, torch.ones(len(x))*i)).double() # add feature column to x

        # create stacked tensors if they don't exits
        if i == 0:
            stacked_X = XX
            stacked_Y = standardize(y.T[i]).double()
        # add to the stack if they do exist
        else:
            stacked_X = torch.cat((stacked_X, XX))
            stacked_Y = torch.cat((stacked_Y, standardize(y.T[i]))).double()

    return stacked_X, stacked_Y.unsqueeze(1)

############################################
### DATA CREATION FOR RANDOM EXPERIEMENT ###
############################################

if EXPERIMENT == "random":
    X_DIM = 3

    # define random data
    X = torch.rand(10, X_DIM).double()

    for _ in range(N_TASKS):
        if _ == 0:
            Y = 1 - torch.norm(X - 0.5, dim=-1, keepdim=True)
            Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
        else:
            YY = 1 - torch.norm(X - 0.5, dim=-1, keepdim=True)
            YY = YY + 0.1 * torch.randn_like(YY)  # add some noise
            Y = torch.column_stack((Y,YY))

    train_X, train_Y = stack_data(X, Y)

############################################
## DATA CREATION FOR ELEVATORS EXPERIEMENT #
############################################

# use the Elevators dataset
if EXPERIMENT=="elevators":
    data = torch.Tensor(loadmat('data/elevators.mat')['data']).double()
    X = data[:, :-1]
    # X = X - X.min(0)[0]
    # X = 2 * (X / X.max(0)[0]) - 1
    Y = data[:, -1]

    # generate symmetric covariance matrix
    covar = torch.rand((N_TASKS, N_TASKS))
    covar = (covar + covar.T)/2
    covar = covar.fill_diagonal_(1).double()

    # generate random correlated tasks
    # tk double check this since the task correlation might not be quite right this way
    lower_chol = torch.from_numpy(cholesky(covar).T)
    YY = torch.randn((len(Y), N_TASKS)).double() # generate random normal values
    YYY = YY @ lower_chol

    train_X, train_Y = stack_data(X[:TRAIN_SIZE], YYY[:TRAIN_SIZE])
    test_X = X[-TEST_SIZE:]
    test_Y = YYY[-TEST_SIZE:]
    true_optimal = max(test_Y.sum(1)) # best combined value

    # make Y correct dimensions
    test_Y = test_Y.unsqueeze(1)

############################################
#### SHARED SETUP FOR IMAGE EXPERIMENTS ####
############################################

N_EXAMPLES = 5
LATENT_DIM = 20
BOUNDS = torch.tensor([[-6.0] * LATENT_DIM, [6.0] * LATENT_DIM], device=DEVICE, dtype=DTYPE)

# Scorer class to score images by probability mass on certain categories from classifier
class Scorer:
    def __init__(self, scoring_indices):
        self.scoring_indices = scoring_indices

    def score_image(self, x):
        sm = Softmax()
        with torch.no_grad():
            # probs = torch.exp(cnn_model(x)) # b x 10
            probs = sm(cnn_model(x))
            scored_indices = torch.zeros(len(probs[0]))
            scored_indices[self.scoring_indices] = 1
        return (probs * scored_indices).sum(dim=1)
    
def evaluate_tasks(x, tasks):
    evaluated_tasks = []
    for task in tasks:
        evaluated_tasks.append(task(x).unsqueeze(-1)) # add dimension so final output is 2D tensor
    return torch.column_stack(evaluated_tasks)

def gen_initial_data(tasks, n=N_EXAMPLES, d=LATENT_DIM, bounds=BOUNDS):
    # generate random initial training data
    X = unnormalize(torch.rand(n, d, device=DEVICE, dtype=DTYPE), bounds=bounds)

    # evaluate initial points
    Y = evaluate_tasks(decode(X), tasks)
    best_observed_value = Y.sum(1).max()

    return X, Y, best_observed_value

############################################
####### SETUP FOR MNIST EXPERIMENT #########
############################################

if EXPERIMENT=="MNIST":
    PRETRAINED_LOCATION = "./pretrained_models"

    # define classifier
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        
    # load pretrained weights
    cnn_model = Net().to(dtype=DTYPE, device=DEVICE)
    cnn_state_dict = torch.load(os.path.join(PRETRAINED_LOCATION, "mnist_cnn.pt"), map_location=DEVICE)
    cnn_model.load_state_dict(cnn_state_dict)

    # define variational autoencoder
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
        
    # load pretrained weights
    vae_model = VAE().to(dtype=DTYPE, device=DEVICE)
    vae_state_dict = torch.load(os.path.join(PRETRAINED_LOCATION, "mnist_vae.pt"), map_location=DEVICE)
    vae_model.load_state_dict(vae_state_dict)

    # functions for decoding and scoring images
    def score_image(x, indices):
        with torch.no_grad():
            probs = torch.exp(cnn_model(x)) # b x 10
            scored_indices = torch.zeroes(len(probs))
            scored_indices[indices] = 1
        return (probs * scored_indices).sum(dim=1)
    
    def decode(z):
        with torch.no_grad():
            decoded = vae_model.decode(z)
        return decoded.view(z.shape[0], 1, 28, 28)
    
    three_scorer = Scorer([3])
    even_scorer = Scorer([1,0,1,0,1,0,1,0,1,0])

    tasks = [three_scorer.score_image, even_scorer.score_image]
    initial_X, initial_Y, best_value = gen_initial_data(tasks)

    train_X, train_Y = stack_data(initial_X, initial_Y)

############################################
####### SETUP FOR CIFAR EXPERIMENT #########
############################################

if EXPERIMENT=="CIFAR":
    LATENT_DIM = 128 # choices: 64, 128, 256, 384
    BOUNDS = torch.tensor([[-6.0] * LATENT_DIM, [6.0] * LATENT_DIM], device=DEVICE, dtype=DTYPE)
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/tutorial9")
    PRETRAINED_FILENAME = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % LATENT_DIM)

    class Encoder(nn.Module):
        def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
            """
            Args:
            num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            latent_dim : Dimensionality of latent representation z
            act_fn : Activation function used throughout the encoder network
            """
            super().__init__()
            c_hid = base_channel_size
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
                act_fn(),
                nn.Flatten(),  # Image grid to single feature vector
                nn.Linear(2 * 16 * c_hid, latent_dim),
            )

        def forward(self, x):
            return self.net(x)
        
    class Decoder(nn.Module):
        def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
            """
            Args:
            num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            latent_dim : Dimensionality of latent representation z
            act_fn : Activation function used throughout the decoder network
            """
            super().__init__()
            c_hid = base_channel_size
            self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
            self.net = nn.Sequential(
                nn.ConvTranspose2d(
                    2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
                ),  # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(
                    c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
                ),  # 16x16 => 32x32
                nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            )

        def forward(self, x):
            x = self.linear(x)
            x = x.reshape(x.shape[0], -1, 4, 4)
            x = self.net(x)
            return x
        
    class Autoencoder(pl.LightningModule):
        def __init__(
            self,
            base_channel_size: int,
            latent_dim: int,
            encoder_class: object = Encoder,
            decoder_class: object = Decoder,
            num_input_channels: int = 3,
            width: int = 32,
            height: int = 32,
        ):
            super().__init__()
            # Saving hyperparameters of autoencoder
            self.save_hyperparameters()
            # Creating encoder and decoder
            self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
            self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
            # Example input array needed for visualizing the graph of the network
            self.example_input_array = torch.zeros(2, num_input_channels, width, height)

        def forward(self, x):
            """The forward function takes in an image and returns the reconstructed image."""
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat

        def _get_reconstruction_loss(self, batch):
            """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
            x, _ = batch  # We do not need the labels
            x_hat = self.forward(x)
            loss = F.mse_loss(x, x_hat, reduction="none")
            loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            # Using a scheduler is optional but can be helpful.
            # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        def training_step(self, batch, batch_idx):
            loss = self._get_reconstruction_loss(batch)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self._get_reconstruction_loss(batch)
            self.log("val_loss", loss)

        def test_step(self, batch, batch_idx):
            loss = self._get_reconstruction_loss(batch)
            self.log("test_loss", loss)

    vae_model = Autoencoder.load_from_checkpoint(PRETRAINED_FILENAME).double()

    from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
    # Pretrained model
    cnn_model = vgg11_bn(pretrained=True).double()
    cnn_model.eval() # for evaluation
    
    def decode(z):
        with torch.no_grad():
            decoded = vae_model.decoder(z)
        # return decoded.view(z.shape[0], 1, 32, 32)
        return decoded
    
    # CIFAR10 categories
    # 0: airplane
    # 1: automobile
    # 2: bird
    # 3: cat
    # 4: deer
    # 5: dog
    # 6: frog
    # 7: horse
    # 8: ship
    # 9: truck

    frog_scorer = Scorer([6])
    ship_scorer = Scorer([8])
    mammal_scorer = Scorer([3,4,5,7])
    
    tasks = [ship_scorer.score_image, ship_scorer.score_image]
    initial_X, initial_Y, best_value = gen_initial_data(tasks, d=LATENT_DIM, bounds=BOUNDS)
    train_X, train_Y = stack_data(initial_X, initial_Y)

############################################
########### DEEP KERNEL SETUP ##############
############################################

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
        self.add_module('linear4', torch.nn.Linear(50, DEEP_KERNEL_DIM))

data_dim = len(train_X[0]) - 1 # remove the task feature from the data
feature_extractor = LargeFeatureExtractor(data_dim).double()

# define the deep kernel
class DeepMaternKernel(MaternKernel):  
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2, diag=False, **params):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
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
    
def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()

class DeepRBFKernel(RBFKernel):  
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2, diag=False, **params):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params),
        )
    
# deep_kernel = ScaleKernel(DeepMaternKernel(feature_extractor, ard_num_dims=DEEP_KERNEL_DIM))
deep_kernel = ScaleKernel(DeepRBFKernel(feature_extractor, ard_num_dims=DEEP_KERNEL_DIM))

############################################
############### BO LOOP ####################
############################################

# keep track of progress on each loop
best_regret = float('inf')
curr_best = 0
curr_best_t = []
best_candidate_t = []
curr_candidate_t = []
regret_t = []
best_regret_t = []

for loop in range(BO_LOOPS):
    if DEEP_KERNEL:
        covar_module = deep_kernel
    else:
        covar_module = None

    # define the GP
    gp = MultiTaskGP(train_X, train_Y, task_feature=-1, covar_module=covar_module)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # fit the GP model
    def train_gp():
        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            # {'params': deep_kernel.parameters()},
            {'params': gp.covar_module.parameters()},
            {'params': gp.mean_module.parameters()},
            {'params': gp.likelihood.parameters()},
        ], lr=0.01)

        for i in range(TRAINING_LOOPS):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = gp.forward(train_X)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_Y).sum()
            loss.backward()
            optimizer.step()

    train_gp() # manually replace fit_gpytorch_mll(mll) in order to update deep kernel params

    # check to make sure weights of feature extractor are updating
    # for name, param in deep_kernel.named_parameters():
    #     print(param)
    #     break

    # TK figure out posterior transform to optimize for 50/50 weighting between tasks
    # use the acquisition function to pick a point
    task_weights = torch.ones(N_TASKS).double() # weight tasks equally
    pt_transform = ScalarizedPosteriorTransform(weights=task_weights) # need to define a multi-task objective for acquisition function
    # UCB = UpperConfidenceBound(gp, beta=0.1, posterior_transform=pt_transform)
    # acq_f = qUpperConfidenceBound(gp, beta=0.1, posterior_transform=pt_transform)
    acq_f = qExpectedImprovement(model=gp, best_f=curr_best, posterior_transform=pt_transform)

    if EXPERIMENT=="elevators":
        # evaluate acquisition function using specific points
        est_values = acq_f(test_X.unsqueeze(1)) # unsqueeze so it meets dimension expectations for UCB
        candidate_idx = est_values.argmax().item()
        acq_value = est_values.max().item()
        candidate = test_X[candidate_idx].unsqueeze(0)

        true_value = test_Y[candidate_idx] # look up actual value for point

        # calculate regret
        regret = true_optimal - true_value.sum()
        best_regret = min(regret, best_regret)

        # remove selected candidate from test set
        test_X = torch.cat((test_X[:candidate_idx], test_X[candidate_idx + 1:]))
        test_Y = torch.cat((test_Y[:candidate_idx], test_Y[candidate_idx + 1:]))
    else:
        # evaluate acquisition function using BoTorch
        bounds = torch.stack([torch.zeros(data_dim), torch.ones(data_dim)])
        candidate, acq_value = optimize_acqf(
            acq_f, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )

    # Decode and evaluate for MNIST experiment
    if EXPERIMENT=="MNIST" or EXPERIMENT=="CIFAR":
        new_X = unnormalize(candidate, bounds=BOUNDS).unsqueeze(0) # unsqueeze new_x to add batch dimension
        new_img = decode(new_X)

        true_value = evaluate_tasks(new_img, tasks)
        regret = 1 - (min(true_value[0][0], .5) + min(true_value[0][1], .5))
        # regret = 1 - true_value[0][0] # hack to mimic single-task learning

    # check if current value is best seen so far
    if regret < best_regret:
        best_regret = regret
        curr_best = acq_value
        best_candidate_t.append(candidate)

    # get scalars from tensors
    regret = regret.item() if isinstance(regret, Tensor) else regret
    best_regret = best_regret.item() if isinstance(best_regret, Tensor) else best_regret

    # maintain time step lists of progress
    curr_best_t.append(acq_value)
    curr_candidate_t.append(candidate)
    regret_t.append(regret)
    best_regret_t.append(best_regret)

    ### Print out progress report ###
    print(f"Iteration: {loop}")
    print(f"candidate: {candidate}")

    if EXPERIMENT=="elevators":
        print(f"Current Regret: {regret}")
        print(f"Best Regret: {best_regret}")

    if EXPERIMENT=="MNIST" or "CIFAR":
        print(f"True Value: {true_value}")

    # save point evaluated at current iteration
    for i in range(N_TASKS):
        if EXPERIMENT=="random":
            true_value = torch.rand(N_TASKS).unsqueeze(0)

        # add the task index to the candidate
        task_candidate = torch.column_stack((candidate, torch.tensor([i]))).double() # add feature column

        train_X = torch.cat((train_X, task_candidate))
        train_Y = torch.cat((train_Y, true_value[:, i].unsqueeze(0)))

print(best_regret_t)

if EXPERIMENT=="MNIST" or EXPERIMENT=="CIFAR":
    # TK figure out a better way to do this
    for z in best_candidate_t:
        img = decode(z.double()).squeeze().cpu()
        if EXPERIMENT=="CIFAR":
            img = img.permute(1,2,0)
        cv2.imshow("image", img.numpy())
        cv2.waitKey(0)