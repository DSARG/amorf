import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from numpy import mean, sum
import numpy as np
import pyro 
from framework.utils import printMessage
from pyro.distributions import Normal, Categorical, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO, TracePredictive, EmpiricalMarginal
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
from framework.utils import EarlyStopping as early

# TODO: clear Imports
# TODO: add Validation loss 
# TODO: Data Loaders

class BayesianNeuralNetworkRegression:
    """Bayesian Neural Network that uses a Pyro model to predict multiple targets

    Args:
        patience (int): Stop training after p continous incrementations
        batch_size (int): Default None - otherwise training set is split into batches of given size
        learning_rate (float): learning rate for optimizer
        print_after_epochs (int): Specifies after how many epochs training and validation error will be printed to command line
        verbosity (int): 0 to only print errors, 1 (default) to print status information
        use_gpu (bool):  Flag that allows usage of cuda cores for calculations
    """

    def __init__(self, patience=5, batch_size=None, learning_rate=1e-3, training_limit=None, print_after_epochs=500,verbosity=1, use_gpu=False):
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.training_limit = training_limit
        self.print_after_epochs = print_after_epochs
        self.verbosity = verbosity
        self.Device = 'cpu'
        if use_gpu is True and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.Device = "cuda:0"

    def fit(self, X_train, y_train):
        """Fits the model to the training data set

        Args:
            X_train (nd.array): Set of descriptive Variables
            y_train (nd.array): Set of target Variables

        Returns:
            NeuralNetRegressor: fitted NeuralNetRegressor
        """
        x_data = torch.tensor(X_train, dtype=torch.float).to(self.Device)
        y_data = torch.tensor(y_train, dtype=torch.float).to(self.Device)

        n_targets = len(y_data[0])
        n_features = len(x_data[0])
        self.net = NN(n_features, n_targets)
        self.net.to(self.Device)
        self.guide = AutoDiagonalNormal(self.__model)
        self.optim = Adam({"lr": self.learning_rate})
        self.svi = SVI(self.__model, self.guide, self.optim, loss=Trace_ELBO())

        X_train_t_batched, y_train_t_batched = self.__split_training_set_to_batches(
            x_data, y_data, self.batch_size)
        pyro.clear_param_store()
        losses = []
        stopper = early(self.patience)
        stop = False
        epochs = 0
        while(stop is False):
            # calculate the loss and take a gradient step
            for batch_X, batch_y in zip(X_train_t_batched, y_train_t_batched):
                loss = self.svi.step(batch_X, batch_y)
                losses.append(loss)
                stop = stopper.stop(loss)

            if epochs % self.print_after_epochs == 0:
                printMessage("[iteration %04d] loss: %.4f" %
                      (epochs + 1, loss / len(x_data)),self.verbosity)
            epochs += 1 

            if self.training_limit is not None and self.training_limit >= epochs:
                stop = True

        return self

    def predict(self, X_test, y_test, num_samples=100):
        """Predicts the target variables for the given test set

        Args:
            X_test (np.ndarray): Test set withdescriptive variables

        Returns:
            np.ndarray: Predicted target variables
        """
        x_data_test = torch.tensor(X_test, dtype=torch.float).to(self.Device)
        y_data_test = torch.tensor(y_test, dtype=torch.float).to(self.Device)

        #get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

        def wrapped_model(x_data, y_data):
            pyro.sample("prediction", Delta(self.__model(x_data, y_data)))

        posterior = self.svi.run(x_data_test, y_data_test)

        trace_pred = TracePredictive(
            wrapped_model, posterior, num_samples)
        post_pred = trace_pred.run(x_data_test, None)

        marginal = EmpiricalMarginal(post_pred, ['obs'])._get_samples_and_weights()[
            0].detach().cpu().numpy()
        predictions = torch.from_numpy(marginal[:, 0, :, :]).to(self.Device)
        stds, means = torch.std_mean(predictions, 0)
        return stds.cpu().detach().numpy(), means.cpu().detach().numpy()

    def save(self, path):
        """Save model and store it at given path

        Args:
            store_path (string): Path to store model at
        """
        model_path = path + '_model'
        opt_path = path + '_opt'
        torch.save(self.net, model_path)

        self.optim.save(opt_path)
        ps = pyro.get_param_store()
        ps.save(path + '_params')


    def load(self, path):
        """Load model from path

        Args:
            load_path (string): Path to saved model
        """
        model_path = path + '_model'
        opt_path = path + '_opt'
        self.net = torch.load(model_path)

        pyro.get_param_store().load(path + '_params')

        self.optim = Adam({"lr": self.learning_rate})
        self.optim.load(opt_path)
        self.guide = AutoDiagonalNormal(self.__model)
        self.svi = SVI(self.__model, self.guide, self.optim, loss=Trace_ELBO())

    # TODO: Extract to utils
    def __split_training_set_to_batches(self, X_train_t, y_train_t, batch_size):
        if batch_size is None:
            return torch.split(X_train_t, len(X_train_t)), torch.split(y_train_t, len(X_train_t))
        else:
            return torch.split(X_train_t, batch_size), torch.split(y_train_t, batch_size)

    def __model(self, x_data, y_data):
        fc1w_prior = Normal(loc=torch.zeros_like(
            self.net.fc1.weight).to(self.Device), scale=torch.ones_like(self.net.fc1.weight).to(self.Device))
        fc1b_prior = Normal(loc=torch.zeros_like(
            self.net.fc1.bias).to(self.Device), scale=torch.ones_like(self.net.fc1.bias).to(self.Device))

        fc2w_prior = Normal(loc=torch.zeros_like(
            self.net.fc2.weight).to(self.Device), scale=torch.ones_like(self.net.fc2.weight).to(self.Device))
        fc2b_prior = Normal(loc=torch.zeros_like(
            self.net.fc2.bias).to(self.Device), scale=torch.ones_like(self.net.fc2.bias).to(self.Device))

        outw_prior = Normal(loc=torch.zeros_like(
            self.net.out.weight).to(self.Device), scale=torch.ones_like(self.net.out.weight).to(self.Device))
        outb_prior = Normal(loc=torch.zeros_like(
            self.net.out.bias).to(self.Device), scale=torch.ones_like(self.net.out.bias).to(self.Device))
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior,
                  'fc2.bias': fc2b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.net, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()
        scale = pyro.sample("sigma", Uniform(0., 10.))
        # with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        with pyro.iarange("observed data", use_cuda=True):
            pyro.sample("obs",
                        Normal(prediction_mean, scale),
                        obs=y_data)
        return prediction_mean


class NN(nn.Module):
    def __init__(self, input_size=None, output_size=None):
        super(NN, self).__init__()
        self.batchNorm = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.batchNorm(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.out(output)
        return output
