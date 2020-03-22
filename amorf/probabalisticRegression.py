import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO, TracePredictive, EmpiricalMarginal
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal

import numpy as np

from sklearn.model_selection import train_test_split
from amorf.utils import EarlyStopping, printMessage
from torch.utils.data import TensorDataset, DataLoader
from amorf.metrics import average_relative_root_mean_squared_error

#from gpar import GPARRegressor
import sklearn.gaussian_process as gp
import inspect
from collections import defaultdict

# FIXME Wronf Format of results


class BayesianNeuralNetworkRegression:
    """Bayesian Neural Network that uses a Pyro model to predict multiple targets

    Uses Pyros Elbo Loss internally
    Args:
        batch_size (int,optional): Otherwise training set is split into batches of given size. Default: None
        shuffle (bool,optional): Set to True to have the data reshuffled at every epoch. Default: False
        learning_rate (float,optional): Learning rate for optimizer. Default: 1e-3
        use_gpu (bool,optional):  Flag that allows usage of cuda cores for calculations. Default: False
        patience (int,optional): Stop training after p continous incrementations. Default: None
        training_limit (int,optional): After specified number of epochs training will be terminated, regardless of early stopping. Default: 100
        verbosity (int,optional): 0 to only print errors, 1 (default) to print status information. Default: 1
        print_after_epochs (int,optional): Specifies after how many epochs training and validation error will be printed to command line. Default: 500
    """

    def __init__(self, batch_size=None, shuffle=False, learning_rate=1e-3, use_gpu=False, patience=None, training_limit=100, verbosity=1, print_after_epochs=500):
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.training_limit = training_limit
        self.print_after_epochs = print_after_epochs
        self.verbosity = verbosity
        self.Device = 'cpu'
        if use_gpu is True and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.Device = "cuda:0"

        if training_limit is None and patience is None:
            raise ValueError('Either training_limit or patience must be set')
    # FIXME: Training behaviour when Patience and traininglimit is set

    def fit(self, X_train, y_train):
        """Fits the model to the training data set

        Args:
            X_train (nd.array): Set of descriptive Variables
            y_train (nd.array): Set of target Variables

        Returns:
            NeuralNetRegressor: fitted NeuralNetRegressor
        """
        X_train, X_validate_t, y_train, y_validate_t = train_test_split(
            X_train, y_train, test_size=0.1)
        X_train_t = torch.tensor(X_train, dtype=torch.float).to(self.Device)
        y_train_t = torch.tensor(y_train, dtype=torch.float).to(self.Device)
        X_validate_t = torch.tensor(
            X_validate_t, dtype=torch.float).to(self.Device)
        y_validate_t = torch.tensor(
            y_validate_t, dtype=torch.float).to(self.Device)

        n_targets = len(y_train_t[0])
        n_features = len(X_train_t[0])
        self.net = NN(n_features, n_targets)
        self.net.to(self.Device)
        self.guide = AutoDiagonalNormal(self.model)
        self.optim = Adam({"lr": self.learning_rate})
        self.svi = SVI(self.model, self.guide, self.optim, loss=Trace_ELBO())

        batch_size = len(
            X_train_t) if self.batch_size is None else self.batch_size
        train_dataloader = DataLoader(TensorDataset(
            X_train_t, y_train_t), batch_size=batch_size, shuffle=self.shuffle)
        pyro.clear_param_store()
        losses = []
        if self.patience is not None:
            stopper = EarlyStopping(self.patience)
        stop = False
        epochs = 0
        while(stop is False):
            # calculate the loss and take a gradient step
            for batch in train_dataloader:
                batch_X = batch[0]
                batch_y = batch[1]
                loss_batch = self.svi.step(batch_X, batch_y)
                losses.append(loss_batch)

            validation_error = self.svi.evaluate_loss(
                X_validate_t, y_validate_t)
            train_error = self.svi.evaluate_loss(X_train_t, y_train_t)
            if self.patience is not None:
                stop = stopper.stop(validation_error, self.net)
            # if stop is True and self.patience > 1:
            #     # TODO: add loading of best,guide, optimizer and model here
            #     self.net = stopper.best_model
            #     self.svi.
            if epochs % self.print_after_epochs == 0:
                printMessage('Epoch: {}\nValidation Error: {} \nTrain Error: {}'.format(
                    epochs, validation_error, train_error), self.verbosity)

            epochs += 1

            if self.training_limit is not None and self.training_limit <= epochs:
                stop = True

        final_train_error = self.svi.evaluate_loss(X_train_t, y_train_t)
        final_validation_error = self.svi.evaluate_loss(
            X_validate_t, y_validate_t)
        printMessage("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(
            epochs, final_train_error, final_validation_error), self.verbosity)
        return self

    def predict(self, X_test, num_samples=100):
        """Predicts the target variables for the given test set
        Args:
            X_test (np.ndarray): Test set withdescriptive variables
        Returns:
            np.ndarray: Predicted target variables
        """
        from pyro.infer import Predictive
        x_data_test = torch.tensor(X_test, dtype=torch.float).to(self.Device)

        predictive = Predictive(self.net, guide=self.guide, num_samples=100,
                                return_sites=("obs", "_RETURN"))

        samples = predictive(x_data_test)
        pred_summary = self.__summary(samples)
        stds = pred_summary['_RETURN']['std']
        means = pred_summary['_RETURN']['mean']

        return stds.cpu().detach().numpy(), means.cpu().detach().numpy()

    def __summary(self, samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats

    def save(self, path):
        """Save model and store it at given path

        Args:
            store_path (string): Path to store model at
        """
        model_path = path + '_model'
        opt_path = path + '_opt'
        guide_path = path + '_guide'
        torch.save(self.net, model_path)
        torch.save(self.guide, guide_path)

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
        guide_path = path + '_guide'
        self.net = torch.load(model_path)

        pyro.get_param_store().load(path + '_params')

        self.optim = Adam({"lr": self.learning_rate})
        self.optim.load(opt_path)
        self.guide = AutoDiagonalNormal(self.model)
        self.guide = torch.load(guide_path)
        self.svi = SVI(self.net, self.guide, self.optim, loss=Trace_ELBO())

    def model(self, x_data, y_data):
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

    def score(self, X_test, y_test):
        """Returns Average Relative Root Mean Squared Error for given test data and targets

        Args:
            X_test (np.ndarray): Test samples
            y_test (np.ndarray): True targets
        """
        # return means
        y_pred = self.predict(X_test)[1]
        return average_relative_root_mean_squared_error(y_pred, y_test)

    # FOLLOWING FUNCTIONS ARE NECESSARY TO PERFORM GRID SEARCH

    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


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

# https://github.com/wesselb/gpar


class GaussianProcessAutoregressiveRegression:
    """[summary]
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


#TODO: AAAdd test
class GaussianProcessRegression:
    """Wrapper around sklearns GaussianProcessRegressor (sklearn.gaussian_process.GaussienProcessRegressor) 
    (from sklearn Documentation https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/gaussian_process/gpr.py)

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.
    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:
       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Args:
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    """

    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super().__init__()
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        # TODO: Either replace with RBF or find explanation on why this is could be a good kernel
        self.kernel = gp.kernels.ConstantKernel(
            1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3)) if kernel is None else kernel
        self.model = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                                 optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                                 normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)

    def fit(self, X_train, y_train):
        """Fit Gaussian process regression model.
        (from sklearn Documentation https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/gaussian_process/gpr.py)
        Args:
            X (array-like) shape = (n_samples, n_features)
            Training data
            y (array-like) shape = (n_samples, [n_output_dims])
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        return self.model.fit(X_train, y_train)

    def predict(self, X_test, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model 
        (from sklearn Documentation https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/gaussian_process/gpr.py)

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.
        Args:
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        Returns:
        y_mean (array): shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        y_std (array): shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        y_cov (array): shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """

        return self.model.predict(X_test, return_std=return_std, return_cov=return_cov)
