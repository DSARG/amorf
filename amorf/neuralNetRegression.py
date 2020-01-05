import traceback
from abc import ABC, abstractmethod

from numpy import mean
import inspect
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from amorf.metrics import average_relative_root_mean_squared_error
from amorf.utils import EarlyStopping, printMessage


class NeuralNetRegressor:
    """Regressor that uses PyTorch models to predict multiple targets

    Raises:
        ValueError: If given model ist not instance of pytorch.NN.nodule

    Args:
        model (pytorch.NN.Module,optional): PyTorch Model to use. Default: None (will use Linear_NN_Model)
        batch_size (int,optional): Otherwise training set is split into batches of given size. Default: None
        shuffle (bool,optional) : Set to True to have the data reshuffled at every epoch. Default: False
        learning_rate (float,optional): learning rate for optimizer. Default: 0.01
        use_gpu (bool,optional): Flag that allows usage of cuda cores for calculations. Default: False
        patience (int,optional): Stop training after p continous incrementations (stops at training limit if it is not none). Default: 0
        training_limit (int,optional): After specified number of epochs training will be terminated, regardless of EarlyStopping stopping. Default: 100 
        verbosity (int,optional): 0 to only print errors, 1 (default) to print status information. Default: 1
        print_after_epochs (int,optional): Specifies after how many epochs training and validation error will be printed to command line. Default: 10
    """

    def __init__(self, model=None, batch_size=None, shuffle=False, learning_rate=0.01, use_gpu=False, patience=None, training_limit=1000, verbosity=1, print_after_epochs=100):
        self.Device = 'cpu'
        if use_gpu is True and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.Device = "cuda:0"
        if model is not None:
            if not isinstance(model, nn.Module):
                raise ValueError(
                    '\'{}\' is not a valid instance of pytorch.nn'.format(model))
            else:
                self.model = model.to(self.Device)
        else:
            self.model = None

        self.loss_fn = nn.MSELoss() #average_relative_root_mean_squared_error  # nn.MSELoss()
        self.patience = patience
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.print_after_epochs = print_after_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training_limit = training_limit if isinstance(
            training_limit, int) else None
        if training_limit is None and patience is None:
            raise ValueError('Either training_limit or patience must be set')

    def fit(self, X_train, y_train):
        """Fits the model to the training data set

        Args:
            X_train (nd.array): Set of descriptive Variables
            y_train (nd.array): Set of target Variables

        Returns:
            NeuralNetRegressor: fitted NeuralNetRegressor
        """

        if self.model is None:
            self.model = Linear_NN_Model(input_dim=len(X_train[0]), output_dim=len(
                y_train[0]), selector='max', p_dropout_1=0.2, p_dropout_2=0.2).to(self.Device)

        # Create Validation Set from train Set
        X_train, X_validate, y_train, y_validate = train_test_split(
            X_train, y_train, test_size=0.1)
        X_train_t, y_train_t = self.model.convert_train_set_to_tensor(
            X_train, y_train, self.Device)
        X_validate_t, y_validate_t = self.model.convert_train_set_to_tensor(
            X_validate, y_validate, self.Device)

        batch_size = len(
            X_train_t) if self.batch_size is None else self.batch_size
        train_dataloader = DataLoader(TensorDataset(
            X_train_t, y_train_t), batch_size=batch_size, shuffle=self.shuffle)
        self.optimizer = optim.Adam(
            self.model.parameters(), self.learning_rate)
        self.model.train()

        if self.patience is not None:
            stopper = EarlyStopping(self.patience)
        stop = False
        epochs = 0

        while(stop is False):
            # train
            for batch in train_dataloader:
                batch_X = batch[0]
                batch_y = batch[1]
                self.optimizer.zero_grad()
                y_pred_train = self.model(batch_X)
                loss = self.loss_fn(y_pred_train, batch_y)
                loss.backward()
                self.optimizer.step()
            # caculate validation loss an perform early stopping
            y_pred_val = self.model(X_validate_t)
            validation_loss = self.loss_fn(y_pred_val, y_validate_t)

            if epochs % self.print_after_epochs == 0:
                y_pred_train = self.model(X_train_t)
                validation_error = average_relative_root_mean_squared_error(
                    y_pred_val, y_validate_t)
                train_error = average_relative_root_mean_squared_error(
                    y_pred_train, y_train_t)
                printMessage('Epoch: {}\nValidation Error: {} \nTrain Error: {}'.format(
                    epochs, validation_error, train_error), self.verbosity)

            if self.patience is not None:
                stop = stopper.stop(validation_loss, self.model) 
            if stop is True and self.patience > 1 :  
                self.model.load_state_dict(stopper.best_model['state_dict'])
            epochs += 1
            if self.training_limit is not None and self.training_limit <= epochs:
                stop = True

        y_pred_train = self.model(X_train_t)
        final_train_error = average_relative_root_mean_squared_error(
            y_pred_train, y_train_t)
        final_validation_error = average_relative_root_mean_squared_error(
            y_pred_val, y_validate_t)

        printMessage("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(
            epochs, final_train_error, final_validation_error), self.verbosity)

        return self

    def predict(self, X_test):
        """Predicts the target variables for the given test set

        Args:
            X_test (np.ndarray): Test set with descriptive variables

        Returns:
            np.ndarray: Predicted target variables
        """

        X_test_t = self.model.convert_test_set_to_tensor(X_test, self.Device)

        self.model.eval()
        with torch.no_grad():
            y_pred_t = self.model(X_test_t)

        return y_pred_t.detach().numpy() if self.Device is 'cpu' else y_pred_t.cpu().detach().numpy()

    def save(self, store_path):
        """Save model and store it at given path

        Args:
            store_path (string): Path to store model at
        """
        try:
            torch.save(self.model, store_path)
        except Exception:
            printMessage(traceback.format_exc(), self.verbosity)

    def load(self, load_path):
        """Load model from path

        Args:
            load_path (string): Path to saved model
        """
        try:
            model = torch.load(load_path).to(self.Device)
            self.model = model
        except Exception:
            printMessage(traceback.format_exc(), self.verbosity)

    def score(self, X_test, y_test):
        """Returns Average Relative Root Mean Squared Error for given test data and targets

        Args:
            X_test (np.ndarray): Test samples
            y_test (np.ndarray): True targets
        """
        return average_relative_root_mean_squared_error(self.predict(X_test), y_test) 
    
    ### FOLLOWING FUNCTIONS ARE NECESSARY TO PERFORM GRID SEARCH 
    
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


class AbstractNeuralNet(ABC):

    @abstractmethod
    def convert_train_set_to_tensor(self, X_train, y_train, device):
        pass

    @abstractmethod
    def convert_test_set_to_tensor(self, X_test, device):
        pass


@AbstractNeuralNet.register
class Linear_NN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, selector='max', p_dropout_1=0.5, p_dropout_2=0.5):
        MIDDLE_LAYER_NEURON_CALCULATION = {
            'mean': mean([input_dim, output_dim]),
            'max': max([input_dim, output_dim]),
            'doubleInput': input_dim * 2
        }
        super().__init__()
        if selector not in MIDDLE_LAYER_NEURON_CALCULATION:
            raise ValueError('Selector \'{}\' is not valid')
        self.dropout_1 = p_dropout_1
        self.dropout_2 = p_dropout_2

        middleLayerNeurons = int(MIDDLE_LAYER_NEURON_CALCULATION[selector])
        self.batchNorm = nn.BatchNorm1d(middleLayerNeurons)

        self.fc1 = nn.Linear(input_dim, middleLayerNeurons)
        self.fc2 = nn.Linear(middleLayerNeurons, middleLayerNeurons)
        self.fc3 = nn.Linear(middleLayerNeurons, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.batchNorm(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_1)
        out = self.fc2(out)
        out = self.batchNorm(out)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_2)
        out = self.fc3(out)

        return out

    def convert_train_set_to_tensor(self, X_train, y_train, device):
        X_train_t = torch.from_numpy(X_train).to(device).float()
        y_train_t = torch.from_numpy(y_train).to(device).float()

        return X_train_t, y_train_t

    def convert_test_set_to_tensor(self, X_test, device):
        X_test_t = torch.from_numpy(X_test).to(device).float()
        return X_test_t

# Experimental
@AbstractNeuralNet.register
class Convolutional_NN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p_dropout = p_dropout
        width_out = self.__get_output_size(input_dim)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(self.p_dropout)
        self.fc1 = nn.Linear(64 * width_out, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def convert_train_set_to_tensor(self, X_train, y_train, device):
        X_train_t = torch.from_numpy(X_train).to(device).float().reshape(
            len(X_train), 1, len(X_train[0]))
        y_train_t = torch.from_numpy(y_train).to(device).float()
        return X_train_t, y_train_t

    def convert_test_set_to_tensor(self, X_test, device):
        X_test_t = torch.from_numpy(X_test).to(device).float().reshape(
            len(X_test), 1, len(X_test[0]))
        return X_test_t

    def __get_output_size(self, input_dim):
        k_c_1 = 2
        k_p_1 = 2
        k_c_2 = 2
        k_p_2 = 2
        s_c_1 = 2
        s_c_2 = 2
        s_p_1 = 2
        s_p_2 = 2
        wk1 = self.__get_size(input_dim, k_c_1, 0, s_c_1)
        wp1 = self.__get_size(wk1, k_p_1, 0, s_p_1)
        wk2 = self.__get_size(wp1, k_c_2, 0, s_c_2)
        wp2 = self.__get_size(wk2, k_p_2, 0, s_p_2)
        return int(wp2)

    def __get_size(self, w, k, p=0, s=0):
        return ((w - k + 2 * p) / s) + 1
