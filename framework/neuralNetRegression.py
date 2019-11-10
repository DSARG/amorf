from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import framework.metrics as er
from framework.utils import EarlyStopping as early
from numpy import mean
import numpy as np
from sklearn.model_selection import train_test_split
import traceback

# TODO: Add Docstring
# TODO: Add Data Loaders


class NeuralNetRegressor:

    def __init__(self, model=None, patience=5, learning_rate=0.01, training_limit=None, print_after_epochs=10, batch_size=None, use_gpu=False):
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

        self.loss_fn = er.tensor_average_relative_root_mean_squared_error  # nn.MSELoss()
        self.patience = patience
        self.learning_rate = learning_rate
        self.print_after_epochs = print_after_epochs
        self.batch_size = batch_size
        if isinstance(training_limit, int):
            self.training_limit = training_limit
        else:
            self.training_limit = None

    def fit(self, X_train, y_train):

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

        self.optimizer = optim.Adam(
            self.model.parameters(), self.learning_rate)
        self.model.train()

        stopper = early(self.patience)
        stop = False
        epochs = 0

        X_train_t_batched, y_train_t_batched = self.__split_training_set_to_batches(
            X_train_t, y_train_t, self.batch_size)

        while(stop is False):
            # train
            for batch_X, batch_y in zip(X_train_t_batched, y_train_t_batched):
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
                validation_error = er.tensor_average_relative_root_mean_squared_error(
                    y_pred_val, y_validate_t)
                train_error = er.tensor_average_relative_root_mean_squared_error(
                    y_pred_train, y_train_t)
                print('Validation Error: {} \nTrain Error: {}'.format(
                    validation_error, train_error))
            stop = stopper.stop(validation_loss)
            epochs += 1
            if self.training_limit is not None and self.training_limit >= epochs:
                stop = False

        y_pred_train = self.model(X_train_t)
        final_train_error = er.tensor_average_relative_root_mean_squared_error(
            y_pred_train, y_train_t)
        final_validation_error = er.tensor_average_relative_root_mean_squared_error(
            y_pred_val, y_validate_t)

        print("Final Epochs: {} \nFinal Train Error: {}\nFinal Validation Error: {}".format(
            epochs, final_train_error, final_validation_error))

        return self

    def __split_training_set_to_batches(self, X_train_t, y_train_t, batch_size):
        if batch_size is None:
            return torch.split(X_train_t, len(X_train_t)), torch.split(y_train_t, len(X_train_t))
        else:
            return torch.split(X_train_t, batch_size), torch.split(y_train_t, batch_size)

    def predict(self, X_test):

        X_test_t = self.model.convert_test_set_to_tensor(X_test, self.Device)

        self.model.eval()
        with torch.no_grad():
            y_pred_t = self.model(X_test_t)

        return y_pred_t.detach().numpy() if self.Device is 'cpu' else y_pred_t.cpu().detach().numpy()

    def save(self, store_path):
        try:
            torch.save(self.model, store_path + '.ckpt')
        except Exception:
            print(traceback.format_exc())

    def load(self, load_path):
        try:
            model = torch.load(load_path).to(self.Device)
            self.model = model
        except Exception:
            print(traceback.format_exc())


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
