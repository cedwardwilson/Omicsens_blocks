''' Neural networks to train'''

# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from collections import OrderedDict

import torch
from torch import nn


class CNN2Layer(nn.Module):  # Encoder of CNN
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel1=3,
        kernel2=3,
        l1_in_channels=1,
        l2_in_channels=64,
        l2_out_channels=128,
        pool1=2,
        pool2=2,
        stride1=1,
        stride2=1,
        padding1=1,
        padding2=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.pool1 = pool1
        self.pool2 = pool2
        self.stride1 = stride1
        self.stride2 = stride2
        self.padding1 = padding1
        self.padding2 = padding2

        self.l1_in_channels = l1_in_channels
        self.l2_in_channels = l2_in_channels
        self.l2_out_channels = l2_out_channels
        assert self.conv1_output_length > 0
        self.conv1 = nn.Conv1d(
            l1_in_channels, l2_in_channels, kernel1, stride=stride1, padding=padding1
        )
        self.relu1 = nn.ReLU()
        assert pool1 <= self.conv1_output_length
        self.maxpool1 = nn.MaxPool1d(kernel_size=pool1)
        self.conv2 = nn.Conv1d(
            l2_in_channels, l2_out_channels, kernel2, stride=stride2, padding=padding2
        )
        self.relu2 = nn.ReLU()
        assert pool2 <= self.conv2_output_length, (
            f'pool2 kernel size: {pool2} should be less than or'
            + f' equal to conv2 output: {self.conv2_output_length}'
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=pool2)
        self.flatten = nn.Flatten()
        self.full = nn.Linear(int(l2_out_channels * self.pool2_output_length), output_dim)
        self.relu = nn.ReLU()

    def output_length(self, length_in, padding, kernel, stride, dilation=1):
        """Compute the output length after a convolution with dilation defaulted to 1."""
        return (length_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    @property
    def conv1_output_length(self):
        """Compute output length of conv1 layer.
        Dilation of kernel is 1.
        """
        return self.output_length(self.input_dim, self.padding1, self.kernel1, self.stride1)

    @property
    def pool1_output_length(self):
        """Compute output size after the first pooling."""
        return self.output_length(
            self.conv1_output_length, padding=0, kernel=self.pool1, stride=self.pool1
        )

    @property
    def pool2_output_length(self):
        """Compute output size after the second pooling."""
        return self.output_length(
            self.conv2_output_length, padding=0, kernel=self.pool2, stride=self.pool2
        )

    @property
    def conv2_output_length(self):
        """Compute output length of conv1 layer.
        Dilation of kernel is 1.
        """
        return (
            self.pool1_output_length + 2 * self.padding2 - 1 * (self.kernel2 - 1) - 1
        ) // self.stride2 + 1

    def forward(self, x):
        """Forward function of CNN."""
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        assert (
            x.ndim == 3
        ), f"Expected input with 3 dims [batch, self.l1_in_channels, self.input_dim], got {x.shape}"
        assert x.shape[1] == self.l1_in_channels

        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        maxpool1 = self.maxpool1(relu1)
        conv2 = self.conv2(maxpool1)
        relu2 = self.relu2(conv2)
        maxpool2 = self.maxpool2(relu2)
        flatten = self.flatten(maxpool2)
        full = self.full(flatten)
        return self.relu(full)

    def get_config(self):
        """Return a config dictionary for recreating this model and saving the model."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "kernel1": self.kernel1,
            "kernel2": self.kernel2,
            "l1_in_channels": self.l1_in_channels,
            "l2_in_channels": self.l2_in_channels,
            "l2_out_channels": self.l2_out_channels,
            "pool1": self.pool1,
            "pool2": self.pool2,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "padding1": self.padding1,
            "padding2": self.padding2,
        }

class MLP(nn.Module):  # Multi Layer Perceptron
    """Basic multi perceptron implementation."""

    def __init__(self, layer_features):
        super().__init__()
        self.layer_features = layer_features
        self.input_dim = layer_features[0][0]
        self.output_dim = layer_features[-1][1]
        middle_layers = [('start', nn.Linear(*layer_features[0]))]
        middle_layers.append(('start_' + 'relu', nn.ReLU()))
        for idx, features in enumerate(layer_features[1:-1]):
            assert features[0] == layer_features[idx][-1]
            middle_layers.append((str(idx), nn.Linear(*features)))
            middle_layers.append((str(idx) + 'relu', nn.ReLU()))
        middle_layers.append(('last_linear', nn.Linear(*layer_features[-1])))
        ord_dict = OrderedDict(middle_layers)
        self.net = nn.Sequential(ord_dict)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Callable function for nn.Module taking input vector as argument.

        Inpute tensor is of shape (batch_size, input_dimension)
        """
        assert x.shape[1] == self.layer_features[0][0]
        return self.relu(self.net(x))

    def get_config(self):
        """Return a config dictionary for recreating this model and saving the model."""
        return {
            "layer_features": self.layer_features,
        }


class RBFLayer(nn.Module):
    """supposed to be better than ReLU to study spectra but not implemented in pytorch"""

    # TODO: does not work for now -> vanishing gradient problem.

    def __init__(self, input_dim, num_centers, gamma=0.1):
        """The output of a RBF is given by:
        y(x) = sum_{i=1}^N a_i * phi(eps_i * ||x - c_i||) with phi being
        exp(-gamma * ||x - c_i||²)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers  # how many neurons to have
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))  # parameters to learn
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        """Callable function for nn.Module taking input vector as argument.

        Input tensor is batch size x input dim
        """
        # Change dimensions
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_centers, input_dim)

        # Euclidean distance squared between x and centers
        dist_sq = torch.sum(
            (x_expanded - centers_expanded) ** 2, dim=2
        )  # (batch_size, num_centers)

        # Apply Gaussian RBF function
        outputs = torch.exp(-self.gamma * dist_sq)

        return outputs


class RBFNeuralNetwork(
    nn.Module
):  # neural network that use RBF. Issue : Vanishing gradient problem
    """RBF NN."""

    def __init__(self, layer_features):
        super().__init__()
        self.layer_features = layer_features
        self.input_dim = layer_features[0][0]
        self.output_dim = layer_features[-1][1]
        middle_layers = [('start', nn.Linear(*layer_features[0]))]
        middle_layers.append(
            ('start_' + 'rbf', RBFLayer(layer_features[0][-1], num_centers=layer_features[0][-1]))
        )
        for idx, features in enumerate(layer_features[1:-1]):
            assert features[0] == layer_features[idx][-1]
            middle_layers.append((str(idx), nn.Linear(*features)))
            middle_layers.append(
                (str(idx) + 'relu', RBFLayer(features[-1], num_centers=features[-1]))
            )
        middle_layers.append(('last_linear', nn.Linear(*layer_features[-1])))
        ord_dict = OrderedDict(middle_layers)
        self.net = nn.Sequential(ord_dict)

    def forward(self, x):
        """Callable function for nn.Module taking input vector as argument.

        Input tensor is of shape (batch_size, input_dimension)
        """
        assert x.shape[1] == self.layer_features[0][0]
        return self.net(x)

    def get_config(self):
        """Return a config dictionary for recreating this model and saving the model."""
        return {
            "layer_features": self.layer_features,
        }


class BilateralLSTM(nn.Module):
    """Bilateral LSTM network."""

    def __init__(self, input_dim=1, output_dim=20, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        """Implementation of the forward pass.

        Takes input as shape batch size X sequence length
        """
        x = x.unsqueeze(-1)
        _output, (hn, _cn) = self.bilstm(x)
        num_directions = 2
        hn = hn.view(self.num_layers, num_directions, x.size(0), self.hidden_dim)
        last_layer_hn = hn[-1]
        final_hidden = torch.cat((last_layer_hn[0], last_layer_hn[1]), dim=1)

        out = self.relu(final_hidden)
        out = self.fc(out)
        return out

    def get_config(self):
        """Return a config dictionary for recreating this model and saving the model."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
