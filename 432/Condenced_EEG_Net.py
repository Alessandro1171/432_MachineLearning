"""Based on the EEG Net model but with condensed temporal deep learning the model 
trains itself on eeg brain imagery data to classify other matrices to one of the four movements
Authors
 * Alessandro Dare, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torchvision import datasets, transforms
import speechbrain as sb
class Condenced_EEG_Net(torch.nn.Module):
    def __init__(
            self,
            input_shape=None,  # (1, T, C, 1)
            condenced_temporal_kernels=8,
            condenced_temporal_kernelsize=(65, 1),
            condenced_selection_pool=(8, 1),
            condenced_selection_kernelsize=(17, 1),
            condenced_depth_pool=(4, 1),

            condenced_depth_multiplier=2,
            condenced_spatial_max_norm=1.,
            condenced_selection_kernels=None,
            activation="elu",
            pool_type="avg",
            dropout=0.5,
            dense_max_norm=0.25,
            dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation == "gelu":
            activation = torch.nn.GELU()
        elif activation == "elu":
            activation = torch.nn.ELU()
        elif activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        # T = input_shape[1]
        C = input_shape[2]
        self.condenced_module = torch.nn.Sequential()

        self.condenced_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=condenced_temporal_kernels,
                kernel_size=condenced_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.condenced_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=condenced_temporal_kernels, momentum=0.01, affine=True,
            ),
        )

        condenced_depth_kernels = (
                condenced_depth_multiplier * condenced_temporal_kernels
        )
        self.condenced_module.add_module(
            "dept_0",
            sb.nnet.CNN.Conv2d(
                in_channels=condenced_temporal_kernels,
                out_channels=condenced_depth_kernels,
                kernel_size=(1, C),
                groups=condenced_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=condenced_spatial_max_norm,
                swap=True,
            ),
        )
        self.condenced_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=condenced_depth_kernels, momentum=0.01, affine=True,
            ),
        )

        self.condenced_module.add_module("act_1", activation)
        self.condenced_module.add_module(
            "avg_pool1",
            sb.nnet.pooling.Pooling2d(
                pool_type=pool_type,
                kernel_size=condenced_depth_pool,
                stride=condenced_depth_pool,
                pool_axis=[1, 2],
            ),
        )

        self.condenced_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))
        if condenced_selection_kernels is None:
            condenced_selection_kernels = condenced_depth_kernels
        self.condenced_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=condenced_depth_kernels,
                out_channels=condenced_selection_kernels,
                kernel_size=condenced_selection_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=True,
                swap=True,
            ),
        )

        self.condenced_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=condenced_selection_kernels, momentum=0.01, affine=True,
            ),
        )

        self.condenced_module.add_module("act_1", activation)

        self.condenced_module.add_module(
            "avg_pool2",
            sb.nnet.pooling.Pooling2d(
                pool_type=pool_type,
                kernel_size=condenced_selection_pool,
                stride=condenced_selection_pool,
                pool_axis=[1, 2],
            ),
        )

        self.condenced_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))

        out = self.condenced_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.condenced_module(x)
        x = self.dense_module(x)
        return x
