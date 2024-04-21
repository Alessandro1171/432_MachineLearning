"""EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Davide Borra, 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torchvision import datasets, transforms
import speechbrain as sb


class IDL(torch.nn.Module):
    def __init__(
        self,
        #InterperableDeepLearning
        input_shape=None,  # (1, T, C, 1)
        idl_temporal_kernels=8,
        idl_temporal_kernelsize=(33, 1),
        idl_fillter_multiplier1=1.8,
        stride_kernel1=4,
        dilation_kernel=2,
        idl_fillter_multiplier2=1.5,
        idl_fillter_kernel_size2=(17, 1),
        stride_kernel2=3,
        idl_fillter_multiplier3=1.2,
        idl_fillter_kernel_size3=(9, 1),
        idl_fillter_kernel4=None,
        idl_fillter_kernel_size4=(1, 1),
        stride_kernel3=2,
        stride_kernel4=1,
        dropout=0.5,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="leaky_relu",
        ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128
        C = input_shape[2]
        self.ild_module = torch.nn.Sequential()
  
        self.ild_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=idl_temporal_kernels,
                kernel_size=idl_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.ild_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=idl_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        idl_fillter_kernel1 = round((
            idl_fillter_multiplier1 * idl_temporal_kernels
        ))
        self.ild_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=idl_temporal_kernels,
                out_channels=idl_fillter_kernel1,
                kernel_size=(1,C),
                stride=stride_kernel1,
                dilation=dilation_kernel,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.ild_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=idl_fillter_kernel1, momentum=0.01, affine=True,
            ),
        )
        idl_fillter_kernel2 = round((
            idl_fillter_multiplier2 * idl_fillter_kernel1
        ))
        self.ild_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=idl_fillter_kernel1,
                out_channels=idl_fillter_kernel2,
                kernel_size=idl_fillter_kernel_size2,
                stride=stride_kernel2,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.ild_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=idl_fillter_kernel2, momentum=0.01, affine=True,
            ),
        )
        self.ild_module.add_module("act_1", activation)
        idl_fillter_kernel3 = round((
            idl_fillter_multiplier3 * idl_fillter_kernel2
        ))
        self.ild_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=idl_fillter_kernel2,
                out_channels=idl_fillter_kernel3,
                kernel_size=idl_fillter_kernel_size3,
                stride=stride_kernel3,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.ild_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=idl_fillter_kernel3, momentum=0.01, affine=True,
            ),
        )
        self.ild_module.add_module("act_2", activation)
        if idl_fillter_kernel4 is None:
            idl_fillter_kernel4 = idl_fillter_kernel3

        self.ild_module.add_module(
            "conv_4",
            sb.nnet.CNN.Conv2d(
                in_channels=idl_fillter_kernel3,
                out_channels=idl_fillter_kernel4,
                kernel_size=idl_fillter_kernel_size4,
                stride=stride_kernel4,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        #print(torch.ones((1,) + tuple(input_shape[1:-1]) + (1,)))
        self.ild_module.add_module("act_3", activation)
        print(self.ild_module)
        out = self.ild_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.filter_module = torch.nn.Sequential()
        self.filter_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.filter_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.filter_module.add_module("act_out", torch.nn.Sigmoid())
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
        x = self.ild_module(x)
        x = self.filter_module(x)
        return x
