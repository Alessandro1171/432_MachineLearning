"""
Based on Deep Learning models to use multiple Conv nodes to find larger patterns in the EEG pictures  and pooling was removed to see 
how it affects the accuracy 
Authors
 * Alessandro Dare, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torchvision import datasets, transforms
import speechbrain as sb


class Pattern_Expansion_DeepLearing(torch.nn.Module):
    def __init__(
        self,
        #InterperableDeepLearning
        input_shape=None,  # (1, T, C, 1)
        pedl_temporal_kernels=8,
        pedl_temporal_kernelsize=(33, 1),
        pedl_fillter_multiplier1=1.8,
        stride_kernel1=4,
        dilation_kernel=2,
        pedl_fillter_multiplier2=1.5,
        pedl_fillter_kernel_size2=(17, 1),
        stride_kernel2=3,
        pedl_fillter_multiplier3=1.2,
        pedl_fillter_kernel_size3=(9, 1),
        pedl_fillter_kernel4=None,
        pedl_fillter_kernel_size4=(1, 1),
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
        self.pedl_module = torch.nn.Sequential()
  
        self.pedl_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=pedl_temporal_kernels,
                kernel_size=pedl_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.pedl_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=pedl_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        pedl_fillter_kernel1 = round((
            pedl_fillter_multiplier1 * pedl_temporal_kernels
        ))
        self.pedl_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=pedl_temporal_kernels,
                out_channels=pedl_fillter_kernel1,
                kernel_size=(1,C),
                stride=stride_kernel1,
                dilation=dilation_kernel,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.pedl_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=pedl_fillter_kernel1, momentum=0.01, affine=True,
            ),
        )
        pedl_fillter_kernel2 = round((
            pedl_fillter_multiplier2 * pedl_fillter_kernel1
        ))
        self.pedl_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=pedl_fillter_kernel1,
                out_channels=pedl_fillter_kernel2,
                kernel_size=pedl_fillter_kernel_size2,
                stride=stride_kernel2,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.pedl_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=pedl_fillter_kernel2, momentum=0.01, affine=True,
            ),
        )
        self.pedl_module.add_module("act_1", activation)
        pedl_fillter_kernel3 = round((
            pedl_fillter_multiplier3 * pedl_fillter_kernel2
        ))
        self.pedl_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=pedl_fillter_kernel2,
                out_channels=pedl_fillter_kernel3,
                kernel_size=pedl_fillter_kernel_size3,
                stride=stride_kernel3,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.pedl_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=pedl_fillter_kernel3, momentum=0.01, affine=True,
            ),
        )
        self.pedl_module.add_module("act_2", activation)
        if pedl_fillter_kernel4 is None:
            pedl_fillter_kernel4 = pedl_fillter_kernel3

        self.pedl_module.add_module(
            "conv_4",
            sb.nnet.CNN.Conv2d(
                in_channels=pedl_fillter_kernel3,
                out_channels=pedl_fillter_kernel4,
                kernel_size=pedl_fillter_kernel_size4,
                stride=stride_kernel4,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        #print(torch.ones((1,) + tuple(input_shape[1:-1]) + (1,)))
        self.pedl_module.add_module("act_3", activation)
        
        out = self.pedl_module(
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
        self.filter_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))
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
        x = self.pedl_module(x)
        x = self.filter_module(x)
        return x
