"""
Model based on EEGNet using RNN (either lstm or gru) instead of Conv to translate eeg brain 
imagery from subjects and classify metrics data for one of the four movements based on subject data
Authors
 * Alessandro Dare, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from torchvision import datasets, transforms
import speechbrain as sb
import math

class CRDNN_V3(torch.nn.Module):
    def __init__(
            self,
            input_shape=None,
            crdnn_temporal_kernels=61,
            crdnn_temporal_kernelsize=(33, 1),
            crdnn_spatial_depth_multiplier=2,
            crdnn_spatial_max_norm=1.,
            crdnn_pool_type="avg",
            crdnn_spatial_pool=(4, 1),
            rnn_multiplier1=0.5,
            rnn_shape1=4,
            rnn_shape2=4,
            activation="elu",
            rnn_type="gru",
            dropout=0.5,
            crdnn_flatten_pool=(8, 1),
            dense_max_norm=0.25,
            dense_n_neurons=4,
            hidden_size1=8,
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
        
        self.crdnn_module1 = torch.nn.Sequential()
        # Temporal convolution
        self.crdnn_module1.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=crdnn_temporal_kernels,
                kernel_size=crdnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.crdnn_module1.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=crdnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        # Spatial depthwise convolution
        crdnn_spatial_kernels = (
            crdnn_spatial_depth_multiplier * crdnn_temporal_kernels
        )
        
        self.crdnn_module1.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=crdnn_temporal_kernels,
                out_channels=crdnn_spatial_kernels,
                kernel_size=(1, C),
                groups=crdnn_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=crdnn_spatial_max_norm,
                swap=True,
            ),
        )
        
        self.crdnn_module1.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=crdnn_spatial_kernels, momentum=0.01, affine=True,
            ),
        )
       
        self.crdnn_module1.add_module("act_1", activation)
   
        self.crdnn_module1.add_module(
            "pool_0",
            sb.nnet.pooling.Pooling2d(
                pool_type=crdnn_pool_type,
                kernel_size=crdnn_spatial_pool,
                stride=crdnn_spatial_pool,
                pool_axis=[1, 2],
            ),
        )
        
        self.crdnn_module1.add_module("dropout_0", torch.nn.Dropout(p=dropout))



        hidden_size1 = (
           round(crdnn_spatial_kernels * rnn_multiplier1)
        )

        if rnn_type == "lstm":
            self.crdnn_module1.add_module(
                "RNN_1",
                sb.nnet.RNN.LSTM(
                    input_shape=[len(input_shape), 125, 1, crdnn_spatial_kernels],
                    num_layers=rnn_shape1,
                    hidden_size=hidden_size1,
                    dropout=dropout,
                    bias=False,
                ),
            )
        elif rnn_type == "gru":
            self.crdnn_module1.add_module(
                "RNN_1",
                sb.nnet.RNN.GRU(
                    input_shape=[len(input_shape), 125, 1, crdnn_spatial_kernels],
                    num_layers=rnn_shape1,
                    hidden_size=hidden_size1,
                    dropout=dropout,
                    bias=False,
                ),
            )
        self.crdnn_module2 = torch.nn.Sequential()  
        self.crdnn_module2.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=hidden_size1,
                momentum=0.01,
                affine=True,
            ),
        )
        
        hidden_size2 = hidden_size1
        if rnn_type=="lstm":
            self.crdnn_module2.add_module(
                "RNN_2",
                sb.nnet.RNN.LSTM(
                    input_shape=[len(input_shape), 125, 1, hidden_size1],
                    num_layers=rnn_shape2,
                    hidden_size=hidden_size2,
                    dropout=dropout,
                    bias=False,
                    bidirectional=True,
                ),
            )
        elif rnn_type=="gru":
            self.crdnn_module2.add_module(
                "RNN_2",
                sb.nnet.RNN.GRU(
                    input_shape=[len(input_shape), 125, 1, hidden_size1],
                    num_layers=rnn_shape2,
                    hidden_size=hidden_size2,
                    dropout=dropout,
                    bias=False,
                    bidirectional=True,
                ),
            )

        self.crdnn_module3 = torch.nn.Sequential()
       
        hidden_size3 = hidden_size2*2
        self.crdnn_module3.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=hidden_size3,
                momentum=0.01,
                affine=True,
            ),
        )
        
        self.crdnn_module3.add_module("act_2", activation)
        self.crdnn_module3.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=crdnn_pool_type,
                kernel_size=crdnn_flatten_pool,
                stride=crdnn_flatten_pool,
                pool_axis=[1, 2],
            ),
        )
        self.crdnn_module3.add_module("dropout_1", torch.nn.Dropout(p=dropout))
      
        
        out1 = self.crdnn_module1(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        
        out = out1[0].unsqueeze(2)  # Add dummy dimension for 
       
        out2= self.crdnn_module2(out)
       
        out3= self.crdnn_module3(out2[0].unsqueeze(2))
        dense_input_size = self._num_flat_features(out3)
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
        
        x = self.crdnn_module1(x)
        x = x[0].unsqueeze(2)  # Add dummy dimension for 
        x= self.crdnn_module2(x)
        x= self.crdnn_module3(x[0].unsqueeze(2))
        x = self.dense_module(x)
        return x
