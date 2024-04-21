"""EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class DeepLearning_RF(torch.nn.Module):
    """EEGNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution.
    cnn_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution.
    cnn_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution.
    cnn_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        dl_temporal_kernels=25,
        dl_kernelsize=(1, 10),
        dl_deep_kernelsize=(128,1),
        dl_depth_multiplier=1.5,
        dl_spatial_max_norm=1.0,
        dl_spatial_pool=(3, 1),
        dl_septemporal_kernels1=None,
        dl_septemporal_kernels2=None,
        dl_septemporal_kernels3=None,
        dl_pool_type="max",
        dropout=0.5,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
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
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        # T = input_shape[1]
        C = input_shape[2]

        # CONVOLUTIONAL MODULE
        self.dl_module = torch.nn.Sequential()
        # Temporal convolution
        self.dl_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=dl_temporal_kernels,
                kernel_size=dl_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.dl_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=dl_temporal_kernels, momentum=0.01, affine=True,
            ),
        )

        self.dl_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=dl_temporal_kernels,
                out_channels=dl_temporal_kernels,
                kernel_size=dl_deep_kernelsize,
                groups=dl_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=dl_spatial_max_norm,
                swap=True,
            ),
        )
        self.dl_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=dl_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        self.dl_module.add_module("act_0", activation)
        self.dl_module.add_module("dropout_0", torch.nn.Dropout(p=dropout))
        self.dl_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=dl_pool_type,
                kernel_size=dl_spatial_pool,
                stride=dl_spatial_pool,
                pool_axis=[1, 2],
            ),
        )

        # Temporal separable convolution
        dl_septemporal_kernels1 = round((
            dl_temporal_kernels * dl_depth_multiplier
        ))
        self.dl_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=dl_temporal_kernels,
                out_channels=dl_septemporal_kernels1,
                kernel_size=dl_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.dl_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=dl_septemporal_kernels1,
                momentum=0.01,
                affine=True,
            ),
        )
        dl_septemporal_kernels2 = round((
                dl_septemporal_kernels1 * dl_depth_multiplier
        ))
        self.dl_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=dl_septemporal_kernels1,
                out_channels=dl_septemporal_kernels2,
                kernel_size=dl_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        dl_septemporal_kernels3 = round((
                dl_septemporal_kernels2 * dl_depth_multiplier
        ))
        self.dl_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=dl_septemporal_kernels2,
                momentum=0.01,
                affine=True,
            ),
        )
        self.dl_module.add_module(
            "conv_4",
            sb.nnet.CNN.Conv2d(
                in_channels=dl_septemporal_kernels2,
                out_channels=dl_septemporal_kernels3,
                kernel_size=dl_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.dl_module.add_module(
            "bnorm_4",
            sb.nnet.normalization.BatchNorm2d(
                input_size=dl_septemporal_kernels3,
                momentum=0.01,
                affine=True,
            ),
        )
 

        # Shape of intermediate feature maps
        out = self.dl_module(
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
        x = self.dl_module(x)
        x = self.dense_module(x)
        return x
