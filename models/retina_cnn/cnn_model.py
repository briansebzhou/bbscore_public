"""
CNN light response model. Copied from primate-retina-cnn-model/src/models.py
and adapted for BBScore (removed sys.path, use local misc_util).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import normal

from . import misc_util as mu


class CNN(nn.Module):
    """
    CNN light response model. Takes as input a config and produces either a
    3 layer CNN or an LN model (if the first two layers are disabled.)

    @param model_config
    @return model object

    @author Sam Cooler and Alex Gogliettino, with inspiration from
    Joshua Melander and the Baccus lab Deep Retina project:
    https://github.com/baccuslab/deep-retina/tree/master/deepretina
    """

    def __init__(self, model_config):
        super(CNN, self).__init__()
        self.name = 'CNN'

        # Unpack hyperparams from model config.
        history_frames = model_config['history_frames']
        self.enable_layer_0 = model_config['enable_layer_0']
        self.enable_layer_1 = model_config['enable_layer_1']
        self.layer_0_noise = model_config['layer_0_noise']
        self.layer_1_noise = model_config['layer_1_noise']
        self.nonlinearity = model_config['nonlinearity']
        conv0_channels = model_config['layer_0']['channels']
        conv1_channels = model_config['layer_1']['channels']
        conv0_kernel_size = model_config['layer_0']['kernel_size']
        conv1_kernel_size = model_config['layer_1']['kernel_size']
        stimulus_dim = model_config['stimulus_dim']
        self.device = model_config['device']
        n_cells = model_config['n_cells']

        """
        Initialize cell-specific shift and scale parameters, which helped with
        fitting lower firing rate cells.
        """
        self.input_scale = nn.Parameter(
                                data=torch.Tensor(
                                    [1 for a in range(n_cells)]),
                                requires_grad=True
                           )
        self.input_bias = nn.Parameter(
                                data=torch.Tensor(
                                    [0 for a in range(n_cells)]),
                                requires_grad=True
                          )
        self.output_scale = nn.Parameter(
                                data=torch.Tensor(
                                    model_config['output_scale_initialize']),
                                requires_grad=True
                            )

        # Figure out the shapes of inputs/outputs to each layer.
        if not self.enable_layer_0:
            conv0_out_dim = (history_frames,) + stimulus_dim
        else:
            conv0_out_dim = (conv0_channels,)
            conv0_out_dim += mu.get_conv_output_shape(
                                    stimulus_dim,
                                    conv0_kernel_size
                             )

        if not self.enable_layer_1:
            conv1_out_dim = conv0_out_dim
        else:
            conv1_out_dim = (conv1_channels,)
            conv1_out_dim += mu.get_conv_output_shape(
                                conv0_out_dim[1:],
                                conv1_kernel_size
                            )

        # Define each layer in the network.
        self.conv0 = nn.Conv2d(
                            history_frames,
                            conv0_channels,
                            kernel_size=conv0_kernel_size
                     )
        self.batch0 = nn.BatchNorm1d(np.prod(conv0_out_dim))

        self.conv1 = nn.Conv2d(
                            conv0_channels,
                            conv1_channels,
                            kernel_size=conv1_kernel_size
                    )
        self.batch1 = nn.BatchNorm1d(np.prod(conv1_out_dim))

        self.linear = nn.Linear(
                            np.prod(conv1_out_dim),
                            n_cells,
                            bias=False
                    )

    def gaussian(self, x, sigma):
        noise = normal.Normal(
                        torch.zeros(x.size()),
                        sigma*torch.ones(x.size())
                )

        return x + noise.sample().to(self.device)

    def forward(self, x):
        x_dim = x.shape[1:]

        if self.enable_layer_0:
            x = self.conv0(x)
            x_dim = x.shape[1:]
            x = self.batch0(x.view(-1, np.prod(x_dim)))
            x = x.view(-1, *x_dim)

            if self.training:
                x = self.gaussian(x, self.layer_0_noise)

            x = nn.functional.relu(x)

        if self.enable_layer_1:
            x = self.conv1(x)
            x_dim = x.shape[1:]
            x = self.batch1(x.view(-1, np.prod(x_dim)))
            x = x.view(-1, *x_dim)

            if self.training:
                x = self.gaussian(x, self.layer_1_noise)

            x = nn.functional.relu(x)

        x = self.linear(x.view(-1, np.prod(x_dim)))
        x = torch.mul(x + self.input_bias, self.input_scale)

        if self.nonlinearity == 'softplus':
            x = nn.functional.softplus(x)
        else:
            assert False, 'Only softplus nonlinearity supported.'

        x = torch.mul(x, self.output_scale)

        return x
