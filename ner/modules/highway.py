__author__ = 'liuwei'

"""
implements of the highway net
include two gate: transform gate(G_T) and carry gate(G_C)
    H = w_h * x + b_h
    G_T = sigmoid(w_t * x + b_t)
    G_C = sigmoid(w_c * x + b_c)
outputs:
    outputs = G_T * H + G_C * x

for sample:
    G_C = (1 - G_T), then:
    outputs = G_T * H + (1 - G_T) * x
    and generally set b_c = -1 or -3, that mean set b_t = 1 or 3
"""

import torch
import torch.nn as nn
import numpy as np

from ner.functions.iniatlize import init_highway

class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1, activation=nn.functional.relu,
                 require_grad=True):
        """
        Args:
            input_dim: the dim
            num_layers: the numer of highway layers
            activation: activation function, tanh or relu
        """
        super(Highway, self).__init__()

        self._input_dim = input_dim
        self._num_layers = num_layers

        # output is input_dim * 2, because one is candidate status, and another
        # is transform gate
        self._layers = torch.nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        i = 0
        for layer in self._layers:
            layer.weight.requires_grad = require_grad
            layer.bias.requires_grad = require_grad
            init_highway(layer, 100)
            layer.bias[input_dim:].data.fill_(1)

            i += 1


    def forward(self, inputs):
        """
        Args:
            inputs: a tensor, size is [batch_size, n_tokens, input_dim]
        """
        current_input = inputs
        for layer in self._layers:
            proj_inputs = layer(current_input)
            linear_part = current_input

            del current_input

            # here the gate is carry gate, if you change it to transform gate
            # the bias init should change too, maybe -1 or -3 even
            nonlinear_part, carry_gate = proj_inputs.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            carry_gate = torch.nn.functional.sigmoid(carry_gate)
            current_input = (1 - carry_gate) * nonlinear_part + carry_gate * linear_part

        return current_input

