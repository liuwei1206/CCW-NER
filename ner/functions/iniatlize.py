__author__ = "liuwei"

"""
some init function
"""

import numpy as np
import torch
import torch.nn as nn

def init_cnn_weight(cnn_layer, seed=100):
    """
    init the weight of cnn net
    Args:
        cnn_layer: weight.size() = [out_channel, in_channels, kernei size]
        seed: int
    """
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(cnn_layer.weight)
    cnn_layer.bias.data.zero_()

def init_embedding(input_embedding, seed=100):
    """
    Args:
        input_embedding: the weight of embedding need to be init
        seed: int
    """
    # torch.manual_seed(seed)
    # scope = np.sqrt(3.0 / input_embedding.size(1))
    # nn.init.uniform_(input_embedding, -scope, scope)

    torch.manual_seed(seed)
    nn.init.normal_(input_embedding, 0, 0.1)

def init_linear(input_linear, seed=100):
    """
    init the weight of linear  net
    Args:
        input_linear: a linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_maxtrix_weight(weights, seed=100):
    """
    init the weight of a matrix
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (weights.size(0) + weights.size(1)))
    nn.init.uniform_(weights, -scope, scope)

def init_vector(vector, seed=100):
    """
    init a vector, note that vector is 1-D
    """
    torch.manual_seed(seed)
    v_size = vector.size(0)
    scale = np.sqrt(3.0 / v_size)
    nn.init.uniform_(vector, -scale, scale)

def init_highway(highway_layer, seed=100):
    """
    init the weight of highway net. do not init the bias
    Args:
        highway_layer: a highway layer
        seed:
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (highway_layer.weight.size(0) + highway_layer.weight.size(1)))
    nn.init.uniform_(highway_layer.weight, -scope, scope)


def get_embedding_weight(weight_file, vocab_size, word_dim):
    """
    get the embedding from weight file, and then use it to init embedding
    Args:
        weight_file: embedding weight file
        vocab_size: the size of vocab
        word_dim: the dim of word embedding
    """
