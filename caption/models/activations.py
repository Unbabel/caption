# -*- coding: utf-8 -*-
import math

import torch
from torch import nn


def build_activation(activation: str):
    """ Builder function that returns a nn.module activation function.

    :param activation: string defining the name of the activation function.
        
    Activations available: 
        GELU, Swish + every native pytorch activation function.
    """
    if hasattr(nn, activation):
        return getattr(nn, activation)()
    elif activation == "Swish":
        return Swish()
    elif activation == "GELU":
        return GELU()
    else:
        raise Exception("{} invalid activation function.".format(activation))


def swish(input):
    """
    Applies Swish element-wise: A self-gated activation function 
        swish(x) = x * sigmoid(x)
    """
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    """
    Applies the Swish function element-wise:
    
        Swish(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1710.05941v1.pdf
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return swish(input)


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
            Also see https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class GELU(nn.Module):
    """
    Applies the GELU function element-wise:
    
        GELU(x) = 0.5*(1 + tanh(√2/π) * (x + 0.044715 * x^3))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return gelu(input)
