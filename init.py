"""
Initialization functions for neural network layers
"""

import torch
import torch.nn as nn
import numpy as np


def xavier_normal_initialization(module):
    """
    Xavier normal initialization for neural network parameters
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


def kaiming_normal_initialization(module):
    """
    Kaiming normal initialization for neural network parameters
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Embedding):
        nn.init.kaiming_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


def uniform_initialization(module, a=-0.1, b=0.1):
    """
    Uniform initialization for neural network parameters
    
    Args:
        module: PyTorch module to initialize
        a: Lower bound for uniform distribution
        b: Upper bound for uniform distribution
    """
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight.data, a, b)
    elif isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight.data, a, b)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


def normal_initialization(module, mean=0.0, std=0.1):
    """
    Normal initialization for neural network parameters
    
    Args:
        module: PyTorch module to initialize
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight.data, mean, std)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, mean, std)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
