"""
Weight Initialization Functions for Graph Neural Networks
Optimized for numerical stability and faster convergence
"""

import torch
import torch.nn as nn
import math
import warnings


def xavier_normal_initialization(module):
    """
    Enhanced Xavier/Glorot normal initialization with stability improvements
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Calculate fan_in and fan_out
        fan_in = module.weight.size(1)
        fan_out = module.weight.size(0)
        
        # Use Xavier normal with adjusted gain for better stability
        if hasattr(module, 'activation'):
            # Adjust gain based on activation function
            if module.activation == 'relu':
                gain = math.sqrt(2.0)  # He initialization for ReLU
            elif module.activation == 'leaky_relu':
                gain = math.sqrt(2.0 / (1 + 0.01**2))
            else:
                gain = 1.0  # Default Xavier gain
        else:
            gain = 1.0
        
        # Apply Xavier normal initialization
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        with torch.no_grad():
            module.weight.normal_(0, std)
            
        # Initialize bias with small positive values for stability
        if module.bias is not None:
            with torch.no_grad():
                module.bias.fill_(0.01)
                
    elif isinstance(module, nn.Embedding):
        # Embedding initialization with reduced variance for stability
        embedding_dim = module.weight.size(1)
        std = 0.1 / math.sqrt(embedding_dim)  # Reduced variance
        
        with torch.no_grad():
            module.weight.normal_(0, std)
            
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        # Batch normalization initialization
        if module.weight is not None:
            module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def he_normal_initialization(module):
    """
    He/Kaiming normal initialization optimized for ReLU activations
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(1)
        std = math.sqrt(2.0 / fan_in)
        
        with torch.no_grad():
            module.weight.normal_(0, std)
            
        if module.bias is not None:
            with torch.no_grad():
                module.bias.fill_(0.01)
                
    elif isinstance(module, nn.Embedding):
        embedding_dim = module.weight.size(1)
        std = math.sqrt(2.0 / embedding_dim)
        
        with torch.no_grad():
            module.weight.normal_(0, std)


def uniform_initialization(module, scale=0.1):
    """
    Uniform initialization with configurable scale
    
    Args:
        module: PyTorch module to initialize
        scale: Scale factor for uniform distribution
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        with torch.no_grad():
            module.weight.uniform_(-scale, scale)
            
        if hasattr(module, 'bias') and module.bias is not None:
            with torch.no_grad():
                module.bias.fill_(0.01)


def orthogonal_initialization(module, gain=1.0):
    """
    Orthogonal initialization for better gradient flow
    
    Args:
        module: PyTorch module to initialize
        gain: Gain factor for the orthogonal matrix
    """
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            nn.init.orthogonal_(module.weight, gain=gain)
            
        if module.bias is not None:
            with torch.no_grad():
                module.bias.fill_(0.01)
                
    elif isinstance(module, nn.Embedding):
        # For embeddings, use Xavier as orthogonal may not be appropriate
        xavier_normal_initialization(module)


def stable_initialization(module, method='xavier', **kwargs):
    """
    Stable initialization with multiple methods and validation
    
    Args:
        module: PyTorch module to initialize
        method: Initialization method ('xavier', 'he', 'uniform', 'orthogonal')
        **kwargs: Additional arguments for specific methods
    """
    try:
        if method == 'xavier':
            xavier_normal_initialization(module)
        elif method == 'he':
            he_normal_initialization(module)
        elif method == 'uniform':
            scale = kwargs.get('scale', 0.1)
            uniform_initialization(module, scale=scale)
        elif method == 'orthogonal':
            gain = kwargs.get('gain', 1.0)
            orthogonal_initialization(module, gain=gain)
        else:
            warnings.warn(f"Unknown initialization method: {method}, using Xavier")
            xavier_normal_initialization(module)
            
        # Validate initialization (check for NaN/Inf)
        for param in module.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                warnings.warn("Invalid values detected after initialization, re-initializing with safe defaults")
                with torch.no_grad():
                    param.fill_(0.01)
                    
    except Exception as e:
        warnings.warn(f"Initialization failed: {e}, using safe fallback")
        # Safe fallback initialization
        for param in module.parameters():
            with torch.no_grad():
                param.normal_(0, 0.01)


def smart_initialization(model, model_type='gcn'):
    """
    Smart initialization based on model type and architecture
    
    Args:
        model: PyTorch model to initialize
        model_type: Type of model ('gcn', 'ode', 'transformer', etc.)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'output' in name.lower() or 'classifier' in name.lower():
                # Output layers: use smaller variance for stability
                std = 0.01
                with torch.no_grad():
                    module.weight.normal_(0, std)
                    if module.bias is not None:
                        module.bias.fill_(0.0)
                        
            elif 'embedding' in name.lower():
                # Embedding-like linear layers
                xavier_normal_initialization(module)
                
            elif model_type == 'ode':
                # ODE models: use very small weights for stability
                std = 0.001
                with torch.no_grad():
                    module.weight.normal_(0, std)
                    if module.bias is not None:
                        module.bias.fill_(0.001)
                        
            else:
                # Standard layers
                xavier_normal_initialization(module)
                
        elif isinstance(module, nn.Embedding):
            # Embeddings: always use careful initialization
            embedding_dim = module.weight.size(1)
            std = min(0.1, 1.0 / math.sqrt(embedding_dim))
            
            with torch.no_grad():
                module.weight.normal_(0, std)
                
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            # Normalization layers
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


class InitializationConfig:
    """Configuration class for different initialization strategies"""
    
    # Predefined configurations for different model types
    CONFIGS = {
        'lightgcn': {
            'embedding_std': 0.1,
            'linear_method': 'xavier',
            'bias_value': 0.0
        },
        'ode_cf': {
            'embedding_std': 0.01,
            'linear_method': 'xavier',
            'linear_gain': 0.5,
            'bias_value': 0.001
        },
        'ultragcn': {
            'embedding_std': 0.1,
            'linear_method': 'he',
            'bias_value': 0.01
        },
        'ngcf': {
            'embedding_std': 0.1,
            'linear_method': 'xavier',
            'bias_value': 0.0
        }
    }
    
    @classmethod
    def get_config(cls, model_name):
        """Get initialization config for specific model"""
        return cls.CONFIGS.get(model_name.lower(), cls.CONFIGS['lightgcn'])
    
    @classmethod
    def apply_config(cls, model, model_name):
        """Apply initialization config to model"""
        config = cls.get_config(model_name)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                std = config['embedding_std']
                with torch.no_grad():
                    module.weight.normal_(0, std)
                    
            elif isinstance(module, nn.Linear):
                method = config['linear_method']
                gain = config.get('linear_gain', 1.0)
                
                if method == 'xavier':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif method == 'he':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    
                if module.bias is not None:
                    bias_value = config['bias_value']
                    with torch.no_grad():
                        module.bias.fill_(bias_value)
