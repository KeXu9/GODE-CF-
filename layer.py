import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Union
import numpy as np


class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN
    
    Implements bi-interaction mechanism for collaborative filtering:
    output = (L+I)EW_1 + LE ⊙ EW_2
    
    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
        
    Args:
        in_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        dropout (float, optional): Dropout probability. Defaults to 0.0
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True
        activation (str, optional): Activation function name. Defaults to None
        gradient_clip (float, optional): Gradient clipping value. Defaults to None
        numerical_stability (bool, optional): Apply numerical stability checks. Defaults to True
        cache_embeddings (bool, optional): Cache intermediate embeddings. Defaults to False
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, bias: bool = True, 
                 activation: Optional[str] = None, gradient_clip: Optional[float] = None,
                 numerical_stability: bool = True, cache_embeddings: bool = False):
        super(BiGNNLayer, self).__init__()
        
        # Input validation
        self._validate_inputs(in_dim, out_dim, dropout)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.gradient_clip = gradient_clip
        self.numerical_stability = numerical_stability
        self.cache_embeddings = cache_embeddings
        
        # Linear transformation for the sum part: (L+I)EW_1
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        
        # Linear transformation for the interaction part: LE ⊙ EW_2
        self.interActTransform = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Activation function
        self.activation_fn = self._get_activation_function(activation)
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(out_dim) if numerical_stability else None
        
        # Initialize weights with proper strategy
        self._init_weights()
        
        # Cache for embeddings if enabled
        self._embedding_cache = {} if cache_embeddings else None
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _validate_inputs(self, in_dim: int, out_dim: int, dropout: float) -> None:
        """Validate input parameters"""
        if not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer, got {in_dim}")
        if not isinstance(out_dim, int) or out_dim <= 0:
            raise ValueError(f"out_dim must be a positive integer, got {out_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
            
    def _get_activation_function(self, activation: Optional[str]):
        """Get activation function by name with validation"""
        if activation is None:
            return None
        
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x),
            'mish': lambda x: x * torch.tanh(F.softplus(x))
        }
        
        if activation.lower() not in activation_map:
            available = ', '.join(activation_map.keys())
            raise ValueError(f"Unsupported activation '{activation}'. Available: {available}")
            
        return activation_map[activation.lower()]
    
    def _init_weights(self) -> None:
        """Initialize layer weights with proper strategy"""
        # Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.interActTransform.weight, gain=nn.init.calculate_gain('linear'))
        
        # Initialize biases to small positive values for stability
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.01)
        if self.interActTransform.bias is not None:
            nn.init.constant_(self.interActTransform.bias, 0.01)
    
    def _apply_numerical_stability(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply numerical stability checks and corrections"""
        if not self.numerical_stability:
            return tensor
            
        # Check for NaN values
        if torch.isnan(tensor).any():
            warnings.warn("NaN values detected in tensor, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0)
            
        # Check for infinite values
        if torch.isinf(tensor).any():
            warnings.warn("Infinite values detected in tensor, clipping")
            tensor = torch.clamp(tensor, min=-1e6, max=1e6)
            
        # Apply gradient clipping if specified
        if self.gradient_clip is not None and tensor.requires_grad:
            tensor.register_hook(lambda grad: torch.clamp(grad, -self.gradient_clip, self.gradient_clip))
            
        return tensor
    
    def _get_cache_key(self, lap_matrix: torch.Tensor, features: torch.Tensor) -> str:
        """Generate cache key for embeddings"""
        if not self.cache_embeddings:
            return None
            
        # Simple hash-based cache key (could be improved for production)
        lap_hash = hash(tuple(lap_matrix._values().cpu().numpy().flatten()[:100]))  # Sample for efficiency
        feat_hash = hash(tuple(features.flatten()[:100].cpu().numpy()))
        return f"{lap_hash}_{feat_hash}_{features.shape}"
    
    def _sparse_mm_safe(self, sparse_matrix: torch.Tensor, dense_matrix: torch.Tensor) -> torch.Tensor:
        """Safe sparse matrix multiplication with error handling"""
        try:
            # Validate dimensions
            if sparse_matrix.size(1) != dense_matrix.size(0):
                raise ValueError(f"Matrix dimension mismatch: {sparse_matrix.shape} x {dense_matrix.shape}")
                
            # Check for empty matrices
            if sparse_matrix._nnz() == 0:
                warnings.warn("Sparse matrix is empty, returning zeros")
                return torch.zeros(sparse_matrix.size(0), dense_matrix.size(1), 
                                 device=dense_matrix.device, dtype=dense_matrix.dtype)
                
            # Perform sparse matrix multiplication
            result = torch.sparse.mm(sparse_matrix, dense_matrix)
            
            # Apply numerical stability
            result = self._apply_numerical_stability(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Sparse matrix multiplication failed: {str(e)}"
            raise RuntimeError(error_msg) from e

    def forward(self, lap_matrix: torch.Tensor, eye_matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BiGNN layer with enhanced error handling and caching
        
        Args:
            lap_matrix: Sparse Laplacian matrix (N+M, N+M) where N=users, M=items
            eye_matrix: Identity matrix (currently unused but kept for compatibility)
            features: Node features/embeddings (N+M, in_dim)
            
        Returns:
            Updated node embeddings (N+M, out_dim)
            
        Raises:
            ValueError: If input dimensions are incompatible
            RuntimeError: If computation fails
        """
        # Input validation
        if features.size(1) != self.in_dim:
            raise ValueError(f"Expected input features dim {self.in_dim}, got {features.size(1)}")
            
        if lap_matrix.size(0) != lap_matrix.size(1):
            raise ValueError(f"Laplacian matrix must be square, got {lap_matrix.shape}")
            
        if lap_matrix.size(0) != features.size(0):
            raise ValueError(f"Matrix size mismatch: lap_matrix {lap_matrix.shape}, features {features.shape}")
        
        # Check cache if enabled
        cache_key = self._get_cache_key(lap_matrix, features)
        if cache_key and cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        else:
            self._cache_misses += 1
        
        try:
            # Graph convolution: Lx (propagate information through graph)
            x = self._sparse_mm_safe(lap_matrix, features)
            
            # Sum part: (L+I)EW_1 = (Lx + x)W_1
            sum_embeddings = features + x
            sum_embeddings = self._apply_numerical_stability(sum_embeddings)
            inter_part1 = self.linear(sum_embeddings)
            
            # Interaction part: LE ⊙ EW_2 = (Lx ⊙ x)W_2
            inter_feature = torch.mul(x, features)
            inter_feature = self._apply_numerical_stability(inter_feature)
            inter_part2 = self.interActTransform(inter_feature)
            
            # Combine both parts
            output = inter_part1 + inter_part2
            output = self._apply_numerical_stability(output)
            
            # Apply batch normalization if enabled
            if self.batch_norm is not None and output.dim() == 2:
                # Reshape for batch norm if needed
                if output.size(0) > 1:  # Only apply if batch size > 1
                    output = self.batch_norm(output)
            
            # Apply dropout if specified and in training mode
            if self.dropout_layer is not None and self.training:
                output = self.dropout_layer(output)
            
            # Apply activation if specified
            if self.activation_fn is not None:
                output = self.activation_fn(output)
            
            # Cache result if enabled
            if cache_key and self._embedding_cache is not None:
                # Limit cache size to prevent memory issues
                if len(self._embedding_cache) < 100:  # Configurable limit
                    self._embedding_cache[cache_key] = output.detach()
                    
            return output
            
        except Exception as e:
            error_msg = f"Forward pass failed in BiGNNLayer: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def reset_cache(self) -> None:
        """Reset embedding cache and statistics"""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._embedding_cache) if self._embedding_cache else 0
        }
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, dropout={self.dropout}, '
                f'gradient_clip={self.gradient_clip}, numerical_stability={self.numerical_stability}')


class SparseDropout(nn.Module):
    """
    Enhanced dropout layer for sparse tensors with better error handling and performance
    
    Applies dropout to sparse tensor values while maintaining sparsity structure.
    During training, randomly sets some values to zero and scales remaining values.
    
    Args:
        p (float): Dropout probability (0 <= p < 1). Default: 0.5
        inplace (bool): Whether to do operation in-place. Default: False
        numerical_stability (bool): Apply numerical stability checks. Default: True
        use_bernoulli (bool): Use Bernoulli distribution for more stable sampling. Default: True
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, numerical_stability: bool = True,
                 use_bernoulli: bool = True):
        super(SparseDropout, self).__init__()
        
        # Input validation
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        
        self.p = p
        self.keep_prob = 1.0 - p
        self.inplace = inplace
        self.numerical_stability = numerical_stability
        self.use_bernoulli = use_bernoulli
        
        # Statistics tracking
        self._dropout_count = 0
        self._total_elements = 0
    
    def _validate_sparse_tensor(self, x: torch.Tensor) -> None:
        """Validate input sparse tensor"""
        if not x.is_sparse:
            raise ValueError("SparseDropout only works with sparse tensors")
            
        if x._nnz() == 0:
            warnings.warn("Input sparse tensor is empty")
            
        # Check for invalid values
        if self.numerical_stability:
            values = x._values()
            if torch.isnan(values).any():
                raise ValueError("Input tensor contains NaN values")
            if torch.isinf(values).any():
                raise ValueError("Input tensor contains infinite values")
    
    def _generate_dropout_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Generate dropout mask with better sampling strategy"""
        if self.use_bernoulli:
            # Use Bernoulli distribution for more stable training
            keep_mask = torch.bernoulli(torch.full_like(values, self.keep_prob)).bool()
        else:
            # Original random-based approach
            keep_mask = ((torch.rand(values.size(), device=values.device) + self.keep_prob)
                        .floor()).type(torch.bool)
        
        return keep_mask
    
    def _apply_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to sparse tensor with error handling"""
        try:
            # Get sparse tensor components
            indices = x._indices()
            values = x._values()
            shape = x.shape
            
            # Generate dropout mask
            keep_mask = self._generate_dropout_mask(values)
            
            # Apply mask to indices and values
            kept_indices = indices[:, keep_mask]
            kept_values = values[keep_mask]
            
            # Scale remaining values to maintain expected sum
            if self.keep_prob > 0:
                kept_values = kept_values / self.keep_prob
            
            # Apply numerical stability if enabled
            if self.numerical_stability:
                kept_values = torch.clamp(kept_values, min=-1e6, max=1e6)
            
            # Update statistics
            self._dropout_count += (values.numel() - kept_values.numel())
            self._total_elements += values.numel()
            
            # Create new sparse tensor
            result = torch.sparse_coo_tensor(kept_indices, kept_values, shape,
                                           dtype=x.dtype, device=x.device)
            
            # Coalesce for efficiency
            return result.coalesce()
            
        except Exception as e:
            error_msg = f"Dropout application failed: {str(e)}"
            raise RuntimeError(error_msg) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to sparse tensor with enhanced error handling
        
        Args:
            x: Input sparse tensor
            
        Returns:
            Sparse tensor with dropout applied
            
        Raises:
            ValueError: If input is not a valid sparse tensor
            RuntimeError: If dropout computation fails
        """
        # Validate input
        self._validate_sparse_tensor(x)
        
        # No dropout during evaluation or if p=0
        if not self.training or self.p == 0:
            return x
        
        # Apply dropout
        return self._apply_dropout(x)
    
    def get_dropout_stats(self) -> dict:
        """Get dropout statistics"""
        dropout_rate = self._dropout_count / self._total_elements if self._total_elements > 0 else 0.0
        
        return {
            'dropout_count': self._dropout_count,
            'total_elements': self._total_elements,
            'actual_dropout_rate': dropout_rate,
            'expected_dropout_rate': self.p
        }
    
    def reset_stats(self) -> None:
        """Reset dropout statistics"""
        self._dropout_count = 0
        self._total_elements = 0
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (f'p={self.p}, inplace={self.inplace}, '
                f'numerical_stability={self.numerical_stability}, '
                f'use_bernoulli={self.use_bernoulli}')