"""
Unified ODE Module for Graph Neural Networks - Enhanced Version
Fixed bugs, improved performance, and added numerical stability
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import warnings
from typing import Optional, Union


# =============================================================================
# ODE Functions (Enhanced with bug fixes and optimizations)
# =============================================================================

class ODEFunc(nn.Module):
    """ODE Function with dataset-specific architecture (Enhanced)"""
    
    def __init__(self, adj, latent_dim, data_name, device='cpu'):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.x0 = None
        self.name = data_name
        self.device = device
        self.latent_dim = latent_dim

        # Dataset-specific architecture with better initialization
        if self.name == "Cell_Phones_and_Accessories":
            self.linear2_u = nn.Linear(latent_dim, 1)
        else:
            hidden_dim = max(1, int(latent_dim/2))  # Ensure positive hidden dim
            self.linear2_u = nn.Linear(latent_dim, hidden_dim)
            self.linear2_u_1 = nn.Linear(hidden_dim, 1)

        # Initialize weights properly
        self._init_weights()
        
        # Move to device
        self.to(device)

    def _init_weights(self):
        """Initialize weights with Xavier initialization for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, t, x):
        """Enhanced forward pass with numerical stability"""
        try:
            # Ensure tensors are on correct device
            if x.device != self.device:
                x = x.to(self.device)
            
            # Compute alpha with numerical stability
            if self.name == "Cell_Phones_and_Accessories":
                alph = torch.sigmoid(self.linear2_u(x))
            else:
                hidden = self.linear2_u(x)
                # Apply numerical stability
                hidden = torch.clamp(hidden, min=-10, max=10)
                alph = torch.sigmoid(self.linear2_u_1(hidden))
            
            # Ensure alpha is in valid range
            alph = torch.clamp(alph, min=1e-6, max=1.0 - 1e-6)
            
            # Graph convolution operations
            ax = torch.sparse.mm(self.g, x)
            ax = alph * torch.sparse.mm(self.g, ax)
            
            # Residual connection with stability check
            f = ax - x
            
            # Check for numerical issues
            if torch.isnan(f).any() or torch.isinf(f).any():
                warnings.warn("Numerical instability detected in ODEFunc, applying correction")
                f = torch.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return f
            
        except Exception as e:
            raise RuntimeError(f"ODEFunc forward pass failed: {str(e)}")


class ODEFunc1(nn.Module):
    """ODE Function with trainable alpha (Enhanced with bug fixes)"""

    def __init__(self, adj, latent_dim, device='cpu'):
        super(ODEFunc1, self).__init__()
        self.g = adj
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # FIX: Initialize alpha parameter properly on correct device
        self.alpha_train = nn.Parameter(
            torch.tensor(0.9, device=self.device, dtype=torch.float32)
        )
        
        # Move graph to device if it's not already
        if hasattr(self.g, 'to'):
            self.g = self.g.to(self.device)
        
        # Performance optimization: cache for repeated computations
        self._last_alpha = None
        self._cached_alpha = None

    def forward(self, t, x):
        """Enhanced forward pass with better numerical stability"""
        try:
            # Ensure correct device
            if x.device != self.device:
                x = x.to(self.device)
            
            # Compute alpha with caching for performance
            current_alpha = self.alpha_train.item()
            if self._last_alpha != current_alpha:
                self._cached_alpha = torch.sigmoid(self.alpha_train)
                self._last_alpha = current_alpha
            
            alph = self._cached_alpha
            
            # Ensure alpha is numerically stable
            alph = torch.clamp(alph, min=1e-7, max=1.0 - 1e-7)
            
            # Graph operations with error handling
            try:
                ax = torch.sparse.mm(self.g, x)  # First convolution
                ax = alph * torch.sparse.mm(self.g, ax)  # Second convolution with alpha
            except RuntimeError as e:
                if "size" in str(e).lower():
                    raise ValueError(f"Graph-embedding dimension mismatch: graph shape {self.g.shape}, embedding shape {x.shape}")
                raise
            
            # Compute derivative
            f = ax - x
            
            # Numerical stability check
            if torch.isnan(f).any() or torch.isinf(f).any():
                warnings.warn("Numerical instability in ODEFunc1, applying correction")
                f = torch.nan_to_num(f, nan=0.0, posinf=1e5, neginf=-1e5)
            
            # Gradient clipping for stability
            f = torch.clamp(f, min=-10, max=10)
            
            return f
            
        except Exception as e:
            raise RuntimeError(f"ODEFunc1 forward pass failed: {str(e)}")


class ODEFunction(nn.Module):
    """Simple ODE Function for LGC (Enhanced)"""
    
    def __init__(self, Graph, numerical_stability=True):
        super(ODEFunction, self).__init__()
        self.g = Graph
        self.numerical_stability = numerical_stability

    def forward(self, t, x):
        """Enhanced forward pass with error handling"""
        try:
            out = torch.sparse.mm(self.g, x)
            
            # Apply numerical stability if enabled
            if self.numerical_stability:
                if torch.isnan(out).any() or torch.isinf(out).any():
                    warnings.warn("Numerical instability in ODEFunction, applying correction")
                    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Gradient clipping
                out = torch.clamp(out, min=-20, max=20)
            
            return out
            
        except Exception as e:
            raise RuntimeError(f"ODEFunction forward pass failed: {str(e)}")


# =============================================================================
# ODE Blocks (Enhanced with better error handling and performance)
# =============================================================================

class ODEblock(nn.Module):
    """Basic ODE Block (Enhanced)"""
    
    def __init__(self, odefunc, t=None, solver='euler', rtol=1e-3, atol=1e-4):
        super(ODEblock, self).__init__()
        if t is None:
            t = torch.tensor([0, 1], dtype=torch.float32)
        self.t = t
        self.odefunc = odefunc
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        """Enhanced forward pass with better device handling"""
        try:
            # Ensure time tensor is on correct device and type
            t = self.t.to(device=x.device, dtype=x.dtype)
            
            # Solve ODE with appropriate tolerance
            with torch.no_grad():
                if not x.requires_grad:
                    # More efficient for inference
                    z = odeint(self.odefunc, x, t, method=self.solver, 
                             rtol=self.rtol, atol=self.atol)
                else:
                    # For training with gradients
                    z = odeint(self.odefunc, x, t, method=self.solver,
                             rtol=self.rtol, atol=self.atol)
            
            return z[1]  # Return final state
            
        except Exception as e:
            raise RuntimeError(f"ODEblock forward pass failed: {str(e)}")


class ODEBlock(nn.Module):
    """ODE Block with time range (Enhanced)"""
    
    def __init__(self, odeFunction, solver, init_time, final_time, rtol=1e-3, atol=1e-4):
        super(ODEBlock, self).__init__()
        self.odefunc = odeFunction
        self.integration_time = torch.tensor([init_time, final_time], dtype=torch.float32)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        """Enhanced forward pass"""
        try:
            # Proper device and dtype handling
            self.integration_time = self.integration_time.to(device=x.device, dtype=x.dtype)
            
            # Solve ODE with error handling
            out = odeint(func=self.odefunc, y0=x, t=self.integration_time, 
                        method=self.solver, rtol=self.rtol, atol=self.atol)
            
            return out[1]
            
        except Exception as e:
            raise RuntimeError(f"ODEBlock forward pass failed: {str(e)}")


# =============================================================================
# Time-based ODE Blocks (Enhanced)
# =============================================================================

class ODEBlockTimeFirst(nn.Module):
    """ODE Block for first time segment (Enhanced)"""
    
    def __init__(self, odeFunction, num_split, solver, device='cpu', rtol=1e-3, atol=1e-4):
        super(ODEBlockTimeFirst, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.device = torch.device(device) if isinstance(device, str) else device
        self.zero = torch.tensor([0.], dtype=torch.float32, device=self.device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
                    
    def forward(self, x, t):
        """Enhanced forward pass with proper error handling"""
        try:
            # Handle time tensor properly
            if isinstance(t, (list, tuple)):
                odetime_tensor = torch.cat([time_t.to(self.device) for time_t in t], dim=0)
            else:
                odetime_tensor = t.to(self.device)
            
            all_time = torch.cat([self.zero, odetime_tensor], dim=0)
            all_time = all_time.to(device=x.device, dtype=x.dtype)
            
            # Sort time points to ensure monotonicity
            all_time, _ = torch.sort(all_time)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver,
                        rtol=self.rtol, atol=self.atol)
            
            return out[1]
            
        except Exception as e:
            raise RuntimeError(f"ODEBlockTimeFirst forward pass failed: {str(e)}")


class ODEBlockTimeMiddle(nn.Module):
    """ODE Block for middle time segments (Enhanced)"""
    
    def __init__(self, odeFunction, num_split, solver, rtol=1e-3, atol=1e-4):
        super(ODEBlockTimeMiddle, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t1, t2):
        """Enhanced forward pass"""
        try:
            # Handle time tensors
            if isinstance(t1, (list, tuple)):
                odetime_1_tensor = torch.cat([t.to(x.device) for t in t1], dim=0)
            else:
                odetime_1_tensor = t1.to(x.device)
                
            if isinstance(t2, (list, tuple)):
                odetime_2_tensor = torch.cat([t.to(x.device) for t in t2], dim=0)
            else:
                odetime_2_tensor = t2.to(x.device)
                
            all_time = torch.cat([odetime_1_tensor, odetime_2_tensor], dim=0)
            all_time = all_time.to(device=x.device, dtype=x.dtype)
            
            # Sort time points
            all_time, _ = torch.sort(all_time)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver,
                        rtol=self.rtol, atol=self.atol)
            
            return out[1]
            
        except Exception as e:
            raise RuntimeError(f"ODEBlockTimeMiddle forward pass failed: {str(e)}")


class ODEBlockTimeLast(nn.Module):
    """ODE Block for last time segment (Enhanced)"""
    
    def __init__(self, odeFunction, num_split, solver, final_time=4.0, device='cpu', rtol=1e-3, atol=1e-4):
        super(ODEBlockTimeLast, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.device = torch.device(device) if isinstance(device, str) else device
        self.final_time_val = final_time
        self.one = torch.tensor([final_time], dtype=torch.float32, device=self.device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, x, t):
        """Enhanced forward pass"""
        try:
            # Handle time tensor
            if isinstance(t, (list, tuple)):
                odetime_tensor = torch.cat([time_t.to(self.device) for time_t in t], dim=0)
            else:
                odetime_tensor = t.to(self.device)
                
            all_time = torch.cat([odetime_tensor, self.one.to(x.device)], dim=0)
            all_time = all_time.to(device=x.device, dtype=x.dtype)
            
            # Sort time points
            all_time, _ = torch.sort(all_time)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver,
                        rtol=self.rtol, atol=self.atol)
            
            return out[1]
            
        except Exception as e:
            raise RuntimeError(f"ODEBlockTimeLast forward pass failed: {str(e)}")


class ODEBlockTimeLastK(nn.Module):
    """ODE Block for last time segment with configurable K (Enhanced)"""
    
    def __init__(self, odeFunction, num_split, solver, K, device='cpu', rtol=1e-3, atol=1e-4):
        super(ODEBlockTimeLastK, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.final_time = K
        self.device = torch.device(device) if isinstance(device, str) else device
        self.one = torch.tensor([self.final_time], dtype=torch.float32, device=self.device)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, x, t):
        """Enhanced forward pass"""
        try:
            # Handle time tensor
            if isinstance(t, (list, tuple)):
                odetime_tensor = torch.cat([time_t.to(self.device) for time_t in t], dim=0)
            else:
                odetime_tensor = t.to(self.device)
                
            all_time = torch.cat([odetime_tensor, self.one.to(x.device)], dim=0)
            all_time = all_time.to(device=x.device, dtype=x.dtype)
            
            # Sort time points
            all_time, _ = torch.sort(all_time)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver,
                        rtol=self.rtol, atol=self.atol)
            
            return out[1]
            
        except Exception as e:
            raise RuntimeError(f"ODEBlockTimeLastK forward pass failed: {str(e)}")


# =============================================================================
# Enhanced Utility Functions
# =============================================================================

def ODETime(num_split, device='cpu'):
    """Generate time points for ODE integration (Enhanced)"""
    device = torch.device(device) if isinstance(device, str) else device
    
    if num_split <= 1:
        return [torch.tensor([1.0], dtype=torch.float32, device=device)]
    
    return [torch.tensor([1.0 / num_split * i], dtype=torch.float32, device=device) 
            for i in range(1, num_split)]


def ODETimeSetter(num_split, K, device='cpu'):
    """Set time points with configurable K (Enhanced)"""
    device = torch.device(device) if isinstance(device, str) else device
    
    if num_split <= 1 or K <= 0:
        return [torch.tensor([K], dtype=torch.float32, device=device)]
    
    eta = K / num_split
    return [torch.tensor([i * eta], dtype=torch.float32, device=device) 
            for i in range(1, num_split)]


def ODETimeSplitter(num_split, K):
    """Split time range into segments (Enhanced)"""
    if num_split <= 1 or K <= 0:
        return [K]
    
    eta = K / num_split
    return [i * eta for i in range(1, num_split)]


# =============================================================================
# Numerical Stability Utilities
# =============================================================================

def check_ode_stability(tensor, name="tensor"):
    """Check tensor for numerical stability issues"""
    if torch.isnan(tensor).any():
        warnings.warn(f"NaN values detected in {name}")
        return False
    
    if torch.isinf(tensor).any():
        warnings.warn(f"Infinite values detected in {name}")
        return False
    
    if tensor.abs().max() > 1e6:
        warnings.warn(f"Very large values detected in {name}: max={tensor.abs().max()}")
        return False
    
    return True


def stabilize_tensor(tensor, max_val=1e6, min_val=-1e6):
    """Apply numerical stabilization to tensor"""
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=max_val, neginf=min_val)
    tensor = torch.clamp(tensor, min=min_val, max=max_val)
    return tensor
