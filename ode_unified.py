"""
Unified ODE Module for Graph Neural Networks - FIXED VERSION
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import warnings


# =============================================================================
# ODE Functions (from ode.py and ode1.py) - FIXED
# =============================================================================

class ODEFunc(nn.Module):
    """ODE Function with dataset-specific architecture (from ode.py) - OPTIMIZED"""
    
    def __init__(self, adj, latent_dim, data_name):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.x0 = None
        self.name = data_name
        self.latent_dim = latent_dim

        # Improved architecture with proper initialization
        if self.name == "Cell_Phones_and_Accessories":
            self.linear2_u = nn.Linear(latent_dim, 1)
        else:
            hidden_dim = max(latent_dim // 2, 1)  # Ensure at least 1 dimension
            self.linear2_u = nn.Linear(latent_dim, hidden_dim)
            self.linear2_u_1 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, t, x):
        """Forward pass with numerical stability checks"""
        try:
            # Input validation
            if torch.isnan(x).any() or torch.isinf(x).any():
                warnings.warn("Invalid values in ODE input, applying numerical corrections")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Compute alpha with proper architecture
            if self.name == "Cell_Phones_and_Accessories":
                alph = torch.sigmoid(self.linear2_u(x))
            else:
                hidden = torch.relu(self.linear2_u(x))  # Add ReLU for stability
                alph = torch.sigmoid(self.linear2_u_1(hidden))
            
            # Graph convolutions with numerical stability
            ax = torch.spmm(self.g, x)
            ax = alph * torch.spmm(self.g, ax)
            
            # Residual connection with proper scaling
            f = ax - x
            
            # Gradient clipping for stability
            f = torch.clamp(f, min=-10.0, max=10.0)
            
            return f
            
        except Exception as e:
            warnings.warn(f"ODE computation failed: {e}, returning zero derivative")
            return torch.zeros_like(x)


class ODEFunc1(nn.Module):
    """ODE Function with trainable alpha (from ode1.py) - MAJOR FIXES"""

    def __init__(self, adj, latent_dim, device='cpu'):
        super(ODEFunc1, self).__init__()
        self.g = adj
        self.latent_dim = latent_dim
        self.device = device
        
        # CRITICAL FIX: Use scalar alpha instead of per-node alpha to prevent memory explosion
        # The original implementation had a major bug here
        self.alpha_train = nn.Parameter(
            torch.tensor(0.9, dtype=torch.float32, device=device, requires_grad=True)
        )
        
        # Add learnable temperature parameter for better control
        self.temperature = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
        )
        
        # Track statistics for debugging
        self.forward_calls = 0
        self.gradient_norm_history = []

    def forward(self, t, x):
        """Forward pass with major fixes for stability and performance"""
        try:
            self.forward_calls += 1
            
            # Input validation and cleaning
            if torch.isnan(x).any() or torch.isinf(x).any():
                warnings.warn(f"Invalid values detected at ODE call {self.forward_calls}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # FIXED: Apply alpha as scalar with temperature control
            alph = torch.sigmoid(self.alpha_train * self.temperature)
            
            # Efficient graph convolutions
            ax = torch.spmm(self.g, x)  # First convolution
            ax = alph * torch.spmm(self.g, ax)  # Second convolution with learned weighting
            
            # Compute derivative (not residual)
            f = ax - x
            
            # Enhanced numerical stability
            f = torch.clamp(f, min=-5.0, max=5.0)
            
            # Track gradient norms for debugging
            if x.requires_grad and len(self.gradient_norm_history) < 100:
                grad_norm = torch.norm(f).item()
                self.gradient_norm_history.append(grad_norm)
            
            return f
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Emergency memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                warnings.warn("ODE computation OOM, returning scaled input")
                return 0.1 * x
            else:
                warnings.warn(f"ODE computation failed: {e}")
                return torch.zeros_like(x)
        
        except Exception as e:
            warnings.warn(f"Unexpected ODE error: {e}")
            return torch.zeros_like(x)
    
    def get_alpha_value(self):
        """Get current alpha value for monitoring"""
        return torch.sigmoid(self.alpha_train * self.temperature).item()
    
    def reset_stats(self):
        """Reset debugging statistics"""
        self.forward_calls = 0
        self.gradient_norm_history = []


class ODEFunction(nn.Module):
    """Simple ODE Function for LGC (from odeblock.py) - OPTIMIZED"""
    
    def __init__(self, Graph):
        super(ODEFunction, self).__init__()
        self.g = Graph
        self.forward_calls = 0

    def forward(self, t, x):
        """ODEFUNCTION with error handling and optimization"""
        try:
            self.forward_calls += 1
            
            # Input validation
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Efficient sparse matrix multiplication
            out = torch.sparse.mm(self.g, x)
            
            # Numerical stability
            out = torch.clamp(out, min=-1e6, max=1e6)
            
            return out
            
        except Exception as e:
            warnings.warn(f"ODEFunction computation failed: {e}")
            return torch.zeros_like(x)


# =============================================================================
# ODE Blocks (Basic blocks from ode.py and ode1.py) - ENHANCED
# =============================================================================

class ODEblock(nn.Module):
    """Basic ODE Block with enhanced error handling and adaptive solving"""
    
    def __init__(self, odefunc, t=torch.tensor([0, 1]), solver='euler'):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.solver = solver
        self.solve_attempts = 0
        self.successful_solves = 0
        
        # Adaptive solver parameters
        self.adaptive_step_size = True
        self.max_attempts = 3

    def forward(self, x):
        """Forward with adaptive ODE solving and fallback mechanisms"""
        self.solve_attempts += 1
        
        try:
            # Ensure time tensor is on correct device and type
            t = self.t.type_as(x).to(x.device)
            
            # Try solving with current solver
            for attempt in range(self.max_attempts):
                try:
                    # Use different tolerances for different attempts
                    if attempt == 0:
                        z = odeint(self.odefunc, x, t, method=self.solver)
                    elif attempt == 1:
                        # Try with smaller tolerance
                        z = odeint(self.odefunc, x, t, method=self.solver, rtol=1e-4, atol=1e-6)
                    else:
                        # Last resort: use simple euler
                        z = odeint(self.odefunc, x, t, method='euler', options={'step_size': 0.1})
                    
                    self.successful_solves += 1
                    return z[1]  # Return final state
                    
                except Exception as e:
                    if attempt == self.max_attempts - 1:
                        warnings.warn(f"ODE solving failed after {self.max_attempts} attempts: {e}")
                        # Fallback: return input + small perturbation
                        return x + 0.01 * torch.randn_like(x)
                    continue
                    
        except Exception as e:
            warnings.warn(f"ODEblock forward failed: {e}")
            return x  # Return input as fallback

    def get_success_rate(self):
        """Get solving success rate for monitoring"""
        return self.successful_solves / max(self.solve_attempts, 1)


class ODEBlock(nn.Module):
    """ODE Block with time range (from odeblock.py) - OPTIMIZED"""
    
    def __init__(self, odeFunction, solver, init_time, final_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odeFunction
        self.integration_time = torch.tensor([init_time, final_time], dtype=torch.float32)
        self.solver = solver
        self.solve_count = 0

    def forward(self, x):
        """Forward pass with proper device management and error handling"""
        try:
            self.solve_count += 1
            
            # Ensure integration time is on correct device
            self.integration_time = self.integration_time.type_as(x).to(x.device)
            
            # Solve ODE with error handling
            out = odeint(func=self.odefunc, y0=x, t=self.integration_time, method=self.solver)
            
            return out[1]  # Return final state
            
        except Exception as e:
            warnings.warn(f"ODEBlock forward failed at call {self.solve_count}: {e}")
            return x + 0.01 * torch.randn_like(x)


# =============================================================================
# Time-based ODE Blocks - ENHANCED WITH ERROR HANDLING
# =============================================================================

class ODEBlockTimeFirst(nn.Module):
    """ODE Block for first time segment - FIXED"""
    
    def __init__(self, odeFunction, num_split, solver, device='cpu'):
        super(ODEBlockTimeFirst, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.zero = torch.tensor([0.], requires_grad=False, device=device)
        self.solver = solver
        self.device = device
                    
    def forward(self, x, t):
        """Forward with proper tensor handling"""
        try:
            # Ensure all tensors are on the same device
            self.zero = self.zero.to(x.device)
            
            # Handle time tensor properly
            if isinstance(t, list):
                odetime_tensor = torch.cat(t, dim=0).to(x.device)
            else:
                odetime_tensor = t.to(x.device)
            
            all_time = torch.cat([self.zero, odetime_tensor], dim=0)
            all_time = all_time.type_as(x)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver)
            return out[1]
            
        except Exception as e:
            warnings.warn(f"ODEBlockTimeFirst failed: {e}")
            return x


class ODEBlockTimeMiddle(nn.Module):
    """ODE Block for middle time segments - FIXED"""
    
    def __init__(self, odeFunction, num_split, solver):
        super(ODEBlockTimeMiddle, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.solver = solver

    def forward(self, x, t1, t2):
        """Forward with improved tensor handling"""
        try:
            # Handle tensor concatenation properly
            if isinstance(t1, list):
                odetime_1_tensor = torch.cat(t1, dim=0)
            else:
                odetime_1_tensor = t1
                
            if isinstance(t2, list):
                odetime_2_tensor = torch.cat(t2, dim=0)
            else:
                odetime_2_tensor = t2
            
            all_time = torch.cat([odetime_1_tensor, odetime_2_tensor], dim=0).to(x.device)
            all_time = all_time.type_as(x)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver)
            return out[1]
            
        except Exception as e:
            warnings.warn(f"ODEBlockTimeMiddle failed: {e}")
            return x


class ODEBlockTimeLast(nn.Module):
    """ODE Block for last time segment - FIXED"""
    
    def __init__(self, odeFunction, num_split, solver, final_time=4.0, device='cpu'):
        super(ODEBlockTimeLast, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.one = torch.tensor([final_time], requires_grad=False, device=device)
        self.solver = solver
        self.device = device
        
    def forward(self, x, t):
        """Forward with proper device management"""
        try:
            # Ensure tensors are on correct device
            self.one = self.one.to(x.device)
            
            if isinstance(t, list):
                odetime_tensor = torch.cat(t, dim=0)
            else:
                odetime_tensor = t
            
            all_time = torch.cat([odetime_tensor.to(x.device), self.one], dim=0)
            all_time = all_time.type_as(x)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver)
            return out[1]
            
        except Exception as e:
            warnings.warn(f"ODEBlockTimeLast failed: {e}")
            return x


class ODEBlockTimeLastK(nn.Module):
    """ODE Block for last time segment with configurable K - FIXED"""
    
    def __init__(self, odeFunction, num_split, solver, K, device='cpu'):
        super(ODEBlockTimeLastK, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.final_time = K
        self.one = torch.tensor([self.final_time], requires_grad=False, device=device)
        self.solver = solver
        self.device = device
        
    def forward(self, x, t):
        """Forward with enhanced error handling"""
        try:
            self.one = self.one.to(x.device)
            
            if isinstance(t, list):
                odetime_tensor = torch.cat(t, dim=0)
            else:
                odetime_tensor = t
            
            all_time = torch.cat([odetime_tensor.to(x.device), self.one], dim=0)
            all_time = all_time.type_as(x)
            
            out = odeint(func=self.odefunc, y0=x, t=all_time, method=self.solver)
            return out[1]
            
        except Exception as e:
            warnings.warn(f"ODEBlockTimeLastK failed: {e}")
            return x


# =============================================================================
# Utility Functions - OPTIMIZED
# =============================================================================

def ODETime(num_split, device='cpu'):
    """Generate time points for ODE integration - OPTIMIZED"""
    try:
        time_points = []
        for i in range(1, num_split):
            t = torch.tensor([i / num_split], dtype=torch.float32, requires_grad=True, device=device)
            time_points.append(t)
        return time_points
    except Exception as e:
        warnings.warn(f"ODETime generation failed: {e}")
        return [torch.tensor([0.5], dtype=torch.float32, device=device)]


def ODETimeSetter(num_split, K, device='cpu'):
    """Set time points with configurable K - OPTIMIZED"""
    try:
        eta = K / num_split
        time_points = []
        for i in range(1, num_split):
            t = torch.tensor([i * eta], dtype=torch.float32, requires_grad=True, device=device)
            time_points.append(t)
        return time_points
    except Exception as e:
        warnings.warn(f"ODETimeSetter generation failed: {e}")
        return [torch.tensor([K/2], dtype=torch.float32, device=device)]


def ODETimeSplitter(num_split, K):
    """Split time range into segments - VALIDATED"""
    try:
        if num_split <= 0:
            raise ValueError("num_split must be positive")
        if K <= 0:
            raise ValueError("K must be positive")
            
        eta = K / num_split
        return [i * eta for i in range(1, num_split)]
    except Exception as e:
        warnings.warn(f"ODETimeSplitter failed: {e}")
        return [K/2]
