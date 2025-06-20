"""
Unified ODE Module for Graph Neural Networks
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


# =============================================================================
# ODE Functions (from ode.py and ode1.py)
# =============================================================================

class ODEFunc(nn.Module):
    """ODE Function with dataset-specific architecture (from ode.py)"""
    
    def __init__(self, adj, latent_dim, data_name):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.x0 = None
        self.name = data_name

        if self.name == "Cell_Phones_and_Accessories":
            self.linear2_u = nn.Linear(latent_dim, 1)
        else:
            self.linear2_u = nn.Linear(latent_dim, int(latent_dim/2))
            self.linear2_u_1 = nn.Linear(int(latent_dim/2), 1)

    def forward(self, t, x):
        if self.name == "Cell_Phones_and_Accessories":
            alph = nn.functional.sigmoid(self.linear2_u(x))
        else:
            alph = nn.functional.sigmoid(self.linear2_u_1(self.linear2_u(x)))
            
        ax = torch.spmm(self.g, x)
        ax = alph * torch.spmm(self.g, ax)
        f = ax - x
        return f


class ODEFunc1(nn.Module):
    """ODE Function with trainable alpha (from ode1.py)"""

    def __init__(self, adj, latent_dim, device='cpu'):
        super(ODEFunc1, self).__init__()
        self.g = adj
        # Fix: Use scalar alpha or per-dimension alpha, not per-node alpha
        self.alpha_train = nn.Parameter(0.9 * torch.ones(1).to(device))  # Scalar alpha
        # Alternative: per-dimension alpha
        # self.alpha_train = nn.Parameter(0.9 * torch.ones(latent_dim).to(device))

    def forward(self, t, x):
        # Apply alpha as a scalar multiplier
        alph = torch.sigmoid(self.alpha_train)  # Shape: [1]
        ax = torch.spmm(self.g, x)  # First graph convolution
        ax = alph * torch.spmm(self.g, ax)  # Second graph convolution with alpha weighting
        # Return the derivative (change rate), not residual
        # The ODE solver will handle integration: x(t) = x(0) + ∫f(t,x)dt
        f = ax - x  # This represents dx/dt
        return f


class ODEFunction(nn.Module):
    """Simple ODE Function for LGC (from odeblock.py)"""
    
    def __init__(self, Graph):
        super(ODEFunction, self).__init__()
        self.g = Graph

    def forward(self, t, x):
        """ODEFUNCTION(| --> only single layer --> |)"""
        out = torch.sparse.mm(self.g, x)
        return out


# =============================================================================
# ODE Blocks (Basic blocks from ode.py and ode1.py)
# =============================================================================

class ODEblock(nn.Module):
    """Basic ODE Block (from ode.py and ode1.py)"""
    
    def __init__(self, odefunc, t=torch.tensor([0, 1]), solver='euler'):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.solver = solver

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method=self.solver)[1]
        return z


class ODEBlock(nn.Module):
    """ODE Block with time range (from odeblock.py)"""
    
    def __init__(self, odeFunction, solver, init_time, final_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odeFunction
        self.integration_time = torch.tensor([init_time, final_time]).float()
        self.solver = solver

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(func=self.odefunc, y0=x, t=self.integration_time, method=self.solver)
        return out[1]


# =============================================================================
# Time-based ODE Blocks (from odeblock.py)
# =============================================================================

class ODEBlockTimeFirst(nn.Module):
    """ODE Block for first time segment"""
    
    def __init__(self, odeFunction, num_split, solver, device='cpu'):
        super(ODEBlockTimeFirst, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.zero = torch.tensor([0.], requires_grad=False).to(device)
        self.solver = solver
                    
    def forward(self, x, t):
        odetime = t
        odetime_tensor = torch.cat(odetime, dim=0)
        all_time = torch.cat([self.zero, odetime_tensor], dim=0).to(x.device)
        
        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        out = odeint(func=self.odefunc, y0=x, t=total_integration_time, method=self.solver)
        return out[1]


class ODEBlockTimeMiddle(nn.Module):
    """ODE Block for middle time segments"""
    
    def __init__(self, odeFunction, num_split, solver):
        super(ODEBlockTimeMiddle, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.solver = solver

    def forward(self, x, t1, t2):
        odetime_1 = t1
        odetime_2 = t2
        odetime_1_tensor = torch.cat(odetime_1, dim=0)
        odetime_2_tensor = torch.cat(odetime_2, dim=0)
        all_time = torch.cat([odetime_1_tensor, odetime_2_tensor], dim=0).to(x.device)

        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        out = odeint(func=self.odefunc, y0=x, t=total_integration_time, method=self.solver)
        return out[1]


class ODEBlockTimeLast(nn.Module):
    """ODE Block for last time segment"""
    
    def __init__(self, odeFunction, num_split, solver, final_time=4.0, device='cpu'):
        super(ODEBlockTimeLast, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.one = torch.tensor([final_time], requires_grad=False).to(device)
        self.solver = solver
        
    def forward(self, x, t):
        odetime = t
        odetime_tensor = torch.cat(odetime, dim=0)
        all_time = torch.cat([odetime_tensor, self.one], dim=0).to(x.device)

        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        out = odeint(func=self.odefunc, y0=x, t=total_integration_time, method=self.solver)
        return out[1]


class ODEBlockTimeLastK(nn.Module):
    """ODE Block for last time segment with configurable K"""
    
    def __init__(self, odeFunction, num_split, solver, K, device='cpu'):
        super(ODEBlockTimeLastK, self).__init__()
        self.odefunc = odeFunction
        self.num_split = num_split
        self.final_time = K
        self.one = torch.tensor([self.final_time], requires_grad=False).to(device)
        self.solver = solver
        
    def forward(self, x, t):
        odetime = t
        odetime_tensor = torch.cat(odetime, dim=0)
        all_time = torch.cat([odetime_tensor, self.one], dim=0).to(x.device)

        all_time1 = all_time.type_as(x)
        total_integration_time = all_time1
        out = odeint(func=self.odefunc, y0=x, t=total_integration_time, method=self.solver)
        return out[1]


# =============================================================================
# Utility Functions (from odeblock.py)
# =============================================================================

def ODETime(num_split, device='cpu'):
    """Generate time points for ODE integration"""
    return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device=device) 
            for i in range(1, num_split)]


def ODETimeSetter(num_split, K, device='cpu'):
    """Set time points with configurable K"""
    eta = K / num_split
    return [torch.tensor([i * eta], dtype=torch.float32, requires_grad=True, device=device) 
            for i in range(1, num_split)]


def ODETimeSplitter(num_split, K):
    """Split time range into segments"""
    eta = K / num_split
    return [i * eta for i in range(1, num_split)]
