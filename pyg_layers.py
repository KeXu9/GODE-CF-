"""
PyTorch Geometric optimized layers for faster graph operations
Replaces sparse tensor operations with efficient message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor
from typing import Optional
import numpy as np


class LightGCNConv(MessagePassing):
    """
    Optimized LightGCN layer using PyTorch Geometric MessagePassing
    """
    
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)
    
    def forward(self, x: torch.Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> torch.Tensor:
        """
        Forward pass with optimized message passing
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        
        Returns:
            Updated node features [num_nodes, num_features]
        """
        # Compute normalization if not provided
        if edge_weight is None:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Message function: normalize neighbor features"""
        return edge_weight.view(-1, 1) * x_j
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'





def create_bipartite_edge_index(user_item_matrix, num_users: int, num_items: int, 
                               device: torch.device = None) -> torch.Tensor:
    """
    Create edge_index for bipartite user-item graph
    
    Args:
        user_item_matrix: Scipy sparse matrix [num_users, num_items]
        num_users: Number of users
        num_items: Number of items
        device: Target device
    
    Returns:
        edge_index: [2, num_edges] tensor with bidirectional edges
    """
    ui_coo = user_item_matrix.tocoo()
    
    # User to item edges (users: 0 to num_users-1, items: num_users to num_users+num_items-1)
    ui_edges = np.stack([ui_coo.row, ui_coo.col + num_users])
    
    # Item to user edges (transpose)
    iu_edges = np.stack([ui_coo.col + num_users, ui_coo.row])
    
    # Combine both directions
    edge_index = np.concatenate([ui_edges, iu_edges], axis=1)
    edge_index = torch.from_numpy(edge_index).long()
    
    if device is not None:
        edge_index = edge_index.to(device)
    
    return edge_index


def create_edge_weights(user_item_matrix, device: torch.device = None) -> torch.Tensor:
    """
    Create edge weights from user-item matrix
    
    Args:
        user_item_matrix: Scipy sparse matrix
        device: Target device
    
    Returns:
        edge_weight: [num_edges] tensor with bidirectional weights
    """
    ui_coo = user_item_matrix.tocoo()
    
    # Bidirectional weights (same for both directions)
    weights = np.concatenate([ui_coo.data, ui_coo.data])
    edge_weight = torch.from_numpy(weights).float()
    
    if device is not None:
        edge_weight = edge_weight.to(device)
    
    return edge_weight


def scipy_to_pyg(adj_matrix, device: torch.device = None) -> tuple:
    """
    Convert scipy sparse matrix to PyG format
    
    Args:
        adj_matrix: Scipy sparse matrix
        device: Target device
    
    Returns:
        edge_index: [2, num_edges] tensor
        edge_weight: [num_edges] tensor
    """
    coo = adj_matrix.tocoo()
    
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float)
    
    if device is not None:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    
    return edge_index, edge_weight


def benchmark_pyg_vs_sparse(num_nodes: int, num_edges: int, embedding_dim: int, 
                          num_layers: int, device: torch.device, num_iterations: int = 10):
    """
    Benchmark PyG vs sparse tensor performance
    
    Returns:
        dict: Performance comparison results
    """
    import time
    
    # Create random graph
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_weight = torch.rand(num_edges, device=device)
    
    # Create sparse tensor equivalent
    sparse_adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=device).coalesce()
    
    # Random embeddings
    x = torch.randn(num_nodes, embedding_dim, device=device)
    
    # PyG approach
    conv = LightGCNConv().to(device)
    
    # Benchmark PyG
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        x_pyg = x.clone()
        for _ in range(num_layers):
            x_pyg = conv(x_pyg, edge_index, edge_weight)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    pyg_time = (time.time() - start_time) / num_iterations
    
    # Benchmark sparse tensor
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        x_sparse = x.clone()
        for _ in range(num_layers):
            x_sparse = torch.sparse.mm(sparse_adj, x_sparse)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    sparse_time = (time.time() - start_time) / num_iterations
    
    return {
        'pyg_time': pyg_time,
        'sparse_time': sparse_time,
        'speedup': sparse_time / pyg_time,
        'pyg_memory': torch.cuda.memory_allocated() if device.type == 'cuda' else 0
    }
