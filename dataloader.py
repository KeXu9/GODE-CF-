import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from time import time
import copy
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import mmap
from pyg_layers import create_bipartite_edge_index

# Setup basic logging
logger = logging.getLogger(__name__)


def pload(path: str) -> Any:
    """Enhanced pickle loading with basic validation"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            res = pickle.load(f)
        print(f'load path = {path} object')
        return res
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {str(e)}")


def pstore(x: Any, path: str) -> None:
    """Enhanced pickle storing with directory creation"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(x, f)
        print(f'store object in path = {path} ok')
    except Exception as e:
        raise RuntimeError(f"Failed to store to {path}: {str(e)}")


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self, ui_mat):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):

    def __init__(self, args):
        super().__init__()
        # Basic configuration
        datapath = './data/'
        data_file = datapath + '/' + args.data_name + '.txt'
        self.args = args
        self.split = False
        self.folds = self.args.a_fold
        self.n_user = 0
        self.m_item = 0
        self.path = './data/'
        
        # Initialize data structures with estimated capacity for better performance
        print(f"Loading data from {data_file}")
        start_time = time()

        # Load and validate data file
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # First pass: count lines and estimate capacity
        with open(data_file, 'r') as f:
            num_lines = sum(1 for _ in f if _.strip())

        # Pre-allocate lists with estimated capacity
        train_data = defaultdict(list)
        valid_data = defaultdict(list)
        test_data = defaultdict(list)

        self.trainSize = 0
        self.validSize = 0
        self.testSize = 0

        # Process data file with optimized I/O
        try:
            with open(data_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue

                        # Optimized parsing
                        parts = line.split(' ')
                        if len(parts) < 4:  # user + at least 3 items
                            warnings.warn(f"User at line {line_num} has insufficient items ({len(parts)-1})")
                            continue

                        uid = int(parts[0]) - 1
                        itemids = [int(item) - 1 for item in parts[1:]]

                        # Validate indices
                        if uid < 0 or any(item_id < 0 for item_id in itemids):
                            warnings.warn(f"Invalid indices at line {line_num}")
                            continue

                        # Update max indices
                        self.n_user = max(self.n_user, uid)
                        self.m_item = max(self.m_item, max(itemids))

                        # Split data efficiently: train (all except last 2), valid (second to last), test (last)
                        train_items = itemids[:-2]
                        valid_item = itemids[-2]
                        test_item = itemids[-1]

                        # Store in dictionaries for efficient access
                        if train_items:
                            train_data[uid].extend(train_items)
                            self.trainSize += len(train_items)

                        valid_data[uid].append(valid_item)
                        self.validSize += 1

                        test_data[uid].append(test_item)
                        self.testSize += 1

                    except (ValueError, IndexError) as e:
                        warnings.warn(f"Error processing line {line_num}: {e}")
                        continue

        except Exception as e:
            raise RuntimeError(f"Failed to process data file: {e}")

        # Convert to numpy arrays efficiently
        trainUser, trainItem = [], []
        validUser, validItem = [], []
        testUser, testItem = [], []
        trainUniqueUsers, validUniqueUsers, testUniqueUsers = [], [], []

        for uid in train_data:
            items = train_data[uid]
            trainUniqueUsers.append(uid)
            trainUser.extend([uid] * len(items))
            trainItem.extend(items)

        for uid in valid_data:
            items = valid_data[uid]
            validUniqueUsers.append(uid)
            validUser.extend([uid] * len(items))
            validItem.extend(items)

        for uid in test_data:
            items = test_data[uid]
            testUniqueUsers.append(uid)
            testUser.extend([uid] * len(items))
            testItem.extend(items)

        # Convert to numpy arrays and adjust indices
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.m_item += 1
        self.n_user += 1
        
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)
        
        # Initialize graph
        self.Graph = None
        self.edge_index = None  # PyG format
        self.edge_weight = None
        
        # Log basic statistics
        load_time = time() - start_time
        print(f"Data loaded in {load_time:.2f}s")
        print(f"#of users: {self.n_users} and #of items: {self.m_items}")
        print(f"{self.trainSize} interactions for training")
        print(f"{self.validSize} interactions for validation")
        print(f"{self.testSize} interactions for testing")
        
        sparsity = (self.trainSize + self.validSize + self.testSize) / self.n_users / self.m_items
        print(f"{self.args.data_name} Sparsity : {sparsity}")

        # Build user-item interaction matrix with validation
        try:
            self.UserItemNet = csr_matrix(
                (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                shape=(self.n_user, self.m_item)
            )
            
            # Validate matrix
            if self.UserItemNet.nnz == 0:
                raise ValueError("User-item matrix is empty")
                
        except Exception as e:
            raise RuntimeError(f"Failed to create user-item matrix: {e}")
        
        # Compute degree statistics with numerical stability
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # Initialize additional networks
        self.UserUserNet = None
        self.ItemItemNet = None
        
        # Pre-calculate derived structures
        try:
            self._allPos = self.getUserPosItems(list(range(self.n_user)))
            self.__testDict, self.test_pred_mask_mat = self.__build_test()
            self.__validDict = self.__build_valid()
            self.valid_pred_mask_mat = self.UserItemNet

            # Performance optimization: Pre-cache frequently used tensors
            self._cached_graph = None
            self._cached_edge_index = None
            self._graph_cache_valid = False
        except Exception as e:
            raise RuntimeError(f"Failed to build derived structures: {e}")

    def random_sample_edges(self, adj, n, exclude):
        """Sample edges with basic validation"""
        if n <= 0:
            raise ValueError("Sample size must be positive")
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square")
            
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Optimized edge sampling with pre-computed valid edges"""
        n_nodes = adj.shape[0]

        # Pre-compute all possible edges for small graphs
        if n_nodes < 1000:
            all_edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
            valid_edges = [edge for edge in all_edges if edge not in exclude and (edge[1], edge[0]) not in exclude]

            if not valid_edges:
                raise RuntimeError("No valid edges available for sampling")

            # Shuffle for randomness
            np.random.shuffle(valid_edges)

            for edge in valid_edges:
                yield edge
                exclude.add(edge)
                exclude.add((edge[1], edge[0]))
        else:
            # For large graphs, use rejection sampling with batching
            max_attempts = min(n_nodes * 100, 100000)  # Reduced max attempts
            attempts = 0
            batch_size = min(1000, n_nodes // 10)  # Sample in batches

            while attempts < max_attempts:
                # Generate batch of candidates
                candidates = np.random.choice(n_nodes, (batch_size, 2), replace=True)

                for candidate in candidates:
                    if candidate[0] != candidate[1]:  # No self-loops
                        t = tuple(sorted(candidate))  # Ensure consistent ordering
                        if t not in exclude:
                            yield t
                            exclude.add(t)
                            exclude.add((t[1], t[0]))

                attempts += batch_size

            raise RuntimeError("Maximum sampling attempts reached")

    def reset_graph(self, newdata):
        """Reset graph with validation"""
        try:
            new_row, new_col, new_val = newdata
            
            # Basic validation
            if len(new_row) != len(new_col) or len(new_row) != len(new_val):
                raise ValueError("Input arrays must have same length")
            
            self.UserItemNet = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_user, self.m_item))
            print('reset graph nnz: ', self.UserItemNet.nnz, ', density: ', self.UserItemNet.nnz/(self.n_user*self.m_item))
            
            # Recompute degree statistics
            self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
            self.users_D[self.users_D == 0.] = 1.
            self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
            self.items_D[self.items_D == 0.] = 1.
            
        except Exception as e:
            raise RuntimeError(f"Graph reset failed: {e}")

    def reset_graph_ease(self, newdata):
        """Reset graph with EASE computation and improved error handling"""
        try:
            oldUserItemNet = self.UserItemNet
            self.reset_graph(newdata)

            print("Computing EASE matrices...")

            # Optimized user-user matrix computation
            print("Computing user-user similarity matrix...")
            # Use sparse operations to avoid memory issues
            uuG_sparse = oldUserItemNet.dot(oldUserItemNet.T)

            # Only convert to dense if matrix is small enough
            if uuG_sparse.shape[0] < 10000:  # Threshold for memory safety
                uuG = uuG_sparse.toarray()
                uu_diagIndices = np.diag_indices(uuG.shape[0])
                uuG[uu_diagIndices] += self.args.uu_lambda

                # Check for numerical stability
                if np.linalg.cond(uuG) > 1e12:
                    warnings.warn("User-User matrix is poorly conditioned")
                    uuG[uu_diagIndices] += 1e-6

                uuP = np.linalg.inv(uuG)
                self.UserUserNet = uuP / (-np.diag(uuP))
                self.UserUserNet[uu_diagIndices] = 0
                self.UserUserNet[self.UserUserNet < 0] = 0
                self.UserUserNet = csr_matrix(self.UserUserNet)
            else:
                # For large matrices, use sparse operations throughout
                warnings.warn("Large user matrix detected, using sparse approximation")
                # Add regularization to diagonal using efficient method
                diag_values = uuG_sparse.diagonal() + self.args.uu_lambda
                uuG_sparse = uuG_sparse.tolil()
                uuG_sparse.setdiag(diag_values)
                uuG_sparse = uuG_sparse.tocsr()

                # Use sparse solver for large matrices
                identity = sp.eye(uuG_sparse.shape[0])
                try:
                    uuP_sparse = spsolve(uuG_sparse, identity)
                    self.UserUserNet = csr_matrix(uuP_sparse)
                    # Remove diagonal efficiently
                    self.UserUserNet = self.UserUserNet.tolil()
                    self.UserUserNet.setdiag(0)
                    self.UserUserNet = self.UserUserNet.tocsr()
                    # Remove negative values
                    self.UserUserNet.data[self.UserUserNet.data < 0] = 0
                except:
                    warnings.warn("Sparse solve failed, using identity matrix")
                    self.UserUserNet = sp.eye(uuG_sparse.shape[0], format='csr')

            print('uu net nnz: ', self.UserUserNet.nnz, ', density: ', self.UserUserNet.nnz/(self.n_user*self.n_user))

            # Optimized item-item matrix computation
            print("Computing item-item similarity matrix...")
            iiG_sparse = oldUserItemNet.T.dot(oldUserItemNet)

            if iiG_sparse.shape[0] < 10000:  # Threshold for memory safety
                iiG = iiG_sparse.toarray()
                ii_diagIndices = np.diag_indices(iiG.shape[0])
                iiG[ii_diagIndices] += self.args.ii_lambda

                # Check for numerical stability
                if np.linalg.cond(iiG) > 1e12:
                    warnings.warn("Item-Item matrix is poorly conditioned")
                    iiG[ii_diagIndices] += 1e-6

                iiP = np.linalg.inv(iiG)
                self.ItemItemNet = iiP / (-np.diag(iiP))
                self.ItemItemNet[ii_diagIndices] = 0
                self.ItemItemNet[self.ItemItemNet < 0] = 0
                self.ItemItemNet = csr_matrix(self.ItemItemNet)
            else:
                # For large matrices, use sparse operations
                warnings.warn("Large item matrix detected, using sparse approximation")
                # Add regularization to diagonal using efficient method
                diag_values = iiG_sparse.diagonal() + self.args.ii_lambda
                iiG_sparse = iiG_sparse.tolil()
                iiG_sparse.setdiag(diag_values)
                iiG_sparse = iiG_sparse.tocsr()

                identity = sp.eye(iiG_sparse.shape[0])
                try:
                    iiP_sparse = spsolve(iiG_sparse, identity)
                    self.ItemItemNet = csr_matrix(iiP_sparse)
                    # Remove diagonal efficiently
                    self.ItemItemNet = self.ItemItemNet.tolil()
                    self.ItemItemNet.setdiag(0)
                    self.ItemItemNet = self.ItemItemNet.tocsr()
                    # Remove negative values
                    self.ItemItemNet.data[self.ItemItemNet.data < 0] = 0
                except:
                    warnings.warn("Sparse solve failed, using identity matrix")
                    self.ItemItemNet = sp.eye(iiG_sparse.shape[0], format='csr')

            print('ii net nnz: ', self.ItemItemNet.nnz, ', density: ', self.ItemItemNet.nnz/(self.m_item*self.m_item))

        except Exception as e:
            raise RuntimeError(f"EASE graph reset failed: {e}")

    def reset_graph_uuii(self, newdata):
        """Reset graph with UU and II matrices"""
        try:
            if len(newdata) != 3:
                raise ValueError("Expected 3 data components")
                
            [newuidata, newuudata, newiidata] = newdata

            # Reset UI matrix
            self.reset_graph(newuidata)

            # Set UU matrix
            new_uu_row, new_uu_col, new_uu_val = newuudata
            self.UserUserNet = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)), shape=(self.n_user, self.n_user))
            # Remove diagonal efficiently
            self.UserUserNet = self.UserUserNet.tolil()
            self.UserUserNet.setdiag(0)
            self.UserUserNet = self.UserUserNet.tocsr()

            # Set II matrix
            new_ii_row, new_ii_col, new_ii_val = newiidata
            self.ItemItemNet = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)), shape=(self.m_item, self.m_item))
            # Remove diagonal efficiently
            self.ItemItemNet = self.ItemItemNet.tolil()
            self.ItemItemNet.setdiag(0)
            self.ItemItemNet = self.ItemItemNet.tocsr()
            
        except Exception as e:
            raise RuntimeError(f"UU-II graph reset failed: {e}")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.trainSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        """Split adjacency matrix"""
        try:
            A_fold = []
            fold_len = (self.n_users + self.m_items) // self.folds

            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

            for i_fold in range(self.folds):
                start = i_fold * fold_len
                if i_fold == self.folds - 1:
                    end = self.n_users + self.m_items
                else:
                    end = (i_fold + 1) * fold_len

                sparse_fold = self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce()
                A_fold.append(sparse_fold.to(device))

            return A_fold
        except Exception as e:
            raise RuntimeError(f"Matrix splitting failed: {e}")

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert sparse matrix to tensor"""
        try:
            if X.nnz == 0:
                # Handle empty matrix
                indices = torch.zeros((2, 0), dtype=torch.long)
                values = torch.zeros(0, dtype=torch.float)
                return torch.sparse.FloatTensor(indices, values, torch.Size(X.shape))

            coo = X.tocoo().astype(np.float32)
            row = torch.tensor(coo.row, dtype=torch.long)
            col = torch.tensor(coo.col, dtype=torch.long)
            index = torch.stack([row, col])
            data = torch.tensor(coo.data, dtype=torch.float32)

            sparse_tensor = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
            return sparse_tensor
        except Exception as e:
            raise RuntimeError(f"Sparse conversion failed: {e}")

    def getSparseGraph(self, ui_spmat, include_uuii=False):
        """Generate sparse graph with optimized memory usage and caching"""
        try:
            # Check memory cache first
            cache_key = f"graph_{ui_spmat.nnz}_{include_uuii}_{self.n_users}_{self.m_items}"
            if (hasattr(self, '_cached_graph') and self._cached_graph is not None and
                self._graph_cache_valid and hasattr(self, '_cache_key') and self._cache_key == cache_key):
                return self._cached_graph

            print("generating adjacency matrix")
            s = time()

            # Validate input matrix
            if ui_spmat.shape != (self.n_users, self.m_items):
                raise ValueError(f"Matrix shape {ui_spmat.shape} doesn't match expected {(self.n_users, self.m_items)}")

            # Check for file cached graph
            cache_path = os.path.join(self.path, f"{self.args.data_name}_{cache_key}.pkl")

            if os.path.exists(cache_path):
                print("Loading cached graph...")
                try:
                    cached_graph = pload(cache_path)
                    device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
                    graph = cached_graph.to(device)

                    # Update memory cache
                    self._cached_graph = graph
                    self._cache_key = cache_key
                    self._graph_cache_valid = True

                    return graph
                except:
                    print("Cache loading failed, recomputing...")

            # Optimized adjacency matrix construction
            total_nodes = self.n_users + self.m_items

            # Use COO format for efficient construction
            row_indices = []
            col_indices = []
            data_values = []

            # Add user-item edges
            ui_coo = ui_spmat.tocoo()
            # User to item edges
            row_indices.extend(ui_coo.row)
            col_indices.extend(ui_coo.col + self.n_users)
            data_values.extend(ui_coo.data)

            # Item to user edges (transpose)
            row_indices.extend(ui_coo.col + self.n_users)
            col_indices.extend(ui_coo.row)
            data_values.extend(ui_coo.data)

            print(f'UI edges: {len(ui_coo.data) * 2}')

            # Add UU and II edges if requested
            if include_uuii and (self.UserUserNet is not None and self.ItemItemNet is not None):
                print('Including UU and II edges')

                # Add UU edges
                uu_coo = self.UserUserNet.tocoo()
                row_indices.extend(uu_coo.row)
                col_indices.extend(uu_coo.col)
                data_values.extend(uu_coo.data)

                # Add II edges
                ii_coo = self.ItemItemNet.tocoo()
                row_indices.extend(ii_coo.row + self.n_users)
                col_indices.extend(ii_coo.col + self.n_users)
                data_values.extend(ii_coo.data)

                print(f'Added UU edges: {len(uu_coo.data)}, II edges: {len(ii_coo.data)}')

            # Create sparse matrix efficiently
            adj_mat = sp.coo_matrix(
                (data_values, (row_indices, col_indices)),
                shape=(total_nodes, total_nodes),
                dtype=np.float32
            ).tocsr()

            print(f'Total adj mat nnz: {adj_mat.nnz}')

            # Optimized normalization
            print("Computing normalization...")
            rowsum = np.asarray(adj_mat.sum(axis=1)).flatten()
            d_inv = np.power(rowsum + 1e-10, -0.5)  # Add small epsilon for stability
            d_inv[np.isinf(d_inv)] = 0.

            # Use sparse diagonal matrix for efficiency
            d_mat = sp.diags(d_inv, format='csr')

            # Compute normalized adjacency matrix
            norm_adj = d_mat @ adj_mat @ d_mat

            end = time()
            print(f"Matrix construction took {end-s:.2f}s")

            # Convert to tensor
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

            graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            graph = graph.coalesce().to(device)

            # Cache the result in both file and memory
            try:
                cached_graph = graph.cpu()
                pstore(cached_graph, cache_path)
                print("Graph cached successfully")

                # Update memory cache
                self._cached_graph = graph
                self._cache_key = cache_key
                self._graph_cache_valid = True

            except Exception as cache_error:
                warnings.warn(f"Failed to cache graph: {cache_error}")

            return graph

        except Exception as e:
            raise RuntimeError(f"Sparse graph generation failed: {e}")

    def __build_test(self):
        """Build test dictionary with validation"""
        try:
            test_pred_mask_mat = copy.deepcopy(self.UserItemNet)

            # Convert to LIL format for efficient modification
            test_pred_mask_mat = test_pred_mask_mat.tolil()

            # Add validation items to mask
            for i, item in enumerate(self.validItem):
                user = self.validUser[i]
                if 0 <= user < self.n_user and 0 <= item < self.m_item:
                    test_pred_mask_mat[user, item] = 1.0

            # Convert back to CSR format
            test_pred_mask_mat = test_pred_mask_mat.tocsr()

            test_data = {}
            for i, item in enumerate(self.testItem):
                user = self.testUser[i]
                if 0 <= user < self.n_user and 0 <= item < self.m_item:
                    if test_data.get(user):
                        test_data[user].append(item)
                    else:
                        test_data[user] = [item]

            return test_data, test_pred_mask_mat
        except Exception as e:
            raise RuntimeError(f"Test dictionary building failed: {e}")

    def __build_valid(self):
        """Build validation dictionary with validation"""
        try:
            valid_data = {}
            for i, item in enumerate(self.validItem):
                user = self.validUser[i]
                if 0 <= user < self.n_user and 0 <= item < self.m_item:
                    if valid_data.get(user):
                        valid_data[user].append(item)
                    else:
                        valid_data[user] = [item]
            return valid_data
        except Exception as e:
            raise RuntimeError(f"Validation dictionary building failed: {e}")

    def getUserItemFeedback(self, users, items):
        """Get user-item feedback with bounds checking"""
        try:
            # Basic validation
            if len(users) != len(items):
                raise ValueError("Users and items arrays must have same length")
            
            if len(users) > 0:
                if np.max(users) >= self.n_user or np.min(users) < 0:
                    raise ValueError("User indices out of bounds")
                if np.max(items) >= self.m_item or np.min(items) < 0:
                    raise ValueError("Item indices out of bounds")
            
            return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
        except Exception as e:
            raise RuntimeError(f"Getting user-item feedback failed: {e}")

    def getUserPosItems(self, users):
        """Get positive items for users with validation"""
        try:
            if not users:
                return []
                
            max_user = max(users)
            if max_user >= self.n_user:
                raise ValueError(f"User index {max_user} out of bounds")
                
            posItems = []
            for user in users:
                posItems.append(self.UserItemNet[user].nonzero()[1])
            return posItems
        except Exception as e:
            raise RuntimeError(f"Getting positive items failed: {e}")

    def getConstraintMat(self):
        """Get constraint matrix with numerical stability"""
        try:
            items_D = np.sum(self.UserItemNet, axis=0).reshape(-1)
            users_D = np.sum(self.UserItemNet, axis=1).reshape(-1)

            beta_uD = (np.sqrt(users_D + 1) / (users_D + 1)).reshape(-1, 1)
            beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

            constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                              "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

            return constraint_mat
        except Exception as e:
            raise RuntimeError(f"Constraint matrix computation failed: {e}")

    def get_ii_constraint_mat(self, ii_diagonal_zero=False):
        """Get item-item constraint matrix with optimized computation and caching"""
        try:
            ii_cons_mat_path = self.path + self.args.data_name + '_ii_constraint_mat'
            ii_neigh_mat_path = self.path + self.args.data_name + '_ii_neighbor_mat'

            if os.path.exists(ii_cons_mat_path):
                ii_constraint_mat = pload(ii_cons_mat_path)
                ii_neighbor_mat = pload(ii_neigh_mat_path)
            else:
                print('Computing \\Omega for the item-item graph... ')

                # Optimized matrix computation
                A = self.UserItemNet.T.dot(self.UserItemNet)
                n_items = A.shape[0]

                if ii_diagonal_zero:
                    # Use more efficient approach to zero diagonal
                    A = A.tolil()  # Convert to LIL for efficient modification
                    A.setdiag(0)
                    A = A.tocsr()  # Convert back to CSR

                # Vectorized degree computation
                items_D = np.asarray(A.sum(axis=0)).flatten()
                users_D = np.asarray(A.sum(axis=1)).flatten()

                # Optimized constraint matrix computation
                beta_uD = np.sqrt(users_D + 1) / (users_D + 1)
                beta_iD = 1 / np.sqrt(items_D + 1)

                # Pre-allocate result matrices
                res_mat = torch.zeros((n_items, self.args.ii_neighbor_num), dtype=torch.long)
                res_sim_mat = torch.zeros((n_items, self.args.ii_neighbor_num), dtype=torch.float32)

                # Process in batches for memory efficiency
                batch_size = min(1000, n_items // 4)  # Adaptive batch size

                for start_idx in range(0, n_items, batch_size):
                    end_idx = min(start_idx + batch_size, n_items)
                    batch_indices = range(start_idx, end_idx)

                    # Batch processing
                    for i in batch_indices:
                        # Get row efficiently
                        row_data = A.getrow(i).toarray().flatten()

                        # Apply constraint weights
                        constraint_weights = beta_uD[i] * beta_iD
                        weighted_row = row_data * constraint_weights

                        # Convert to tensor and find top-k
                        row_tensor = torch.from_numpy(weighted_row)
                        if row_tensor.numel() >= self.args.ii_neighbor_num:
                            row_sims, row_idxs = torch.topk(row_tensor, self.args.ii_neighbor_num)
                        else:
                            # Handle case where we have fewer items than neighbors requested
                            row_sims, row_idxs = torch.sort(row_tensor, descending=True)
                            # Pad with zeros if necessary
                            if len(row_sims) < self.args.ii_neighbor_num:
                                pad_size = self.args.ii_neighbor_num - len(row_sims)
                                row_sims = torch.cat([row_sims, torch.zeros(pad_size)])
                                row_idxs = torch.cat([row_idxs, torch.zeros(pad_size, dtype=torch.long)])

                        res_mat[i] = row_idxs[:self.args.ii_neighbor_num]
                        res_sim_mat[i] = row_sims[:self.args.ii_neighbor_num]

                    # Progress reporting
                    if start_idx % (batch_size * 15) == 0:
                        print(f'i-i constraint matrix batch {start_idx}-{end_idx} ok')

                print('Computation \\Omega OK!')
                ii_neighbor_mat = res_mat
                ii_constraint_mat = res_sim_mat
                pstore(ii_neighbor_mat, ii_neigh_mat_path)
                pstore(ii_constraint_mat, ii_cons_mat_path)

            return ii_neighbor_mat.long(), ii_constraint_mat.float()
        except Exception as e:
            raise RuntimeError(f"II constraint matrix computation failed: {e}")

    def getPyGGraph(self, ui_spmat):
        """
        Get PyTorch Geometric format graph for faster operations

        Args:
            ui_spmat: User-item sparse matrix

        Returns:
            edge_index: [2, num_edges] tensor
            edge_weight: [num_edges] tensor (optional)
        """
        try:
            print("Creating PyG format graph...")

            # Create cache path
            cache_path = f"./data/{self.args.data_name}_pyg_graph.pkl"

            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    cached_data = pload(cache_path)
                    edge_index = cached_data['edge_index'].to(self.args.device)
                    edge_weight = cached_data.get('edge_weight', None)
                    if edge_weight is not None:
                        edge_weight = edge_weight.to(self.args.device)

                    print("PyG graph loaded from cache")
                    return edge_index, edge_weight
                except Exception as cache_error:
                    warnings.warn(f"Failed to load cached PyG graph: {cache_error}")

            # Create bipartite edge index
            edge_index = create_bipartite_edge_index(
                ui_spmat, self.n_users, self.m_items, device=self.args.device
            )

            # Optional: create edge weights (all ones for unweighted graph)
            edge_weight = None  # For LightGCN, we typically don't use edge weights

            # Cache the result
            try:
                cache_data = {
                    'edge_index': edge_index.cpu(),
                    'edge_weight': edge_weight.cpu() if edge_weight is not None else None
                }
                pstore(cache_data, cache_path)
                print("PyG graph cached successfully")
            except Exception as cache_error:
                warnings.warn(f"Failed to cache PyG graph: {cache_error}")

            print(f"PyG graph created: {edge_index.size(1)} edges")
            return edge_index, edge_weight

        except Exception as e:
            raise RuntimeError(f"PyG graph creation failed: {e}")

    def getEdgeIndex(self):
        """Get cached edge_index for PyG models"""
        if self.edge_index is None:
            self.edge_index, self.edge_weight = self.getPyGGraph(self.UserItemNet)
        return self.edge_index, self.edge_weight