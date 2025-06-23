import utils
import torch
import numpy as np
import dataloader
from parse import parse_args
import multiprocessing
import os
from os.path import join
from model import LightGCN, PureMF, UltraGCN, SGCN, LTOCF, LayerGCN, IMP_GCN, EASE, NGCF, ODE_CF, CDE_CF
from trainers import GraphRecTrainer
from utils import EarlyStopping
from time import perf_counter
import pandas as pd
import platform
from tqdm import tqdm

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Optimize for Apple Silicon
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'

args = parse_args()

ROOT_PATH = "./"
DATA_PATH = join(ROOT_PATH, 'data')
CHECKPOINTS_PATH = join(ROOT_PATH, 'checkpoints')

# Ensure checkpoints directory exists
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

# Optimized device selection for M4 Mac
def get_optimal_device():
    """Get the best available device with M4 Mac optimizations and fallback"""
    # For graph neural networks with sparse operations
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected! Using CPU backend for maximum stability and compatibility.")
        return torch.device('cpu')
    elif torch.cuda.is_available():
        print("Using CUDA backend")
        return torch.device('cuda')
    else:
        print("Using CPU backend")
        return torch.device('cpu')

args.device = get_optimal_device()

# Optimize CPU cores for M4 Mac (Performance + Efficiency cores)
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    # M4 has 10 cores (4 performance + 6 efficiency), use performance cores primarily
    args.cores = min(8, multiprocessing.cpu_count())  # Use most cores but leave some for system
    torch.set_num_threads(args.cores)
    print(f"🔧 Optimized for M4 Mac: Using {args.cores} threads")
else:
    args.cores = multiprocessing.cpu_count() // 2

# ENABLE PERFORMANCE OPTIMIZATIONS BY DEFAULT
if not hasattr(args, 'use_pyg'):
    args.use_pyg = True  # Enable PyG by default for 2-3x speedup
if not hasattr(args, 'fast_sampling'):
    args.fast_sampling = True  # Enable ultra-fast sampling
if not hasattr(args, 'enable_cache'):
    args.enable_cache = True  # Enable advanced caching
if not hasattr(args, 'optimize_memory'):
    args.optimize_memory = True  # Enable memory optimizations
if not hasattr(args, 'gradient_clip'):
    args.gradient_clip = 1.0  # Default gradient clipping

print("🚀 Performance optimizations enabled:")
print(f"   - PyTorch Geometric: {args.use_pyg}")
print(f"   - Fast sampling: {getattr(args, 'fast_sampling', True)}")
print(f"   - Advanced caching: {getattr(args, 'enable_cache', True)}")
print(f"   - Memory optimization: {getattr(args, 'optimize_memory', True)}")
print(f"   - Gradient clipping: {getattr(args, 'gradient_clip', 1.0)}")

utils.set_seed(args.seed)

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

# M4 Mac specific optimizations
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    print("🔧 Applying M4 Mac CPU optimizations...")
    # Use optimized BLAS for Apple Silicon
    torch.set_num_threads(args.cores)

    # Advanced memory optimizations for M4 Mac
    torch.backends.mkl.enabled = True
    torch.backends.mkldnn.enabled = True

    # Enable memory efficient attention if available
    if hasattr(torch.backends, 'opt_einsum'):
        torch.backends.opt_einsum.enabled = True

print("📊 Loading dataset...")
dataset = dataloader.Loader(args)
print(f"✅ Dataset loaded: {dataset.n_users} users, {dataset.m_items} items")
print(f"🏗️ Building {args.model_name} model...")

if args.model_name == 'LightGCN':
    model = LightGCN(args, dataset)
elif args.model_name == 'UltraGCN':
    print("Computing constraint matrices...")
    constraint_mat = dataset.getConstraintMat()
    ii_neighbor_mat, ii_constraint_mat = dataset.get_ii_constraint_mat()
    model = UltraGCN(args, dataset, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
elif args.model_name == 'SGCN':
    model = SGCN(args, dataset)
elif args.model_name =='LTOCF':
    model = LTOCF(args, dataset)
elif args.model_name == 'layerGCN':
    model = LayerGCN(args, dataset)
elif args.model_name == 'IMP_GCN':
    model = IMP_GCN(args, dataset)
elif args.model_name =='EASE':
    model = EASE(args, dataset)
elif args.model_name == 'NGCF':
    model = NGCF(args, dataset)
elif args.model_name =='ODE_CF':
    model = ODE_CF(args, dataset)
elif args.model_name =='CDE_CF':
    model = CDE_CF(args, dataset)
else:
    model = PureMF(args, dataset)

print(f"📱 Moving model to {args.device}...")
model = model.to(args.device)

# ADVANCED PERFORMANCE OPTIMIZATIONS
print("⚡ Applying advanced performance optimizations...")

# Model compilation (PyTorch 2.0+)
if getattr(args, 'compile_model', False) or platform.system() == 'Darwin':
    try:
        if hasattr(torch, 'compile'):
            print("🔧 Compiling model with torch.compile...")
            if args.model_name == 'ODE_CF':
                # Conservative compilation for ODE models
                model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            else:
                # Aggressive compilation for standard models
                model = torch.compile(model, mode='max-autotune' if torch.cuda.is_available() else 'default')
            print("✅ Model compiled successfully")
        else:
            print("⚠️ torch.compile not available (requires PyTorch 2.0+)")
    except Exception as e:
        print(f"⚠️ Model compilation failed: {e}")

# Mixed precision training setup
scaler = None
if getattr(args, 'use_mixed_precision', False) and torch.cuda.is_available():
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("✅ Mixed precision training enabled (2x speedup expected)")
    except ImportError:
        print("⚠️ Mixed precision not available")

# Memory optimization flags
if getattr(args, 'optimize_memory', True):
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for faster training
        torch.backends.cuda.matmul.allow_tf32 = True

trainer = GraphRecTrainer(model, dataset, args)

# Pass mixed precision scaler to trainer if available
if 'scaler' in locals() and scaler is not None:
    trainer.scaler = scaler
    print("🔧 Mixed precision scaler attached to trainer")

checkpoint_path = utils.getFileName("./checkpoints/", args)
print(f"load and save to {checkpoint_path}")

t = perf_counter()
if args.do_eval:
    #trainer.load(checkpoint_path)
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    print(f'Load model from {checkpoint_path} for test!')
    scores, result_info, _ = trainer.complicated_eval()
else:
    val_result = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    early_stopping = EarlyStopping(checkpoint_path, patience=50, verbose=True)
    if args.model_name == 'layerGCN':
            trainer.model.pre_epoch_processing()

    print(f"🚀 Starting training for {args.epochs} epochs...")

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(args.epochs), desc="🔥 Training", unit="epoch",
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    best_ndcg = 0.0
    best_epoch = 0

    for epoch in epoch_pbar:
        epoch_start = perf_counter()
        trainer.train(epoch)
        epoch_time = perf_counter() - epoch_start

        # Memory management
        if epoch % 10 == 0:  # Periodic cleanup
            import gc
            gc.collect()

        # evaluate on MRR
        if (epoch+1) % 10 == 0:
            eval_start = perf_counter()
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            eval_time = perf_counter() - eval_start

            val_result.append(scores)
            early_stopping(np.array(scores[-1:]), trainer.model)

            # Extract current metrics
            current_ndcg = scores[9] if len(scores) > 9 else 0.0  # NDCG@20
            current_hit = scores[8] if len(scores) > 8 else 0.0   # HIT@20
            current_mrr = scores[12] if len(scores) > 12 else 0.0  # MRR

            # Track best performance
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1

            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'NDCG@20': f'{current_ndcg:.4f}',
                'HIT@20': f'{current_hit:.4f}',
                'MRR': f'{current_mrr:.4f}',
                'Best': f'{best_ndcg:.4f}@{best_epoch}',
                'Train': f'{epoch_time:.1f}s',
                'Eval': f'{eval_time:.1f}s'
            })

            if early_stopping.early_stop:
                epoch_pbar.set_description("🛑 Early Stop")
                epoch_pbar.close()
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"🏆 Best NDCG@20: {best_ndcg:.4f} at epoch {best_epoch}")
                break
        else:
            # Update progress bar for non-evaluation epochs
            epoch_pbar.set_postfix({
                'Train': f'{epoch_time:.1f}s',
                'Status': 'Training...'
            })

    if not early_stopping.early_stop:
        epoch_pbar.set_description("✅ Complete")
        epoch_pbar.close()
        print(f"\n✅ Training completed!")
        print(f"🏆 Best NDCG@20: {best_ndcg:.4f} at epoch {best_epoch}")

    print('---------------Change to Final testing!-------------------')
    # load the best model if checkpoint exists
    if os.path.exists(checkpoint_path):
        trainer.model.load_state_dict(torch.load(checkpoint_path))
        print(f"✅ Loaded best model from {checkpoint_path}")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}, using current model")

    valid_scores, val_r, _ = trainer.valid('best', full_sort=True)
    scores, result_info, _ = trainer.test('best', full_sort=True)
    val_result.append(scores)

col_name = ["HIT@1", "NDCG@1", "HIT@5", "NDCG@5", "HIT@10", "NDCG@10", "HIT@15", "NDCG@15", "HIT@20", "NDCG@20", "HIT@40", "NDCG@40", "MRR"]
val_result = pd.DataFrame(val_result, columns=col_name)
path = './results/' +args.model_name + '_' + args.data_name+ '_val_result_t=' + str(args.t) +'_solver_' + args.solver  + '.csv'
val_result.to_csv(path, index=False)
train_time = perf_counter()-t
with open('./results/overall_time.txt', 'a') as f:
    f.writelines(path + ': '+ "Train time: {:.4f}s".format(train_time) + '\n')
    f.writelines('best_val: ' + val_r+'\n')
    f.writelines('test: ' + result_info+'\n')
    f.writelines('\n')
print("Train time: {:.4f}s".format(train_time))