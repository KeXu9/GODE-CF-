#!/usr/bin/env python3
"""
Performance Analysis and Benchmarking Tool for GODE-CF
Enhanced version with comprehensive monitoring and optimization recommendations
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import platform
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerformanceProfiler:
    """Advanced performance profiler for GODE-CF project"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timestamps = []
        self.memory_snapshots = []
        self.gpu_snapshots = []
        self.start_time = None
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Collect comprehensive system information"""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'torch_num_threads': torch.get_num_threads()
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else None
            })
        
        return info
    
    def start_profiling(self):
        """Start performance profiling"""
        self.start_time = time.time()
        print("🔍 Performance profiling started")
        print(f"💻 System: {self.system_info['platform']}")
        print(f"🐍 Python: {self.system_info['python_version'].split()[0]}")
        print(f"⚡ PyTorch: {self.system_info['pytorch_version']}")
        if self.system_info['cuda_available']:
            print(f"🚀 GPU: {self.system_info['gpu_name']} ({self.system_info['gpu_memory_gb']:.1f}GB)")
        print("-" * 60)
    
    def record_metric(self, metric_name, value, category='general'):
        """Record a performance metric"""
        timestamp = time.time() - self.start_time if self.start_time else 0
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp,
            'category': category
        })
        self.timestamps.append(timestamp)
        
        # Record system metrics
        self._record_system_metrics()
    
    def _record_system_metrics(self):
        """Record system-level metrics"""
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_snapshots.append({
            'timestamp': time.time() - self.start_time if self.start_time else 0,
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        })
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                self.gpu_snapshots.append({
                    'timestamp': time.time() - self.start_time if self.start_time else 0,
                    'allocated_gb': gpu_memory,
                    'reserved_gb': gpu_reserved
                })
            except:
                pass
    
    def benchmark_operation(self, operation_func, name, *args, **kwargs):
        """Benchmark a specific operation"""
        print(f"⏱️  Benchmarking {name}...")
        
        # Warm up
        try:
            _ = operation_func(*args, **kwargs)
        except:
            pass
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Actual benchmark
        times = []
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        for i in range(3):  # 3 runs for average
            start_time = time.time()
            try:
                result = operation_func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"❌ Benchmark failed for {name}: {e}")
                return None
        
        memory_after = psutil.virtual_memory().used / (1024**3)
        
        benchmark_result = {
            'name': name,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'memory_delta_gb': memory_after - memory_before,
            'runs': len(times)
        }
        
        print(f"✅ {name}: {benchmark_result['avg_time']:.4f}s ± {benchmark_result['std_time']:.4f}s")
        
        self.record_metric(f"{name}_time", benchmark_result['avg_time'], 'benchmark')
        return benchmark_result
    
    def compare_implementations(self, implementations, test_data, name_prefix="impl"):
        """Compare multiple implementations of the same functionality"""
        print(f"🔬 Comparing {len(implementations)} implementations...")
        
        results = []
        for i, (impl_name, impl_func) in enumerate(implementations.items()):
            result = self.benchmark_operation(impl_func, f"{name_prefix}_{impl_name}", test_data)
            if result:
                results.append(result)
        
        # Find best implementation
        if results:
            best = min(results, key=lambda x: x['avg_time'])
            print(f"🏆 Best implementation: {best['name']} ({best['avg_time']:.4f}s)")
            
            # Calculate speedups
            for result in results:
                if result != best:
                    speedup = result['avg_time'] / best['avg_time']
                    print(f"📈 {best['name']} is {speedup:.2f}x faster than {result['name']}")
        
        return results
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        if not self.memory_snapshots:
            print("❌ No memory data collected")
            return
        
        df = pd.DataFrame(self.memory_snapshots)
        
        print("\n📊 Memory Usage Analysis:")
        print(f"💾 Peak memory usage: {df['used_gb'].max():.2f}GB")
        print(f"📉 Memory efficiency: {(df['available_gb'].mean() / self.system_info['memory_total_gb'] * 100):.1f}% available on average")
        
        # Memory growth rate
        if len(df) > 1:
            memory_growth = (df['used_gb'].iloc[-1] - df['used_gb'].iloc[0]) / df['timestamp'].iloc[-1]
            print(f"📈 Memory growth rate: {memory_growth:.3f}GB/s")
            
            if memory_growth > 0.1:  # >100MB/s growth
                print("⚠️  High memory growth detected - possible memory leak!")
    
    def analyze_gpu_usage(self):
        """Analyze GPU usage patterns"""
        if not self.gpu_snapshots:
            print("❌ No GPU data collected")
            return
        
        df = pd.DataFrame(self.gpu_snapshots)
        
        print("\n🚀 GPU Usage Analysis:")
        print(f"📊 Peak GPU memory: {df['allocated_gb'].max():.2f}GB")
        print(f"📊 Peak reserved: {df['reserved_gb'].max():.2f}GB")
        print(f"⚡ GPU efficiency: {(df['allocated_gb'].mean() / df['reserved_gb'].mean() * 100):.1f}% utilization")
    
    def generate_recommendations(self):
        """Generate performance optimization recommendations"""
        print("\n💡 Performance Optimization Recommendations:")
        
        recommendations = []
        
        # Memory recommendations
        if self.memory_snapshots:
            avg_memory = np.mean([s['used_gb'] for s in self.memory_snapshots])
            if avg_memory > self.system_info['memory_total_gb'] * 0.8:
                recommendations.append("🔴 High memory usage detected. Consider reducing batch size or using gradient checkpointing.")
        
        # GPU recommendations
        if self.gpu_snapshots and self.system_info['cuda_available']:
            avg_gpu_mem = np.mean([s['allocated_gb'] for s in self.gpu_snapshots])
            if avg_gpu_mem < self.system_info['gpu_memory_gb'] * 0.3:
                recommendations.append("🟡 Low GPU utilization. Consider increasing batch size or model complexity.")
        
        # PyTorch recommendations
        if self.system_info['torch_num_threads'] != self.system_info['cpu_count']:
            recommendations.append(f"🟡 PyTorch using {self.system_info['torch_num_threads']} threads but {self.system_info['cpu_count']} CPU cores available.")
        
        # Platform-specific recommendations
        if 'arm64' in self.system_info['platform'].lower():
            recommendations.append("🍎 Apple Silicon detected. Consider using MPS backend for better performance.")
        
        if not recommendations:
            recommendations.append("✅ No immediate optimization opportunities detected.")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        return recommendations
    
    def save_report(self, filename=None):
        """Save detailed performance report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = {
            'system_info': self.system_info,
            'metrics': dict(self.metrics),
            'memory_snapshots': self.memory_snapshots,
            'gpu_snapshots': self.gpu_snapshots,
            'recommendations': self.generate_recommendations(),
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Report saved to {filename}")
        return filename
    
    def plot_metrics(self, save_path=None):
        """Generate visualization plots"""
        if not self.metrics:
            print("❌ No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GODE-CF Performance Analysis', fontsize=16, fontweight='bold')
        
        # Memory usage over time
        if self.memory_snapshots:
            df_mem = pd.DataFrame(self.memory_snapshots)
            axes[0, 0].plot(df_mem['timestamp'], df_mem['used_gb'], label='Used Memory', color='red', linewidth=2)
            axes[0, 0].plot(df_mem['timestamp'], df_mem['available_gb'], label='Available Memory', color='green', linewidth=2)
            axes[0, 0].set_title('Memory Usage Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Memory (GB)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # GPU usage over time
        if self.gpu_snapshots:
            df_gpu = pd.DataFrame(self.gpu_snapshots)
            axes[0, 1].plot(df_gpu['timestamp'], df_gpu['allocated_gb'], label='Allocated', color='blue', linewidth=2)
            axes[0, 1].plot(df_gpu['timestamp'], df_gpu['reserved_gb'], label='Reserved', color='orange', linewidth=2)
            axes[0, 1].set_title('GPU Memory Usage Over Time', fontweight='bold')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('GPU Memory (GB)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Benchmark comparison
        benchmark_metrics = {k: v for k, v in self.metrics.items() if any(x['category'] == 'benchmark' for x in v)}
        if benchmark_metrics:
            names = list(benchmark_metrics.keys())
            times = [benchmark_metrics[name][-1]['value'] for name in names]
            
            bars = axes[1, 0].bar(range(len(names)), times, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[1, 0].set_title('Benchmark Results', fontweight='bold')
            axes[1, 0].set_xlabel('Operations')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].set_xticks(range(len(names)))
            axes[1, 0].set_xticklabels([name.replace('_time', '') for name in names], rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{time_val:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # System metrics summary
        axes[1, 1].text(0.1, 0.9, f"💻 Platform: {self.system_info['platform']}", transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.8, f"🔢 CPU Cores: {self.system_info['cpu_count']}", transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.7, f"💾 Memory: {self.system_info['memory_total_gb']:.1f}GB", transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.6, f"⚡ PyTorch: {self.system_info['pytorch_version']}", transform=axes[1, 1].transAxes, fontsize=10)
        
        if self.system_info['cuda_available']:
            axes[1, 1].text(0.1, 0.5, f"🚀 GPU: {self.system_info['gpu_name']}", transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].text(0.1, 0.4, f"📊 VRAM: {self.system_info['gpu_memory_gb']:.1f}GB", transform=axes[1, 1].transAxes, fontsize=10)
        
        axes[1, 1].set_title('System Information', fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {save_path}")
        else:
            plt.show()
        
        return fig


def benchmark_data_loading(profiler, data_file, args_mock):
    """Benchmark data loading performance"""
    print("\n🔄 Benchmarking Data Loading...")
    
    def load_original():
        from dataloader import Loader
        return Loader(args_mock)
    
    # Mock args for testing
    class MockArgs:
        def __init__(self):
            self.data_name = "Beauty"  # Use smallest dataset for testing
            self.device = 'cpu'
            self.a_fold = 100
    
    if Path(data_file).exists():
        args_mock = MockArgs()
        result = profiler.benchmark_operation(load_original, "data_loading")
        return result
    else:
        print(f"⚠️  Data file {data_file} not found, skipping data loading benchmark")
        return None


def benchmark_model_operations(profiler):
    """Benchmark core model operations"""
    print("\n🧠 Benchmarking Model Operations...")
    
    # Create synthetic data for benchmarking
    num_users, num_items = 1000, 500
    embedding_dim = 64
    batch_size = 256
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different embedding operations
    user_emb = torch.randn(num_users, embedding_dim, device=device)
    item_emb = torch.randn(num_items, embedding_dim, device=device)
    user_batch = torch.randint(0, num_users, (batch_size,), device=device)
    
    def matrix_multiply():
        return torch.matmul(user_emb[user_batch], item_emb.t())
    
    def sparse_operations():
        # Simulate sparse matrix operations
        indices = torch.randint(0, min(num_users, num_items), (2, num_users * 2), device=device)
        values = torch.randn(num_users * 2, device=device)
        sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_users, num_items), device=device)
        return torch.sparse.mm(sparse_matrix, item_emb)
    
    # Benchmark operations
    profiler.benchmark_operation(matrix_multiply, "matrix_multiply")
    profiler.benchmark_operation(sparse_operations, "sparse_operations")


def run_comprehensive_analysis():
    """Run comprehensive performance analysis"""
    print("🚀 Starting Comprehensive Performance Analysis for GODE-CF")
    print("=" * 70)
    
    profiler = PerformanceProfiler()
    profiler.start_profiling()
    
    try:
        # Benchmark core operations
        benchmark_model_operations(profiler)
        
        # Check for data files and benchmark if available
        data_files = ["./data/Beauty.txt", "./data/Office_Products.txt"]
        for data_file in data_files:
            if Path(data_file).exists():
                benchmark_data_loading(profiler, data_file, None)
                break
        
        # Analyze results
        profiler.analyze_memory_usage()
        profiler.analyze_gpu_usage()
        profiler.generate_recommendations()
        
        # Save report and plots
        report_file = profiler.save_report()
        plot_file = "performance_analysis.png"
        profiler.plot_metrics(plot_file)
        
        print("\n✅ Performance analysis completed successfully!")
        print(f"📊 Detailed report: {report_file}")
        print(f"📈 Visualization: {plot_file}")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🏁 Analysis finished.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    run_comprehensive_analysis()