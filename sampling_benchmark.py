#!/usr/bin/env python3
"""
Comprehensive Sampling Benchmark for GODE-CF
Tests and compares all sampling implementations for performance validation
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import psutil
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import our sampling functions
sys.path.append('.')
from utils import (
    UniformSample_original, 
    UniformSample_baseline
)

class MockDataset:
    """Mock dataset for benchmarking sampling without needing real data"""
    
    def __init__(self, n_users, m_items, density=0.1):
        self.n_users = n_users
        self.m_items = m_items
        self.density = density
        
        # Generate synthetic user-item interactions
        self.allPos = self._generate_synthetic_interactions()
        self.trainDataSize = sum(len(items) for items in self.allPos)
        
        print(f"🎯 Mock dataset created:")
        print(f"   👥 Users: {self.n_users:,}")
        print(f"   🛍️  Items: {self.m_items:,}")
        print(f"   📊 Density: {self.density:.1%}")
        print(f"   🔢 Total interactions: {self.trainDataSize:,}")
        
    def _generate_synthetic_interactions(self):
        """Generate realistic user-item interaction patterns"""
        allPos = []
        
        # Create different user types for realistic testing
        for user_id in range(self.n_users):
            # Vary interaction counts to simulate real-world distributions
            if user_id < self.n_users * 0.1:
                # 10% heavy users (many interactions)
                n_interactions = np.random.randint(
                    int(self.m_items * self.density * 2), 
                    int(self.m_items * self.density * 5)
                )
            elif user_id < self.n_users * 0.3:
                # 20% medium users
                n_interactions = np.random.randint(
                    int(self.m_items * self.density * 0.5), 
                    int(self.m_items * self.density * 2)
                )
            else:
                # 70% sparse users (few interactions)
                n_interactions = np.random.randint(
                    1, 
                    int(self.m_items * self.density * 0.5) + 1
                )
            
            # Generate random item interactions
            items = np.random.choice(
                self.m_items, 
                size=min(n_interactions, self.m_items), 
                replace=False
            )
            allPos.append(items)
        
        return allPos


class SamplingBenchmark:
    """Comprehensive sampling benchmark suite"""
    
    def __init__(self):
        self.results = []
        self.memory_profiles = []
        
    def benchmark_implementation(self, impl_func, impl_name, dataset, num_runs=3):
        """Benchmark a specific sampling implementation"""
        print(f"\n🔬 Benchmarking {impl_name}...")
        
        times = []
        memory_usage = []
        
        # Warm-up run
        try:
            _ = impl_func(dataset)
            gc.collect()
        except Exception as e:
            print(f"❌ Warm-up failed for {impl_name}: {e}")
            return None
        
        # Actual benchmark runs
        for run in range(num_runs):
            # Memory before
            mem_before = psutil.virtual_memory().used / (1024**3)
            
            # Time the implementation
            start_time = time.time()
            try:
                result = impl_func(dataset)
                end_time = time.time()
                
                # Memory after
                mem_after = psutil.virtual_memory().used / (1024**3)
                
                # Validate result
                if not self._validate_sampling_result(result, dataset):
                    print(f"⚠️  {impl_name} produced invalid results")
                    continue
                
                times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
                
            except Exception as e:
                print(f"❌ {impl_name} failed on run {run + 1}: {e}")
                continue
            
            # Cleanup
            gc.collect()
        
        if not times:
            return None
        
        # Calculate statistics
        benchmark_result = {
            'implementation': impl_name,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory_mb': np.mean(memory_usage) * 1024,
            'samples_per_second': len(result) / np.mean(times) if len(result) > 0 else 0,
            'runs': len(times),
            'dataset_size': f"{dataset.n_users}x{dataset.m_items}"
        }
        
        print(f"✅ {impl_name}: {benchmark_result['avg_time']:.4f}s ± {benchmark_result['std_time']:.4f}s")
        print(f"   📊 {benchmark_result['samples_per_second']:,.0f} samples/sec")
        print(f"   💾 {benchmark_result['avg_memory_mb']:.1f}MB memory")
        
        return benchmark_result
    
    def _validate_sampling_result(self, result, dataset):
        """Validate that sampling result is correct"""
        if result is None or len(result) == 0:
            return False
        
        if result.shape[1] != 3:  # Should have [user, pos_item, neg_item]
            return False
        
        # Check that users are valid
        if np.any(result[:, 0] >= dataset.n_users) or np.any(result[:, 0] < 0):
            return False
        
        # Check that items are valid
        if np.any(result[:, 1] >= dataset.m_items) or np.any(result[:, 1] < 0):
            return False
        if np.any(result[:, 2] >= dataset.m_items) or np.any(result[:, 2] < 0):
            return False
        
        # Check some positive/negative constraints (sample check)
        for i in range(min(100, len(result))):
            user, pos_item, neg_item = result[i]
            user_pos_items = set(dataset.allPos[user])
            
            # Positive item should be in user's positive items
            if pos_item not in user_pos_items:
                return False
            
            # Negative item should NOT be in user's positive items
            if neg_item in user_pos_items:
                return False
        
        return True
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across multiple scenarios"""
        print("🚀 Starting Comprehensive Sampling Benchmark")
        print("=" * 60)
        
        # Define test scenarios
        scenarios = [
            {"name": "Small Dense", "n_users": 1000, "m_items": 500, "density": 0.3},
            {"name": "Medium Sparse", "n_users": 5000, "m_items": 2000, "density": 0.05},
            {"name": "Large Realistic", "n_users": 10000, "m_items": 5000, "density": 0.1},
        ]
        
        # Implementations to test
        implementations = {
            "Baseline": UniformSample_baseline,
            "Ultimate": UniformSample_original,
        }
        
        all_results = []
        
        for scenario in scenarios:
            print(f"\n🎯 Testing Scenario: {scenario['name']}")
            print("-" * 40)
            
            # Create dataset for this scenario
            dataset = MockDataset(
                scenario['n_users'], 
                scenario['m_items'], 
                scenario['density']
            )
            
            scenario_results = []
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                result = self.benchmark_implementation(
                    impl_func, impl_name, dataset, num_runs=3
                )
                if result:
                    result['scenario'] = scenario['name']
                    scenario_results.append(result)
                    all_results.append(result)
            
            # Calculate speedups for this scenario
            if len(scenario_results) > 1:
                baseline_time = None
                for r in scenario_results:
                    if 'Baseline' in r['implementation']:
                        baseline_time = r['avg_time']
                        break
                
                if baseline_time:
                    print(f"\n📈 Speedup Analysis for {scenario['name']}:")
                    for r in scenario_results:
                        if r['implementation'] != 'Baseline':
                            speedup = baseline_time / r['avg_time']
                            print(f"   🚀 {r['implementation']}: {speedup:.1f}x faster than baseline")
        
        # Store results
        self.results = all_results
        
        # Generate comprehensive analysis
        self._analyze_results()
        self._plot_results()
        
        return all_results
    
    def _analyze_results(self):
        """Analyze benchmark results and generate insights"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print("\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 50)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Overall performance summary
        print("\n🏆 Overall Performance Summary:")
        for impl in df['implementation'].unique():
            impl_data = df[df['implementation'] == impl]
            avg_time = impl_data['avg_time'].mean()
            avg_throughput = impl_data['samples_per_second'].mean()
            avg_memory = impl_data['avg_memory_mb'].mean()
            
            print(f"  {impl}:")
            print(f"    ⏱️  Avg Time: {avg_time:.4f}s")
            print(f"    🚀 Avg Throughput: {avg_throughput:,.0f} samples/sec")
            print(f"    💾 Avg Memory: {avg_memory:.1f}MB")
        
        # Best implementation analysis
        print("\n🥇 Best Implementation by Metric:")
        
        fastest = df.loc[df['avg_time'].idxmin()]
        print(f"  ⚡ Fastest: {fastest['implementation']} ({fastest['avg_time']:.4f}s)")
        
        highest_throughput = df.loc[df['samples_per_second'].idxmax()]
        print(f"  🚀 Highest Throughput: {highest_throughput['implementation']} ({highest_throughput['samples_per_second']:,.0f} samples/sec)")
        
        lowest_memory = df.loc[df['avg_memory_mb'].idxmin()]
        print(f"  💾 Lowest Memory: {lowest_memory['implementation']} ({lowest_memory['avg_memory_mb']:.1f}MB)")
        
        # Scenario-specific analysis
        print("\n📋 Scenario-Specific Performance:")
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            best_impl = scenario_data.loc[scenario_data['avg_time'].idxmin()]
            print(f"  {scenario}: {best_impl['implementation']} is fastest ({best_impl['avg_time']:.4f}s)")
        
        # Calculate overall speedups
        baseline_results = df[df['implementation'] == 'Baseline']
        if not baseline_results.empty:
            print("\n🚀 Overall Speedup Analysis:")
            baseline_avg = baseline_results['avg_time'].mean()
            
            for impl in df['implementation'].unique():
                if impl != 'Baseline':
                    impl_avg = df[df['implementation'] == impl]['avg_time'].mean()
                    speedup = baseline_avg / impl_avg
                    print(f"  {impl}: {speedup:.1f}x faster than baseline")
    
    def _plot_results(self):
        """Generate visualization plots"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sampling Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Execution time comparison
        sns.barplot(data=df, x='scenario', y='avg_time', hue='implementation', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time by Scenario', fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput comparison
        sns.barplot(data=df, x='scenario', y='samples_per_second', hue='implementation', ax=axes[0, 1])
        axes[0, 1].set_title('Throughput (Samples/Second)', fontweight='bold')
        axes[0, 1].set_ylabel('Samples per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Memory usage comparison
        sns.barplot(data=df, x='scenario', y='avg_memory_mb', hue='implementation', ax=axes[1, 0])
        axes[1, 0].set_title('Memory Usage by Implementation', fontweight='bold')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Speedup analysis
        baseline_data = df[df['implementation'] == 'Baseline']
        if not baseline_data.empty:
            speedup_data = []
            for scenario in df['scenario'].unique():
                scenario_baseline = baseline_data[baseline_data['scenario'] == scenario]
                if not scenario_baseline.empty:
                    baseline_time = scenario_baseline['avg_time'].iloc[0]
                    scenario_data = df[df['scenario'] == scenario]
                    
                    for _, row in scenario_data.iterrows():
                        if row['implementation'] != 'Baseline':
                            speedup = baseline_time / row['avg_time']
                            speedup_data.append({
                                'scenario': scenario,
                                'implementation': row['implementation'],
                                'speedup': speedup
                            })
            
            if speedup_data:
                speedup_df = pd.DataFrame(speedup_data)
                sns.barplot(data=speedup_df, x='scenario', y='speedup', hue='implementation', ax=axes[1, 1])
                axes[1, 1].set_title('Speedup vs Baseline', fontweight='bold')
                axes[1, 1].set_ylabel('Speedup Factor')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.tight_layout()
        plt.savefig('sampling_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("📊 Benchmark plots saved to 'sampling_benchmark_results.png'")
        
        return fig
    
    def generate_report(self, filename=None):
        """Generate detailed benchmark report"""
        if filename is None:
            filename = 'sampling_benchmark_report.txt'
        
        with open(filename, 'w') as f:
            f.write("GODE-CF Sampling Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total implementations tested: {len(set(r['implementation'] for r in self.results))}\n")
            f.write(f"Total scenarios tested: {len(set(r['scenario'] for r in self.results))}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            df = pd.DataFrame(self.results)
            for scenario in df['scenario'].unique():
                f.write(f"\nScenario: {scenario}\n")
                scenario_data = df[df['scenario'] == scenario]
                
                for _, row in scenario_data.iterrows():
                    f.write(f"  {row['implementation']}:\n")
                    f.write(f"    Time: {row['avg_time']:.4f}s ± {row['std_time']:.4f}s\n")
                    f.write(f"    Throughput: {row['samples_per_second']:,.0f} samples/sec\n")
                    f.write(f"    Memory: {row['avg_memory_mb']:.1f}MB\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            
            fastest_overall = df.loc[df['avg_time'].idxmin()]
            f.write(f"🏆 Use '{fastest_overall['implementation']}' for best overall performance\n")
            
            if 'Ultimate' in df['implementation'].values:
                ultimate_data = df[df['implementation'] == 'Ultimate']
                baseline_data = df[df['implementation'] == 'Baseline']
                
                if not baseline_data.empty:
                    avg_speedup = (baseline_data['avg_time'].mean() / 
                                 ultimate_data['avg_time'].mean())
                    f.write(f"🚀 Ultimate implementation provides {avg_speedup:.1f}x average speedup\n")
            
            f.write("💡 For production use, prioritize the Ultimate implementation\n")
            f.write("🔧 Monitor memory usage for large-scale datasets\n")
        
        print(f"📝 Detailed report saved to '{filename}'")


def main():
    """Run the comprehensive sampling benchmark"""
    print("🎯 GODE-CF Sampling Performance Benchmark")
    print("🚀 Testing state-of-the-art sampling optimizations")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = SamplingBenchmark()
    
    try:
        # Run comprehensive tests
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate report
        benchmark.generate_report()
        
        print("\n✅ Benchmark completed successfully!")
        print("📊 Check 'sampling_benchmark_results.png' for visualizations")
        print("📝 Check 'sampling_benchmark_report.txt' for detailed results")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()