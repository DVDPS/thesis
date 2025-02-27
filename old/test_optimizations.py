import time
import argparse
import torch
import numpy as np
import logging
from agent import EnhancedPPOAgent
from game2048 import Game2048, preprocess_state_onehot
from performance_optimization import (
    fast_preprocess_state_onehot,
    optimize_gpu_utilization,
    FastPrioritizedReplayBuffer,
    compute_monotonicity_fast
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def benchmark_preprocessing(num_iterations=10000):
    """Compare original preprocessing with optimized version"""
    env = Game2048()
    state = env.reset()
    
    # Pre-allocate buffer for optimized version
    output_buffer = np.zeros((16, 4, 4), dtype=np.float32)
    
    # Benchmark original preprocessing
    start_time = time.time()
    for _ in range(num_iterations):
        original_result = preprocess_state_onehot(state)
    original_time = time.time() - start_time
    
    # Benchmark optimized preprocessing
    start_time = time.time()
    for _ in range(num_iterations):
        optimized_result = fast_preprocess_state_onehot(state, output_buffer)
    optimized_time = time.time() - start_time
    
    # Verify results match
    if np.array_equal(original_result, optimized_result):
        match_status = "Results match ✓"
    else:
        match_status = "Results differ ✗"
    
    # Print results
    logging.info(f"Preprocessing Benchmark ({num_iterations} iterations):")
    logging.info(f"  Original: {original_time:.4f} seconds")
    logging.info(f"  Optimized: {optimized_time:.4f} seconds")
    logging.info(f"  Speedup: {original_time / optimized_time:.2f}x")
    logging.info(f"  {match_status}")

def benchmark_replay_buffer(buffer_size=100000, num_operations=10000, batch_size=128):
    """Compare standard replay buffer with optimized version"""
    from collections import deque
    from training import PrioritizedReplayBuffer
    
    # Create dummy state
    state = np.random.random((16, 4, 4)).astype(np.float32)
    next_state = np.random.random((16, 4, 4)).astype(np.float32)
    
    # Initialize both buffer types
    standard_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
    optimized_buffer = FastPrioritizedReplayBuffer(capacity=buffer_size)
    
    # Benchmark adding to buffers
    logging.info(f"Buffer Add Benchmark ({num_operations} operations):")
    
    start_time = time.time()
    for i in range(num_operations):
        action = i % 4
        reward = float(i % 10)
        done = False
        standard_buffer.add(state, action, reward, next_state, done, 64)
    standard_add_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(num_operations):
        action = i % 4
        reward = float(i % 10)
        done = False
        optimized_buffer.add(state, action, reward, next_state, done)
    optimized_add_time = time.time() - start_time
    
    logging.info(f"  Standard Buffer Add: {standard_add_time:.4f} seconds")
    logging.info(f"  Optimized Buffer Add: {optimized_add_time:.4f} seconds")
    logging.info(f"  Speedup: {standard_add_time / optimized_add_time:.2f}x")
    
    # Benchmark sampling from buffers
    logging.info(f"Buffer Sample Benchmark ({num_operations} operations):")
    
    start_time = time.time()
    for _ in range(num_operations // 100):  # Sampling is slower, so do fewer iterations
        batch, weights, indices = standard_buffer.sample(batch_size)
    standard_sample_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(num_operations // 100):
        batch = optimized_buffer.sample(batch_size)
    optimized_sample_time = time.time() - start_time
    
    logging.info(f"  Standard Buffer Sample: {standard_sample_time:.4f} seconds")
    logging.info(f"  Optimized Buffer Sample: {optimized_sample_time:.4f} seconds")
    logging.info(f"  Speedup: {standard_sample_time / optimized_sample_time:.2f}x")

def test_monotonicity_calculation():
    """Compare standard monotonicity calculation with JIT-compiled version"""
    from game2048 import compute_monotonicity
    
    # Create random boards
    boards = [np.random.randint(0, 2048, size=(4, 4)) for _ in range(1000)]
    
    # Benchmark standard calculation
    start_time = time.time()
    standard_results = [compute_monotonicity(board) for board in boards]
    standard_time = time.time() - start_time
    
    # Benchmark JIT-compiled calculation
    start_time = time.time()
    optimized_results = [compute_monotonicity_fast(board) for board in boards]
    optimized_time = time.time() - start_time
    
    # Check if results match
    match_count = sum(np.isclose(s, o) for s, o in zip(standard_results, optimized_results))
    
    logging.info(f"Monotonicity Calculation Benchmark (1000 boards):")
    logging.info(f"  Standard: {standard_time:.4f} seconds")
    logging.info(f"  JIT-compiled: {optimized_time:.4f} seconds")
    logging.info(f"  Speedup: {standard_time / optimized_time:.2f}x")
    logging.info(f"  Results match: {match_count}/1000")

def test_gpu_optimizations():
    """Test GPU optimization settings"""
    if not torch.cuda.is_available():
        logging.info("CUDA not available, skipping GPU optimization test")
        return
    
    # Create a simple model and data
    model = EnhancedPPOAgent(board_size=4, hidden_dim=256, input_channels=16).to('cuda')
    dummy_input = torch.randn(100, 16, 4, 4, device='cuda')
    
    # Time inference without optimizations
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(dummy_input)
    standard_time = time.time() - start_time
    
    # Apply optimizations
    optimize_gpu_utilization()
    
    # Warmup again
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    
    # Time inference with optimizations
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(dummy_input)
    optimized_time = time.time() - start_time
    
    logging.info(f"GPU Optimization Benchmark (100 forward passes):")
    logging.info(f"  Standard: {standard_time:.4f} seconds")
    logging.info(f"  Optimized: {optimized_time:.4f} seconds")
    logging.info(f"  Speedup: {standard_time / optimized_time:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Test performance optimizations")
    parser.add_argument("--test-preprocessing", action="store_true", help="Test preprocessing optimizations")
    parser.add_argument("--test-replay-buffer", action="store_true", help="Test replay buffer optimizations")
    parser.add_argument("--test-monotonicity", action="store_true", help="Test monotonicity calculation optimizations")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU optimizations")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all
    run_all = args.test_all or not (args.test_preprocessing or args.test_replay_buffer or 
                                    args.test_monotonicity or args.test_gpu)
    
    logging.info("Starting performance optimization tests")
    
    if args.test_preprocessing or run_all:
        benchmark_preprocessing()
        
    if args.test_replay_buffer or run_all:
        benchmark_replay_buffer()
        
    if args.test_monotonicity or run_all:
        test_monotonicity_calculation()
        
    if args.test_gpu or run_all:
        test_gpu_optimizations()
    
    logging.info("Performance optimization tests complete")

if __name__ == "__main__":
    main() 