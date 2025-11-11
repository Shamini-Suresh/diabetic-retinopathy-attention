#!/usr/bin/env python
# coding: utf-8

"""
Seed Utility Module for APTOS Diabetic Retinopathy Detection
Ensures reproducibility across all random operations

Usage:
    from utils.seed import set_all_seeds, set_worker_seeds
    
    # Set all seeds at the start of your script
    set_all_seeds(42)
    
    # Add to DataLoader for worker reproducibility
    DataLoader(..., worker_init_fn=set_worker_seeds)
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed=42, cudnn_deterministic=True, cudnn_benchmark=False):
    """
    Set all random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA
    
    Args:
        seed (int): Random seed value (default: 42)
        cudnn_deterministic (bool): Enable deterministic CUDNN operations (default: True)
            - When True, ensures reproducible results but may reduce speed by 10-20%
        cudnn_benchmark (bool): Enable CUDNN auto-tuner (default: False)
            - When False, ensures reproducibility
            - When True, may improve performance but reduces reproducibility
    
    Note:
        - Call this function BEFORE any imports that use randomness
        - Setting cudnn_deterministic=True may reduce training speed by 10-20%
        - For maximum reproducibility, keep cudnn_benchmark=False
        - For faster training without reproducibility guarantees, set:
          cudnn_deterministic=False, cudnn_benchmark=True
    
    Example:
        >>> from utils.seed import set_all_seeds
        >>> set_all_seeds(42)
        ✓ All random seeds set to: 42
          - CUDNN Deterministic: True
          - CUDNN Benchmark: False
          ⚠️  Deterministic mode enabled (may reduce speed by 10-20%)
    """
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # CUDNN settings for reproducibility
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    
    # Set environment variable for CUBLAS (additional CUDA reproducibility)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch use deterministic algorithms (PyTorch >= 1.8)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Older PyTorch versions don't have this function
        pass
    
    # Print confirmation
    print(f"✓ All random seeds set to: {seed}")
    print(f"  - CUDNN Deterministic: {cudnn_deterministic}")
    print(f"  - CUDNN Benchmark: {cudnn_benchmark}")
    if cudnn_deterministic:
        print("  ⚠️  Deterministic mode enabled (may reduce speed by 10-20%)")


def set_worker_seeds(worker_id):
    """
    Set random seeds for DataLoader workers to ensure reproducibility
    
    This function is designed to be used with PyTorch's DataLoader worker_init_fn
    parameter. Each worker gets a unique but deterministic seed based on the
    initial seed and worker ID.
    
    Args:
        worker_id (int): Worker ID assigned by DataLoader (0 to num_workers-1)
    
    Usage:
        >>> from torch.utils.data import DataLoader
        >>> from utils.seed import set_worker_seeds
        >>> 
        >>> train_loader = DataLoader(
        ...     dataset,
        ...     batch_size=16,
        ...     num_workers=4,
        ...     worker_init_fn=set_worker_seeds  # Add this line
        ... )
    
    Note:
        - This ensures that each worker produces reproducible results
        - Workers get different but deterministic seeds
        - Essential for multi-worker DataLoaders (num_workers > 0)
    """
    # Each worker gets a unique but deterministic seed
    # Based on PyTorch's initial seed and worker ID
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state():
    """
    Get current random number generator states for all libraries
    
    Returns:
        dict: Dictionary containing RNG states for Python, NumPy, PyTorch, and CUDA
            - 'python': Python random state
            - 'numpy': NumPy random state
            - 'torch': PyTorch CPU random state
            - 'torch_cuda': PyTorch CUDA random states (None if CUDA unavailable)
    
    Usage:
        >>> from utils.seed import get_rng_state, set_rng_state
        >>> 
        >>> # Save current state
        >>> state = get_rng_state()
        >>> 
        >>> # ... do some random operations ...
        >>> 
        >>> # Restore state to reproduce results
        >>> set_rng_state(state)
    
    Note:
        Useful for:
        - Saving state before validation/testing
        - Debugging random behavior
        - Creating reproducible checkpoints
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_rng_state(state_dict):
    """
    Restore random number generator states for all libraries
    
    Args:
        state_dict (dict): Dictionary containing RNG states from get_rng_state()
    
    Usage:
        >>> from utils.seed import get_rng_state, set_rng_state
        >>> 
        >>> # Save state
        >>> state = get_rng_state()
        >>> 
        >>> # Generate some random numbers
        >>> x1 = torch.rand(10)
        >>> 
        >>> # Restore state
        >>> set_rng_state(state)
        >>> 
        >>> # Generate same random numbers again
        >>> x2 = torch.rand(10)
        >>> 
        >>> # x1 and x2 will be identical
        >>> assert torch.allclose(x1, x2)
    
    Note:
        Useful for:
        - Reproducing specific random sequences
        - Debugging random behavior
        - Implementing custom training loops with save/resume
    """
    random.setstate(state_dict['python'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    
    if torch.cuda.is_available() and state_dict['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state_dict['torch_cuda'])
    
    print("✓ Random number generator states restored")


def test_reproducibility(seed=42, verbose=True):
    """
    Test if seed setting produces reproducible results
    
    Args:
        seed (int): Seed to test (default: 42)
        verbose (bool): Print detailed test results (default: True)
    
    Returns:
        bool: True if reproducibility test passes, False otherwise
    
    Usage:
        >>> from utils.seed import test_reproducibility
        >>> 
        >>> # Quick test
        >>> test_reproducibility()
        Testing reproducibility with seed 42...
        ✓ Python random: PASS
        ✓ NumPy random: PASS
        ✓ PyTorch random: PASS
        ✓ All reproducibility tests PASSED!
        True
    """
    if verbose:
        print(f"Testing reproducibility with seed {seed}...")
        print("="*50)
    
    all_passed = True
    
    # Test 1: Python random
    set_all_seeds(seed)
    python_result1 = random.random()
    
    set_all_seeds(seed)
    python_result2 = random.random()
    
    python_pass = (python_result1 == python_result2)
    all_passed = all_passed and python_pass
    
    if verbose:
        status = "✓ PASS" if python_pass else "✗ FAIL"
        print(f"{status} - Python random: {python_result1:.10f}")
    
    # Test 2: NumPy random
    set_all_seeds(seed)
    numpy_result1 = np.random.random()
    
    set_all_seeds(seed)
    numpy_result2 = np.random.random()
    
    numpy_pass = (numpy_result1 == numpy_result2)
    all_passed = all_passed and numpy_pass
    
    if verbose:
        status = "✓ PASS" if numpy_pass else "✗ FAIL"
        print(f"{status} - NumPy random: {numpy_result1:.10f}")
    
    # Test 3: PyTorch random
    set_all_seeds(seed)
    torch_result1 = torch.rand(5)
    
    set_all_seeds(seed)
    torch_result2 = torch.rand(5)
    
    torch_pass = torch.allclose(torch_result1, torch_result2)
    all_passed = all_passed and torch_pass
    
    if verbose:
        status = "✓ PASS" if torch_pass else "✗ FAIL"
        print(f"{status} - PyTorch random")
        print(f"  Sample 1: {torch_result1[:3]}")
        print(f"  Sample 2: {torch_result2[:3]}")
    
    if verbose:
        print("="*50)
        if all_passed:
            print("✓ All reproducibility tests PASSED!")
        else:
            print("✗ Some tests FAILED - reproducibility may not be guaranteed")
    
    return all_passed


# Example usage and testing
if __name__ == "__main__":
    print("Seed Utility Module - Test Suite")
    print("="*70)
    
    # Test 1: Basic seed setting
    print("\n1. Testing basic seed setting:")
    print("-"*70)
    set_all_seeds(42)
    
    # Test 2: Generate sample random numbers
    print("\n2. Generating sample random numbers with seed 42:")
    print("-"*70)
    print(f"Python random:  {random.random():.10f}")
    print(f"NumPy random:   {np.random.random():.10f}")
    print(f"PyTorch random: {torch.rand(1).item():.10f}")
    
    # Test 3: Reproducibility test
    print("\n3. Running reproducibility test:")
    print("-"*70)
    test_passed = test_reproducibility(seed=42, verbose=True)
    
    # Test 4: State save/restore
    print("\n4. Testing RNG state save/restore:")
    print("-"*70)
    set_all_seeds(42)
    state = get_rng_state()
    before = torch.rand(3)
    print(f"Before restore: {before}")
    
    # Generate some numbers to change state
    _ = torch.rand(100)
    
    # Restore state
    set_rng_state(state)
    after = torch.rand(3)
    print(f"After restore:  {after}")
    print(f"Are they equal? {torch.allclose(before, after)}")
    
    # Test 5: Worker seed function
    print("\n5. Testing worker seed function:")
    print("-"*70)
    print("Simulating worker seeding:")
    for worker_id in range(4):
        set_worker_seeds(worker_id)
        sample = np.random.random()
        print(f"  Worker {worker_id}: {sample:.10f}")
    
    print("\n" + "="*70)
    if test_passed:
        print("✅ All tests completed successfully!")
        print("Seed utility is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check your environment.")
    print("="*70)
