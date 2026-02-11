#!/usr/bin/env python3
"""
Demo script to show CI performance optimizations in action.
"""

import os
import time
from evals.metrics import TaskSuccessMetric, RelevanceMetric, ConsistencyMetric, _get_embedding_model

def demo_performance():
    """Demonstrate the performance difference between CI and normal mode."""
    
    print("🚀 LLM Quality Gate - CI Performance Optimization Demo")
    print("=" * 60)
    
    # Test data
    test_cases = [
        ("What is the capital of France?", "Paris", "The capital of France is Paris."),
        ("What is 2+2?", "4", "2+2 equals 4"),
        ("Name a programming language", "Python", "Python is a popular programming language"),
    ]
    
    consistency_outputs = [
        "The answer is 42",
        "The answer is forty-two", 
        "42 is the correct answer"
    ]
    
    # Test in normal mode
    print("\n📊 Testing in NORMAL mode...")
    os.environ.pop('CI', None)  # Remove CI flag
    os.environ.pop('LLMQ_LIGHTWEIGHT_MODE', None)
    
    # Clear cached values
    import evals.metrics
    evals.metrics._embedding_model = None
    evals.metrics._is_ci_mode = None
    
    start_time = time.time()
    
    task_metric = TaskSuccessMetric()
    relevance_metric = RelevanceMetric()
    consistency_metric = ConsistencyMetric()
    
    model = _get_embedding_model()
    print(f"   Embedding model: {type(model).__name__}")
    
    for prompt, expected, generated in test_cases:
        task_result = task_metric.evaluate(generated, expected)
        relevance_result = relevance_metric.evaluate(prompt, generated)
    
    consistency_result = consistency_metric.evaluate(consistency_outputs)
    
    normal_time = time.time() - start_time
    print(f"   ⏱️  Normal mode time: {normal_time:.3f} seconds")
    
    # Test in CI mode
    print("\n🏃‍♂️ Testing in CI mode...")
    os.environ['CI'] = 'true'
    
    # Clear cached values to force re-detection
    evals.metrics._embedding_model = None
    evals.metrics._is_ci_mode = None
    
    start_time = time.time()
    
    task_metric = TaskSuccessMetric()
    relevance_metric = RelevanceMetric()
    consistency_metric = ConsistencyMetric()
    
    model = _get_embedding_model()
    print(f"   Embedding model: {type(model).__name__}")
    
    for prompt, expected, generated in test_cases:
        task_result = task_metric.evaluate(generated, expected)
        relevance_result = relevance_metric.evaluate(prompt, generated)
    
    consistency_result = consistency_metric.evaluate(consistency_outputs)
    
    ci_time = time.time() - start_time
    print(f"   ⏱️  CI mode time: {ci_time:.3f} seconds")
    
    # Show results
    print(f"\n📈 Performance Improvement:")
    if normal_time > 0:
        speedup = normal_time / ci_time
        time_saved = normal_time - ci_time
        print(f"   🚀 Speedup: {speedup:.1f}x faster")
        print(f"   ⏰ Time saved: {time_saved:.3f} seconds")
        print(f"   📉 CI mode is {((normal_time - ci_time) / normal_time * 100):.1f}% faster")
    
    print(f"\n✅ CI Optimizations Active:")
    print(f"   • Mock embeddings: ✓")
    print(f"   • Reduced consistency runs: ✓") 
    print(f"   • Lightweight similarity computation: ✓")
    print(f"   • Deterministic results: ✓")
    
    print(f"\n🎯 Expected CI Performance:")
    print(f"   • Target: <30 seconds for full evaluation")
    print(f"   • Actual demo time: {ci_time:.3f} seconds")
    print(f"   • Status: {'✅ PASS' if ci_time < 1.0 else '⚠️  SLOW'}")

if __name__ == "__main__":
    demo_performance()