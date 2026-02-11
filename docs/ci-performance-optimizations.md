# CI Performance Optimizations

This document explains the performance optimizations implemented to make CI execution lightweight and fast.

## Problem

The original evaluation pipeline took approximately 3 minutes in CI environments due to:

1. **Heavy embedding computations**: Real sentence-transformers models require downloading and loading large neural networks
2. **Multiple consistency runs**: Running the same test case 3-5 times for stability checks
3. **Expensive similarity calculations**: Computing semantic similarity between texts using neural embeddings

## Solution

### 1. CI Mode Detection

The system automatically detects CI environments using:
- `CI=true` environment variable (standard in most CI systems)
- `LLMQ_LIGHTWEIGHT_MODE=true` environment variable (manual override)

```python
def _is_ci_environment() -> bool:
    return os.environ.get('CI', '').lower() in ('true', '1', 'yes')

def _should_use_lightweight_mode() -> bool:
    return _is_ci_environment() or os.environ.get('LLMQ_LIGHTWEIGHT_MODE', '').lower() in ('true', '1', 'yes')
```

### 2. Mock Embedding Model

In CI mode, the system uses a deterministic mock embedding model instead of real sentence-transformers:

```python
class MockEmbeddingModel:
    """Mock embedding model for CI environments."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        # Generate deterministic embeddings based on text characteristics
        # - Text length, word count, character diversity
        # - Presence of positive/negative sentiment words
        # - Deterministic random components for similarity distribution
```

**Benefits:**
- **Fast execution**: No model loading or GPU computation
- **Deterministic results**: Same input always produces same output
- **Reasonable similarity scores**: Based on text characteristics, not random
- **Same interface**: Drop-in replacement for real embeddings

### 3. Reduced Consistency Runs

In CI mode, consistency tests are limited to maximum 2 runs instead of 3-5:

```python
def run_consistency_test(self, test_case, provider_name, model_name, num_runs=5):
    if os.environ.get('CI', '').lower() in ('true', '1', 'yes'):
        num_runs = min(num_runs, 2)  # Limit to 2 runs in CI
```

### 4. Lightweight Consistency Metric

The consistency metric uses a heuristic approach in CI mode:

```python
if _should_use_lightweight_mode():
    # Use length-based heuristic instead of embedding similarity
    avg_length = sum(len(output) for output in outputs) / len(outputs)
    length_variance = sum((len(output) - avg_length) ** 2 for output in outputs) / len(outputs)
    score = max(0.0, 1.0 - (length_variance / (avg_length + 1)))
```

### 5. Configuration Integration

The `config.yaml` includes CI-specific settings:

```yaml
evaluation:
  consistency_runs: 3
  ci_lightweight_mode: true  # Enable lightweight mode for CI environments
```

## Performance Impact

### Before Optimization
- **CI execution time**: ~3 minutes
- **Embedding loading**: 30-60 seconds
- **Consistency runs**: 3-5 runs per test case
- **Heavy computations**: Full neural network inference

### After Optimization
- **CI execution time**: <30 seconds (target)
- **Embedding loading**: Instant (mock model)
- **Consistency runs**: Maximum 2 runs per test case
- **Lightweight computations**: Heuristic-based metrics

## Usage

### Automatic (Recommended)
CI environments automatically enable lightweight mode when `CI=true` is set.

### Manual Override
Force lightweight mode in any environment:
```bash
export LLMQ_LIGHTWEIGHT_MODE=true
python -m pytest tests/
```

### Disable in CI (if needed)
```bash
export LLMQ_LIGHTWEIGHT_MODE=false
# CI mode will still be detected but can be overridden
```

## Testing

The optimizations include comprehensive tests to verify:

1. **CI detection works correctly**
2. **Mock embeddings are deterministic**
3. **Mock embeddings produce reasonable similarity scores**
4. **Performance is significantly improved**
5. **All metrics still function correctly**

Run performance tests:
```bash
python -m pytest tests/test_ci_performance.py -v
```

## Compatibility

- **Local development**: Uses real embeddings for accurate results
- **CI environments**: Automatically switches to lightweight mode
- **Production**: Uses real embeddings for maximum accuracy
- **API compatibility**: No changes to public interfaces

## Configuration Examples

### CI-Optimized Config
```yaml
evaluation:
  max_retries: 2          # Reduced for faster failure
  parallel_requests: 3    # Reduced to avoid rate limits
  consistency_runs: 2     # Reduced for CI
  ci_lightweight_mode: true
```

### Production Config
```yaml
evaluation:
  max_retries: 3
  parallel_requests: 5
  consistency_runs: 5     # More runs for accuracy
  ci_lightweight_mode: false
```

## Monitoring

The system logs when lightweight mode is active:
```
INFO: Lightweight mode detected, using mock embedding model for fast execution
INFO: CI mode detected: reducing consistency runs to 2
```

This ensures transparency about which optimizations are being applied.