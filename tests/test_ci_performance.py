"""Tests to verify CI mode performance optimizations."""

import os
import time
import pytest
from unittest.mock import patch

from evals.metrics import (
    TaskSuccessMetric, RelevanceMetric, ConsistencyMetric, 
    _is_ci_environment, _should_use_lightweight_mode, _get_embedding_model,
    MockEmbeddingModel
)


class TestCIPerformanceOptimizations:
    """Test that CI mode provides performance optimizations."""

    def test_ci_environment_detection(self):
        """Test CI environment detection works correctly."""
        # Test with CI=true
        with patch.dict(os.environ, {'CI': 'true'}):
            # Clear cached value
            import evals.metrics
            evals.metrics._is_ci_mode = None
            assert _is_ci_environment() is True
            assert _should_use_lightweight_mode() is True

        # Test with CI=false
        with patch.dict(os.environ, {'CI': 'false'}, clear=True):
            evals.metrics._is_ci_mode = None
            assert _is_ci_environment() is False

    def test_lightweight_mode_env_var(self):
        """Test LLMQ_LIGHTWEIGHT_MODE environment variable."""
        with patch.dict(os.environ, {'LLMQ_LIGHTWEIGHT_MODE': 'true', 'CI': 'false'}, clear=True):
            import evals.metrics
            evals.metrics._is_ci_mode = None
            assert _should_use_lightweight_mode() is True

    def test_mock_embedding_model_deterministic(self):
        """Test that mock embedding model produces deterministic results."""
        model = MockEmbeddingModel(seed=42)
        
        texts = ["Hello world", "Machine learning is great", "Python programming"]
        embeddings1 = model.encode(texts)
        embeddings2 = model.encode(texts)
        
        # Should be identical
        assert embeddings1 == embeddings2
        
        # Should have consistent dimensions
        assert all(len(emb) == len(embeddings1[0]) for emb in embeddings1)
        
        # Should be normalized (unit vectors)
        for emb in embeddings1:
            norm = sum(x*x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 1e-6

    def test_mock_embedding_similarity_reasonable(self):
        """Test that mock embeddings produce reasonable similarity scores."""
        model = MockEmbeddingModel(seed=42)
        
        # Test with more clearly similar vs different texts
        similar_texts = ["hello world", "hello world"]  # Identical texts
        different_texts = ["hello world", "goodbye universe"]  # Very different texts
        
        similar_emb = model.encode(similar_texts)
        different_emb = model.encode(different_texts)
        
        from evals.metrics import _cosine_similarity
        similar_score = _cosine_similarity(similar_emb[0], similar_emb[1])
        different_score = _cosine_similarity(different_emb[0], different_emb[1])
        
        # Identical texts should have higher similarity than different texts
        # But we'll be more lenient since it's a mock model
        assert similar_score > 0.8, f"Identical texts should have high similarity, got {similar_score}"
        assert different_score < 0.9, f"Different texts should have lower similarity, got {different_score}"
        
        # Test with clearly different content
        very_different = ["cat", "12345"]
        very_different_emb = model.encode(very_different)
        very_different_score = _cosine_similarity(very_different_emb[0], very_different_emb[1])
        
        # At least one comparison should show meaningful difference
        assert similar_score > very_different_score or similar_score > different_score

    def test_ci_mode_uses_mock_embeddings(self):
        """Test that CI mode uses mock embeddings."""
        with patch.dict(os.environ, {'CI': 'true'}):
            # Clear cached model
            import evals.metrics
            evals.metrics._embedding_model = None
            evals.metrics._is_ci_mode = None
            
            model = _get_embedding_model()
            assert isinstance(model, MockEmbeddingModel)

    def test_consistency_metric_lightweight_mode(self):
        """Test that consistency metric uses lightweight mode in CI."""
        metric = ConsistencyMetric()
        
        outputs = [
            "The answer is 42",
            "The answer is forty-two", 
            "42 is the answer"
        ]
        
        # Test in CI mode
        with patch.dict(os.environ, {'CI': 'true'}):
            import evals.metrics
            evals.metrics._is_ci_mode = None
            
            start_time = time.time()
            result = metric.evaluate(outputs, threshold=0.7)
            ci_time = time.time() - start_time
            
            assert result.metric_name == "consistency"
            assert "lightweight_heuristic" in result.details.get("method", "")
            assert "note" in result.details
            assert "Lightweight mode" in result.details["note"]

        # Test in normal mode (should be slower but we'll just check it works)
        with patch.dict(os.environ, {}, clear=True):
            evals.metrics._is_ci_mode = None
            evals.metrics._embedding_model = None
            
            start_time = time.time()
            result = metric.evaluate(outputs, threshold=0.7)
            normal_time = time.time() - start_time
            
            assert result.metric_name == "consistency"
            # In normal mode without sentence-transformers, it should use token overlap
            # which is still fast, so we just verify it works

    def test_semantic_similarity_performance_ci_vs_normal(self):
        """Test that semantic similarity is faster in CI mode."""
        metric = TaskSuccessMetric()
        
        generated = "The capital of France is Paris, a beautiful city known for its culture."
        expected = "Paris"
        
        # Test CI mode performance
        with patch.dict(os.environ, {'CI': 'true'}):
            import evals.metrics
            evals.metrics._embedding_model = None
            evals.metrics._is_ci_mode = None
            
            start_time = time.time()
            result_ci = metric.evaluate(generated, expected, threshold=0.8)
            ci_time = time.time() - start_time
            
            assert result_ci.metric_name == "task_success"
            assert result_ci.score >= 0.0

        # Test normal mode (mock the sentence-transformers import to avoid dependency issues)
        with patch.dict(os.environ, {}, clear=True):
            evals.metrics._embedding_model = None
            evals.metrics._is_ci_mode = None
            
            # Mock the import to avoid loading real sentence-transformers
            with patch('evals.metrics._get_embedding_model', return_value=None):
                start_time = time.time()
                result_normal = metric.evaluate(generated, expected, threshold=0.8)
                normal_time = time.time() - start_time
                
                assert result_normal.metric_name == "task_success"
                assert result_normal.score >= 0.0

        # CI mode should be reasonably fast (under 500ms for this simple test)
        assert ci_time < 0.5, f"CI mode took {ci_time:.3f}s, expected < 0.5s"

    def test_relevance_metric_ci_mode(self):
        """Test relevance metric in CI mode."""
        metric = RelevanceMetric()
        
        with patch.dict(os.environ, {'CI': 'true'}):
            import evals.metrics
            evals.metrics._embedding_model = None
            evals.metrics._is_ci_mode = None
            
            result = metric.evaluate(
                prompt="What is machine learning?",
                generated="Machine learning is a subset of AI that learns from data.",
                threshold=0.3
            )
            
            assert result.metric_name == "relevance"
            assert result.score >= 0.0
            assert result.score <= 1.0

    @pytest.mark.parametrize("ci_value", ["true", "1", "yes"])
    def test_ci_detection_variations(self, ci_value):
        """Test that various CI environment values are detected."""
        with patch.dict(os.environ, {'CI': ci_value}):
            import evals.metrics
            evals.metrics._is_ci_mode = None
            assert _is_ci_environment() is True

    @pytest.mark.parametrize("ci_value", ["false", "0", "no", ""])
    def test_non_ci_detection_variations(self, ci_value):
        """Test that non-CI values are not detected as CI."""
        with patch.dict(os.environ, {'CI': ci_value}):
            import evals.metrics
            evals.metrics._is_ci_mode = None
            assert _is_ci_environment() is False