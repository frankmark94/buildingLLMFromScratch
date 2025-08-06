#!/usr/bin/env python3
"""
Evaluation package for neural LLM.
Contains perplexity calculation, benchmarks, and comprehensive metrics.
"""

from .perplexity import (
    PerplexityEvaluator, LanguageModelingMetrics,
    evaluate_model_comprehensive, compare_model_performance,
    perplexity_over_training, calculate_bits_per_byte
)
from .benchmarks import (
    TextClassificationBenchmark, GenerationQualityBenchmark, ReasoningBenchmark,
    ComprehensiveBenchmark, create_sample_benchmark_data,
    save_benchmark_results, load_benchmark_results
)
from .metrics import (
    LanguageModelMetrics, GenerationMetrics, CalibrationMetrics, LossAnalysis,
    compute_comprehensive_metrics, compare_model_metrics, print_metrics_summary
)

__all__ = [
    # Perplexity evaluation
    'PerplexityEvaluator',
    'LanguageModelingMetrics',
    'evaluate_model_comprehensive',
    'compare_model_performance', 
    'perplexity_over_training',
    'calculate_bits_per_byte',
    
    # Benchmarking
    'TextClassificationBenchmark',
    'GenerationQualityBenchmark',
    'ReasoningBenchmark',
    'ComprehensiveBenchmark',
    'create_sample_benchmark_data',
    'save_benchmark_results',
    'load_benchmark_results',
    
    # Metrics
    'LanguageModelMetrics',
    'GenerationMetrics',
    'CalibrationMetrics',
    'LossAnalysis',
    'compute_comprehensive_metrics',
    'compare_model_metrics',
    'print_metrics_summary',
]