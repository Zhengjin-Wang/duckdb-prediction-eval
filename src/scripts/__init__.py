"""
Initialization module for scripts package.
"""
from .download_imdb import download_imdb_dataset
from .generate_workload import generate_workload
from .run_workload import load_dataset_to_duckdb, run_workload
from .train_model import train_baseline_model, load_workload_results
from .evaluate_model import evaluate_baseline_model, compare_models, load_workload_results as load_test_workload

__all__ = [
    'download_imdb_dataset',
    'generate_workload',
    'load_dataset_to_duckdb',
    'run_workload',
    'train_baseline_model',
    'load_workload_results',
    'evaluate_baseline_model',
    'compare_models',
    'load_test_workload'
]