#!/usr/bin/env python3
"""
Example usage of DuckDB LCM Evaluation framework.

This script demonstrates how to use the framework to:
1. Setup IMDB dataset
2. Generate and run workloads
3. Train baseline models
4. Evaluate models
5. Compare model performance
"""
import os
import json
import time
from pathlib import Path

# Import framework components
from src.database.loader import load_imdb_to_duckdb
from src.workloads.generator import generate_and_execute_workload, WorkloadConfig
from src.workloads.analyzer import analyze_workload
from src.models.registry import train_model, evaluate_model, get_available_models
from src.utils.metrics import ModelEvaluator


def setup_environment():
    """Setup directories and environment."""
    directories = [
        'data/datasets/imdb',
        'data/runs',
        'models',
        'workloads',
        'output'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Environment setup completed")


def example_dataset_setup():
    """Example: Setup IMDB dataset."""
    print("\n=== Dataset Setup Example ===")

    db_path = "data/imdb.duckdb"
    data_dir = "data/datasets/imdb"

    # Create dummy dataset files for demonstration
    print("Creating dummy IMDB dataset...")

    # Create title.csv
    with open(f"{data_dir}/title.csv", 'w') as f:
        f.write("id,title,year,kind\n")
        for i in range(1000):
            f.write(f"{i},Movie {i},{2000 + (i % 20)},movie\n")

    # Create name.csv
    with open(f"{data_dir}/name.csv", 'w') as f:
        f.write("id,name,birth_year,death_year\n")
        for i in range(500):
            f.write(f"{i},Person {i},{1950 + (i % 50)},\n")

    # Create cast_info.csv
    with open(f"{data_dir}/cast_info.csv", 'w') as f:
        f.write("id,person_id,movie_id,role\n")
        for i in range(2000):
            f.write(f"{i},{i % 500},{i % 1000},actor\n")

    print(f"Created dummy dataset in {data_dir}")
    print("Note: In practice, you would download the actual IMDB dataset")


def example_workload_generation():
    """Example: Generate SQL workload."""
    print("\n=== Workload Generation Example ===")

    config = WorkloadConfig(
        num_queries=100,
        max_joins=2,
        max_predicates=2,
        seed=42
    )

    # Generate simple workload for demonstration
    workload_file = "workloads/example_workload.sql"
    Path("workloads").mkdir(exist_ok=True)

    with open(workload_file, 'w') as f:
        for i in range(config.num_queries):
            if i % 3 == 0:
                query = f"SELECT COUNT(*) FROM title WHERE id = {i};"
            elif i % 3 == 1:
                query = f"SELECT title FROM title WHERE id BETWEEN {i} AND {i + 10};"
            else:
                query = f"SELECT COUNT(*) FROM title t JOIN cast_info c ON t.id = c.movie_id WHERE c.person_id = {i % 100};"
            f.write(query + '\n')

    print(f"Generated {config.num_queries} queries to {workload_file}")
    return workload_file


def example_model_training():
    """Example: Train baseline models."""
    print("\n=== Model Training Example ===")

    # Create dummy workload results for training
    workload_results = []
    for i in range(100):
        result = {
            'query': f"SELECT COUNT(*) FROM title WHERE id = {i};",
            'avg_time_ms': 10 + (i % 50),
            'min_time_ms': 5 + (i % 20),
            'max_time_ms': 20 + (i % 80),
            'successful_executions': 3
        }
        workload_results.append(result)

    # Train different models
    models_to_train = ['flat_vector']  # Start with simple model

    trained_models = {}
    for model_type in models_to_train:
        print(f"Training {model_type} model...")

        try:
            # In practice, you would call:
            # results = train_model(model_type, workload_results)

            # For demo, create mock results
            results = {
                'training_time': 1.5,
                'rmse': 5.2,
                'mape': 12.3,
                'q_error_95': 1.8
            }

            trained_models[model_type] = results
            print(f"  Training completed: RMSE={results['rmse']:.2f}, MAPE={results['mape']:.2f}%")

        except Exception as e:
            print(f"  Training failed: {e}")

    return trained_models


def example_model_evaluation():
    """Example: Evaluate models."""
    print("\n=== Model Evaluation Example ===")

    # Create test workload
    test_workload = []
    for i in range(50):
        result = {
            'query': f"SELECT COUNT(*) FROM title WHERE id = {i + 100};",
            'avg_time_ms': 15 + (i % 40),
            'min_time_ms': 8 + (i % 15),
            'max_time_ms': 25 + (i % 60),
            'successful_executions': 3
        }
        test_workload.append(result)

    # Mock evaluation results
    evaluation_results = {
        'flat_vector': {
            'rmse': 6.1,
            'mape': 15.2,
            'q_error_95': 2.1,
            'r_squared': 0.85
        },
        'mscn': {
            'rmse': 4.8,
            'mape': 11.7,
            'q_error_95': 1.6,
            'r_squared': 0.89
        },
        'qppnet': {
            'rmse': 4.2,
            'mape': 9.8,
            'q_error_95': 1.4,
            'r_squared': 0.92
        }
    }

    print("Model evaluation results:")
    for model_name, metrics in evaluation_results.items():
        print(f"  {model_name}:")
        print(f"    RMSE: {metrics['rmse']:.2f}")
        print(f"    MAPE: {metrics['mape']:.2f}%")
        print(f"    Q-Error(95): {metrics['q_error_95']:.2f}")
        print(f"    R²: {metrics['r_squared']:.2f}")

    return evaluation_results


def example_model_comparison(evaluation_results):
    """Example: Compare model performance."""
    print("\n=== Model Comparison Example ===")

    baseline = 'flat_vector'
    print(f"Comparing models against {baseline} baseline:")

    for model_name, metrics in evaluation_results.items():
        if model_name == baseline:
            continue

        baseline_metrics = evaluation_results[baseline]

        print(f"\n{model_name} vs {baseline}:")
        for metric in ['rmse', 'mape', 'q_error_95']:
            baseline_val = baseline_metrics[metric]
            model_val = metrics[metric]

            if metric in ['rmse', 'mape', 'q_error_95']:
                # Lower is better
                improvement = ((baseline_val - model_val) / baseline_val) * 100
                print(f"  {metric.upper()}: {improvement:+.1f}% improvement")
            elif metric == 'r_squared':
                # Higher is better
                improvement = ((model_val - baseline_val) / baseline_val) * 100
                print(f"  R²: {improvement:+.1f}% improvement")


def example_pipeline():
    """Example: Complete pipeline execution."""
    print("\n=== Complete Pipeline Example ===")

    start_time = time.time()

    # Step 1: Setup
    print("Step 1: Setting up environment...")
    setup_environment()

    # Step 2: Dataset setup
    print("Step 2: Setting up dataset...")
    example_dataset_setup()

    # Step 3: Workload generation
    print("Step 3: Generating workload...")
    workload_file = example_workload_generation()

    # Step 4: Model training
    print("Step 4: Training models...")
    trained_models = example_model_training()

    # Step 5: Model evaluation
    print("Step 5: Evaluating models...")
    evaluation_results = example_model_evaluation()

    # Step 6: Model comparison
    print("Step 6: Comparing models...")
    example_model_comparison(evaluation_results)

    total_time = time.time() - start_time
    print(f"\nPipeline completed in {total_time:.2f} seconds")


def main():
    """Main example execution."""
    print("DuckDB LCM Evaluation Framework - Examples")
    print("=" * 50)

    # Run complete pipeline example
    example_pipeline()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nTo use with real data:")
    print("1. Download actual IMDB dataset")
    print("2. Use real DuckDB database")
    print("3. Generate actual workloads")
    print("4. Train and evaluate models on real data")


if __name__ == '__main__':
    main()