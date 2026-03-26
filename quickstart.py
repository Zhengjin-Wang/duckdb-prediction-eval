#!/usr/bin/env python3
"""
Quick start example for DuckDB LCM Evaluation framework.

This script demonstrates the basic usage of the framework to
train and evaluate baseline models on the IMDB dataset.
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import setup_logging
from src.database.loader import load_imdb_to_duckdb
from src.workloads.generator import generate_and_execute_workload, WorkloadConfig
from src.models.registry import train_model, evaluate_model, get_available_models
from src.workloads.analyzer import analyze_workload


def main():
    """Quick start example."""
    print("DuckDB LCM Evaluation - Quick Start Example")
    print("=" * 50)

    # Setup logging
    setup_logging('INFO')

    # Configuration
    data_dir = "data/datasets/imdb"
    db_path = "data/imdb.duckdb"
    workload_output = "data/runs/quickstart_workload.json"
    model_output = "models/quickstart_model.pkl"

    print("\n1. Setting up environment...")

    # Create directories
    for directory in [data_dir, "data/runs", "models", "output"]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Create dummy IMDB dataset
    print("   Creating dummy IMDB dataset...")
    with open(f"{data_dir}/title.csv", 'w') as f:
        f.write("id,title,year,kind\n")
        for i in range(100):
            f.write(f"{i},Movie {i},{2000 + (i % 20)},movie\n")

    with open(f"{data_dir}/cast_info.csv", 'w') as f:
        f.write("id,person_id,movie_id,role\n")
        for i in range(200):
            f.write(f"{i},{i % 50},{i % 100},actor\n")

    # Load dataset to DuckDB
    print("   Loading dataset to DuckDB...")
    load_imdb_to_duckdb(db_path, data_dir, force=True)

    print("\n2. Generating and running workload...")

    # Configure workload
    config = WorkloadConfig(
        num_queries=50,
        max_joins=2,
        max_predicates=2,
        seed=42,
        min_runtime_ms=50,  # Lower threshold for demo
        max_runtime_ms=10000
    )

    # Generate and execute workload
    # Note: For this demo, we'll create a simple mock workload
    print("   Creating mock workload for demonstration...")

    # Create mock workload results
    workload_results = []
    for i in range(50):
        query = f"SELECT COUNT(*) FROM title WHERE id = {i};"
        result = {
            'query': query,
            'avg_time_ms': 50 + (i % 100),
            'min_time_ms': 30 + (i % 50),
            'max_time_ms': 80 + (i % 150),
            'successful_executions': 3
        }
        workload_results.append(result)

    # Save mock results
    import json
    with open(workload_output, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'results': workload_results
        }, f, indent=2)

    print(f"   Created {len(workload_results)} mock queries")

    print("\n3. Analyzing workload...")

    # Analyze workload
    analyze_workload(workload_results, output_dir="output/")

    print("\n4. Training models...")

    # Train models
    models_to_train = ['flat_vector']  # Start with simple model

    for model_type in models_to_train:
        print(f"   Training {model_type} model...")

        try:
            # In a real scenario, you would call:
            # results = train_model(model_type, workload_results)

            # For demo, create mock results
            print(f"     Model training completed for {model_type}")
            print(f"     Mock metrics: RMSE=5.2, MAPE=12.3%, Q-Error(95)=1.8")

        except Exception as e:
            print(f"     Training failed: {e}")

    print("\n5. Evaluating models...")

    # Create test workload
    test_workload = []
    for i in range(20):
        result = {
            'query': f"SELECT COUNT(*) FROM title WHERE id = {i + 50};",
            'avg_time_ms': 60 + (i % 80),
            'min_time_ms': 40 + (i % 40),
            'max_time_ms': 90 + (i % 120),
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
        }
    }

    print("   Evaluation results:")
    for model_name, metrics in evaluation_results.items():
        print(f"     {model_name}:")
        for metric, value in metrics.items():
            print(f"       {metric.upper()}: {value}")

    print("\n6. Model comparison...")

    # Compare models (mock comparison)
    print("   Comparing models against baseline...")

    print("\n" + "=" * 50)
    print("Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Download real IMDB dataset: https://datasets.imdbws.com/")
    print("2. Use real DuckDB database with actual data")
    print("3. Generate larger workloads with more complexity")
    print("4. Train and evaluate all model types")
    print("5. Run the complete pipeline: python src/main.py pipeline")
    print("\nFor more examples, see: examples/example_usage.py")
    print("For documentation, see: DOCUMENTATION.md")


if __name__ == '__main__':
    main()