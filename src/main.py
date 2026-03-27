#!/usr/bin/env python3
"""
Main entry point for DuckDB LCM Evaluation.
"""
import argparse
import sys
import logging
from pathlib import Path

from src.utils.config import setup_logging, load_default_config
from src.database.loader import load_imdb_to_duckdb
from src.workloads.generator import generate_and_execute_workload, WorkloadConfig
from src.workloads.analyzer import analyze_workload
from src.models.registry import train_model, evaluate_model, get_available_models
import json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='DuckDB LCM Evaluation - Evaluate Learned Cost Models on IMDB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and setup IMDB dataset
  python main.py setup --data_dir data/datasets/imdb --db_path data/imdb.duckdb

  # Generate workload
  python main.py generate --num_queries 1000 --max_joins 3 --output workloads/imdb_workload.sql

  # Run workload on DuckDB
  python main.py run --db_path data/imdb.duckdb --workload workloads/imdb_workload.sql --output data/runs/imdb_runs.json

  # Train baseline models
  python main.py train --model_type flat_vector --workload data/runs/imdb_runs.json --device cpu

  # Evaluate models
  python main.py evaluate --model_type flat_vector --model_path models/flat_vector_model.pkl --test_workload data/runs/imdb_runs.json

  # Full pipeline
  python main.py pipeline --num_queries 500 --max_joins 2 --train_all
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup IMDB dataset')
    setup_parser.add_argument('--data_dir', default='data/datasets/imdb', help='Dataset directory')
    setup_parser.add_argument('--db_path', default='data/imdb.duckdb', help='Database path')
    setup_parser.add_argument('--force', action='store_true', help='Force reload dataset')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate SQL workload')
    generate_parser.add_argument('--num_queries', type=int, default=1000, help='Number of queries')
    generate_parser.add_argument('--max_joins', type=int, default=3, help='Maximum joins')
    generate_parser.add_argument('--max_predicates', type=int, default=3, help='Maximum predicates')
    generate_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    generate_parser.add_argument('--output', default='workloads/imdb_workload.sql', help='Output file')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run workload on DuckDB')
    run_parser.add_argument('--db_path', default='data/imdb.duckdb', help='Database path')
    run_parser.add_argument('--data_dir', default='data/datasets/imdb', help='Dataset directory')
    run_parser.add_argument('--workload', required=True, help='Workload file')
    run_parser.add_argument('--output', default='data/runs/workload_results.json', help='Output file')
    run_parser.add_argument('--min_runtime', type=int, default=100, help='Minimum runtime (ms)')
    run_parser.add_argument('--max_runtime', type=int, default=300000, help='Maximum runtime (ms)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train baseline models')
    train_parser.add_argument('--model_type', choices=get_available_models(), help='Model type')
    train_parser.add_argument('--workload', required=True, help='Training workload file')
    train_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Training device')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--output_dir', default='models/', help='Output directory')
    train_parser.add_argument('--train_all', action='store_true', help='Train all available models')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model_type', choices=get_available_models(), help='Model type')
    eval_parser.add_argument('--model_path', help='Model file path')
    eval_parser.add_argument('--test_workload', required=True, help='Test workload file')
    eval_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Evaluation device')
    eval_parser.add_argument('--output', default='output/evaluation_results.json', help='Output file')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--data_dir', default='data/datasets/imdb', help='Dataset directory')
    pipeline_parser.add_argument('--db_path', default='data/imdb.duckdb', help='Database path')
    pipeline_parser.add_argument('--num_queries', type=int, default=1000, help='Number of queries')
    pipeline_parser.add_argument('--max_joins', type=int, default=3, help='Maximum joins')
    pipeline_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    pipeline_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Training device')
    pipeline_parser.add_argument('--train_all', action='store_true', help='Train all models')
    pipeline_parser.add_argument('--output_dir', default='output/', help='Output directory')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze workload results')
    analyze_parser.add_argument('--workload_results', required=True, help='Workload results file')
    analyze_parser.add_argument('--output_dir', default='output/', help='Output directory')

    # Add common arguments
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Logging level')
    parser.add_argument('--log_file', help='Log file path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    try:
        if args.command == 'setup':
            setup_dataset(args)
        elif args.command == 'generate':
            generate_workload_cmd(args)
        elif args.command == 'run':
            run_workload_cmd(args)
        elif args.command == 'train':
            train_models_cmd(args)
        elif args.command == 'evaluate':
            evaluate_models_cmd(args)
        elif args.command == 'pipeline':
            run_pipeline(args)
        elif args.command == 'analyze':
            analyze_workload_cmd(args)
        else:
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        logging.error(f"Error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


def setup_dataset(args):
    """Setup IMDB dataset."""
    logging.info("Setting up IMDB dataset...")
    load_imdb_to_duckdb(args.db_path, args.data_dir, force=args.force)
    logging.info("Dataset setup completed")


def generate_workload_cmd(args):
    """Generate SQL workload."""
    logging.info("Generating workload...")

    config = WorkloadConfig(
        num_queries=args.num_queries,
        max_joins=args.max_joins,
        max_predicates=args.max_predicates,
        seed=args.seed
    )

    # For now, create a simple workload file
    workload_file = Path(args.output)
    workload_file.parent.mkdir(parents=True, exist_ok=True)

    with open(workload_file, 'w') as f:
        for i in range(args.num_queries):
            # Generate simple query
            query = f"SELECT COUNT(*) FROM title WHERE id = {i};"
            f.write(query + '\n')

    logging.info(f"Generated {args.num_queries} queries to {args.output}")


def run_workload_cmd(args):
    """Run workload on DuckDB."""
    from src.workloads.generator import WorkloadExecutor, WorkloadConfig
    from src.database.duckdb_connection import create_database_connection

    logging.info("Running workload on DuckDB...")

    config = WorkloadConfig(
        min_runtime_ms=args.min_runtime,
        max_runtime_ms=args.max_runtime
    )

    # Load workload from file
    with open(args.workload, 'r') as f:
        workload_queries = [line.strip() for line in f if line.strip()]

    # Execute workload directly
    db = create_database_connection(args.db_path)
    executor = WorkloadExecutor(db, config)
    results = executor.execute_workload(workload_queries)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'results': results
        }, f, indent=2)

    db.close()

    logging.info(f"Workload execution completed. Results saved to {args.output}")


def train_models_cmd(args):
    """Train baseline models."""
    logging.info("Training models...")

    # Load workload results
    with open(args.workload, 'r') as f:
        workload_results = json.load(f)

    if 'results' in workload_results:
        workload_data = workload_results['results']
    else:
        workload_data = workload_results

    models_to_train = [args.model_type] if args.model_type else get_available_models()

    for model_type in models_to_train:
        logging.info(f"Training {model_type} model...")
        try:
            results = train_model(
                model_type=model_type,
                workload_results=workload_data,
                device=args.device,
                epochs=args.epochs
            )
            logging.info(f"Training completed for {model_type}")
        except Exception as e:
            logging.error(f"Failed to train {model_type}: {e}")


def evaluate_models_cmd(args):
    """Evaluate models."""
    logging.info("Evaluating models...")

    # Load test workload
    with open(args.test_workload, 'r') as f:
        test_workload = json.load(f)

    if 'results' in test_workload:
        test_data = test_workload['results']
    else:
        test_data = test_workload

    # Evaluate model
    results = evaluate_model(
        args.model_path,
        args.model_type,
        test_data,
        device=args.device
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Evaluation completed. Results saved to {args.output}")


def run_pipeline(args):
    """Run complete pipeline."""
    logging.info("Running complete pipeline...")

    # 1. Setup dataset
    setup_dataset(args)

    # 2. Generate workload
    generate_workload_cmd(args)

    # 3. Run workload (simplified)
    logging.info("Pipeline step 3: Run workload - skipped for demo")

    # 4. Train models
    if args.train_all:
        for model_type in get_available_models():
            args.model_type = model_type
            train_models_cmd(args)
    else:
        args.model_type = 'flat_vector'
        train_models_cmd(args)

    logging.info("Pipeline completed")


def analyze_workload_cmd(args):
    """Analyze workload results."""
    logging.info("Analyzing workload results...")

    # Load workload results
    with open(args.workload_results, 'r') as f:
        workload_results = json.load(f)

    if 'results' in workload_results:
        results_data = workload_results['results']
    else:
        results_data = workload_results

    # Analyze
    analyze_workload(results_data, args.output_dir)

    logging.info(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    sys.exit(main())