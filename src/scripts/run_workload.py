#!/usr/bin/env python3
"""
Run workload on DuckDB script.
"""
import argparse
import json
import time
from pathlib import Path
import logging

from src.database.duckdb_connection import create_database_connection
from src.database.loader import load_imdb_to_duckdb
from src.workloads.generator import WorkloadExecutor, WorkloadConfig
from src.workloads.analyzer import analyze_workload

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset_to_duckdb(data_dir: str, db_path: str, force: bool = False):
    """Load IMDB dataset to DuckDB."""
    logger.info(f"Loading dataset from {data_dir} to {db_path}")
    load_imdb_to_duckdb(db_path, data_dir, force=force)
    logger.info("Dataset loaded successfully")


def run_workload(
    db_path: str,
    workload_file: str,
    config: WorkloadConfig,
    output_file: str = "data/runs/workload_results.json"
):
    """
    Run workload on DuckDB database.

    Args:
        db_path: Path to DuckDB database
        workload_file: Path to workload file
        config: Workload configuration
        output_file: Output file for results
    """
    logger.info(f"Running workload from {workload_file}")

    # Load workload
    workload_file = Path(workload_file)
    if not workload_file.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_file}")

    with open(workload_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(queries)} queries")

    # Initialize database connection
    db = create_database_connection(db_path)

    # Initialize executor
    executor = WorkloadExecutor(db, config)

    # Execute workload
    start_time = time.time()
    results = executor.execute_workload(queries)
    execution_time = time.time() - start_time

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workload_results = {
        'config': config.__dict__,
        'execution_time': execution_time,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(workload_results, f, indent=2)

    logger.info(f"Workload execution completed in {execution_time:.2f} seconds")
    logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run workload on DuckDB')
    parser.add_argument('--db_path', default='data/imdb.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--data_dir', default='data/datasets/imdb',
                       help='Path to dataset directory')
    parser.add_argument('--workload', required=True,
                       help='Path to workload file')
    parser.add_argument('--load_data', action='store_true',
                       help='Load dataset to DuckDB before running workload')
    parser.add_argument('--force_load', action='store_true',
                       help='Force reload dataset even if database exists')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Number of queries to execute (0 for all)')
    parser.add_argument('--max_joins', type=int, default=3,
                       help='Maximum number of joins')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', default='data/runs/workload_results.json',
                       help='Output file for results')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze workload results')

    args = parser.parse_args()

    # Load dataset if requested
    if args.load_data:
        load_dataset_to_duckdb(args.data_dir, args.db_path, args.force_load)

    # Configure workload
    config = WorkloadConfig(
        num_queries=args.num_queries if args.num_queries > 0 else 1000000,
        max_joins=args.max_joins,
        seed=args.seed
    )

    # Run workload
    results = run_workload(args.db_path, args.workload, config, args.output)

    # Analyze results if requested
    if args.analyze and results:
        analyze_workload(results, output_dir="output/")

    logger.info("Workload execution completed")


if __name__ == '__main__':
    main()