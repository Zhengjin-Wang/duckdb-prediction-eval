#!/usr/bin/env python3
"""
Generate SQL workload script.
"""
import argparse
import json
from pathlib import Path
import logging

from src.datasets.imdb_dataset import create_imdb_dataset
from src.workloads.generator import WorkloadGenerator, WorkloadConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_workload(
    num_queries: int = 1000,
    max_joins: int = 3,
    max_predicates: int = 3,
    seed: int = 42,
    output_file: str = "workloads/imdb_workload.sql"
):
    """
    Generate SQL workload for IMDB dataset.

    Args:
        num_queries: Number of queries to generate
        max_joins: Maximum number of joins per query
        max_predicates: Maximum number of predicates per query
        seed: Random seed for reproducibility
        output_file: Output file path
    """
    logger.info("Initializing workload generator...")

    # Create dataset and generator
    dataset = create_imdb_dataset()
    config = WorkloadConfig(
        num_queries=num_queries,
        max_joins=max_joins,
        max_predicates=max_predicates,
        seed=seed
    )
    generator = WorkloadGenerator(dataset, config)

    # Generate workload
    logger.info(f"Generating {num_queries} queries...")
    queries = generator.generate_workload()

    # Save workload
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for query in queries:
            f.write(query + '\n')

    logger.info(f"Generated workload saved to {output_path}")
    logger.info(f"Sample queries:")
    for i, query in enumerate(queries[:3]):
        logger.info(f"  {i+1}: {query[:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Generate SQL workload for IMDB dataset')
    parser.add_argument('--num_queries', type=int, default=1000,
                       help='Number of queries to generate')
    parser.add_argument('--max_joins', type=int, default=3,
                       help='Maximum number of joins per query')
    parser.add_argument('--max_predicates', type=int, default=3,
                       help='Maximum number of predicates per query')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', default='workloads/imdb_workload.sql',
                       help='Output file path')

    args = parser.parse_args()

    generate_workload(
        num_queries=args.num_queries,
        max_joins=args.max_joins,
        max_predicates=args.max_predicates,
        seed=args.seed,
        output_file=args.output
    )


if __name__ == '__main__':
    main()