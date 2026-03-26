#!/usr/bin/env python3
"""
Evaluate models script.
"""
import argparse
import json
from pathlib import Path
import logging

from src.models.registry import evaluate_model, load_model, get_available_models
from src.utils.metrics import ModelEvaluator
from src.utils.config import load_default_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_baseline_model(
    model_type: str,
    model_path: str,
    test_workload: list,
    device: str = 'cpu'
):
    """
    Evaluate a baseline model.

    Args:
        model_type: Type of model
        model_path: Path to saved model
        test_workload: Test workload results
        device: Device for evaluation

    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {model_type} model from {model_path}")

    try:
        # Load model
        model = load_model(model_path, model_type, device=device)

        # Evaluate model
        results = model.evaluate(test_workload)

        logger.info(f"Evaluation completed for {model_type}")
        return results

    except Exception as e:
        logger.error(f"Error evaluating {model_type}: {e}")
        raise


def compare_models(
    model_results: dict,
    baseline_model: str = None
):
    """
    Compare multiple model results.

    Args:
        model_results: Dictionary of model evaluation results
        baseline_model: Model to use as baseline for comparison

    Returns:
        Comparison results
    """
    if not model_results:
        return {}

    if baseline_model is None:
        baseline_model = list(model_results.keys())[0]

    logger.info(f"Comparing models with {baseline_model} as baseline")

    comparison = {}
    baseline_metrics = model_results[baseline_model]

    for model_name, metrics in model_results.items():
        if model_name == baseline_model:
            continue

        model_comparison = {}
        for metric in baseline_metrics:
            if metric in metrics:
                baseline_val = baseline_metrics[metric]
                model_val = metrics[metric]

                if metric in ["rmse", "mape", "mae", "max_error"]:
                    # Lower is better
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                elif metric == "r_squared":
                    # Higher is better
                    if baseline_val != 0:
                        improvement = ((model_val - baseline_val) / baseline_val) * 100
                    else:
                        improvement = 0
                else:
                    improvement = 0

                model_comparison[metric] = {
                    "baseline": baseline_val,
                    "model": model_val,
                    "improvement": improvement
                }

        comparison[model_name] = model_comparison

    return comparison


def load_workload_results(workload_file: str) -> list:
    """Load workload results from file."""
    workload_file = Path(workload_file)

    if not workload_file.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_file}")

    with open(workload_file, 'r') as f:
        data = json.load(f)

    if 'results' in data:
        return data['results']
    else:
        return data  # Assume the file contains results directly


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--model_type', choices=get_available_models(),
                       help='Type of model to evaluate')
    parser.add_argument('--model_path', required=True,
                       help='Path to saved model file')
    parser.add_argument('--test_workload', required=True,
                       help='Path to test workload results')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device for evaluation')
    parser.add_argument('--compare', nargs='+',
                       help='Models to compare (provide model paths)')
    parser.add_argument('--baseline', default='flat_vector',
                       help='Baseline model for comparison')
    parser.add_argument('--output', default='output/evaluation_results.json',
                       help='Output file for evaluation results')

    args = parser.parse_args()

    # Load test workload
    logger.info(f"Loading test workload from {args.test_workload}")
    test_workload = load_workload_results(args.test_workload)

    if not test_workload:
        logger.error("No test workload found")
        return

    logger.info(f"Loaded {len(test_workload)} test queries")

    evaluation_results = {}

    # Evaluate single model
    if args.model_type:
        results = evaluate_baseline_model(
            model_type=args.model_type,
            model_path=args.model_path,
            test_workload=test_workload,
            device=args.device
        )
        evaluation_results[args.model_type] = results

    # Compare multiple models
    elif args.compare:
        for model_path in args.compare:
            # Extract model type from filename or path
            model_type = Path(model_path).stem.replace('_model', '')
            if model_type not in get_available_models():
                logger.warning(f"Unknown model type for {model_path}, skipping")
                continue

            results = evaluate_baseline_model(
                model_type=model_type,
                model_path=model_path,
                test_workload=test_workload,
                device=args.device
            )
            evaluation_results[model_type] = results

        # Compare models
        if len(evaluation_results) > 1:
            comparison = compare_models(evaluation_results, args.baseline)

            # Print comparison summary
            print("\n=== Model Comparison ===")
            for model_name, metrics in comparison.items():
                print(f"\n{model_name}:")
                for metric, values in metrics.items():
                    if 'improvement' in values:
                        print(f"  {metric}: {values['improvement']:+.2f}% vs {args.baseline}")

    # Print evaluation results
    print("\n=== Evaluation Results ===")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name}:")
        for metric, value in results.items():
            if metric != 'predictions' and metric != 'true_values':  # Skip large arrays
                print(f"  {metric}: {value}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)

    logger.info(f"Evaluation completed. Results saved to {output_path}")


if __name__ == '__main__':
    main()