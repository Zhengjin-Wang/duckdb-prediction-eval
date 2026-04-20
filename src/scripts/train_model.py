#!/usr/bin/env python3
"""
Train baseline models script.
"""
import argparse
import json
import time
from pathlib import Path
import logging

import torch
from models.registry import create_model, train_model, get_available_models
from utils.config import load_default_config
from utils.preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_baseline_model(
    model_type: str,
    workload_results: list,
    model_params: dict = None,
    epochs: int = 100,
    device: str = 'cpu',
    output_dir: str = 'models/'
):
    """
    Train a baseline model.

    Args:
        model_type: Type of model to train
        workload_results: Training data
        model_params: Model-specific parameters
        epochs: Number of training epochs
        device: Device for training ('cpu' or 'cuda')
        output_dir: Directory to save trained model

    Returns:
        Training results
    """
    logger.info(f"Training {model_type} model...")

    if model_params is None:
        model_params = {}

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Create and train model
        if model_type == 'mscn':
            model_params['device'] = device
            model_params['epochs'] = epochs
        elif model_type == 'qppnet':
            model_params['device'] = device
            model_params['epochs'] = epochs

        results = train_model(model_type, workload_results, **model_params)

        # Save model
        model_file = output_path / f"{model_type}_model.pkl"
        if hasattr(results, 'model'):
            results.model.save(str(model_file))
        else:
            logger.warning(f"Could not save {model_type} model - no save method available")

        logger.info(f"Training completed for {model_type}")
        return results

    except Exception as e:
        logger.error(f"Error training {model_type}: {e}")
        raise


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
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model_type', choices=get_available_models(),
                       help='Type of model to train')
    parser.add_argument('--workload', required=True,
                       help='Path to workload results file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device for training')
    parser.add_argument('--output_dir', default='models/',
                       help='Directory to save trained models')
    parser.add_argument('--model_params', type=json.loads, default='{}',
                       help='Model-specific parameters as JSON string')
    parser.add_argument('--all_models', action='store_true',
                       help='Train all available models')

    args = parser.parse_args()

    # Load workload results
    logger.info(f"Loading workload results from {args.workload}")
    workload_results = load_workload_results(args.workload)

    if not workload_results:
        logger.error("No workload results found")
        return

    logger.info(f"Loaded {len(workload_results)} query results")

    # Train models
    if args.all_models:
        models_to_train = get_available_models()
    else:
        models_to_train = [args.model_type]

    training_results = {}

    for model_type in models_to_train:
        try:
            logger.info(f"Starting training for {model_type}")

            start_time = time.time()
            results = train_baseline_model(
                model_type=model_type,
                workload_results=workload_results,
                model_params=args.model_params,
                epochs=args.epochs,
                device=args.device,
                output_dir=args.output_dir
            )
            training_time = time.time() - start_time

            training_results[model_type] = {
                'training_time': training_time,
                'results': results
            }

            logger.info(f"Completed training for {model_type} in {training_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            training_results[model_type] = {'error': str(e)}

    # Save training summary
    summary_file = Path(args.output_dir) / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)

    logger.info(f"Training completed. Summary saved to {summary_file}")


if __name__ == '__main__':
    main()