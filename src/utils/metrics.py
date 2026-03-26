"""
Evaluation metrics and utilities.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class MetricsCalculator:
    """Calculate various evaluation metrics for cost estimation models."""

    @staticmethod
    def rmse(y_true: List[float], y_pred: List[float]) -> float:
        """Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Percentage Error."""
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    @staticmethod
    def q_error(y_true: List[float], y_pred: List[float], percentile: int = 95) -> float:
        """Q-Error at given percentile."""
        errors = np.abs(np.array(y_pred) - np.array(y_true)) / np.array(y_true)
        return np.percentile(errors, percentile)

    @staticmethod
    def max_error(y_true: List[float], y_pred: List[float]) -> float:
        """Maximum absolute error."""
        return np.max(np.abs(np.array(y_true) - np.array(y_pred)))

    @staticmethod
    def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    @staticmethod
    def r_squared(y_true: List[float], y_pred: List[float]) -> float:
        """Coefficient of determination (R²)."""
        correlation_matrix = np.corrcoef(y_true, y_pred)
        correlation_xy = correlation_matrix[0, 1]
        return correlation_xy ** 2

    @staticmethod
    def relative_error(y_true: List[float], y_pred: List[float]) -> List[float]:
        """Relative error for each prediction."""
        return np.abs(np.array(y_pred) - np.array(y_true)) / np.array(y_true)

    @staticmethod
    def error_distribution(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Error distribution statistics."""
        errors = MetricsCalculator.relative_error(y_true, y_pred)

        return {
            "mean": np.mean(errors),
            "median": np.median(errors),
            "std": np.std(errors),
            "min": np.min(errors),
            "max": np.max(errors),
            "percentile_25": np.percentile(errors, 25),
            "percentile_50": np.percentile(errors, 50),
            "percentile_75": np.percentile(errors, 75),
            "percentile_90": np.percentile(errors, 90),
            "percentile_95": np.percentile(errors, 95),
            "percentile_99": np.percentile(errors, 99)
        }


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""

    def __init__(self, y_true: List[float], y_pred: List[float]):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.metrics_calculator = MetricsCalculator()

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all available metrics."""
        return {
            "rmse": self.metrics_calculator.rmse(self.y_true, self.y_pred),
            "mape": self.metrics_calculator.mape(self.y_true, self.y_pred),
            "q_error_50": self.metrics_calculator.q_error(self.y_true, self.y_pred, 50),
            "q_error_95": self.metrics_calculator.q_error(self.y_true, self.y_pred, 95),
            "q_error_99": self.metrics_calculator.q_error(self.y_true, self.y_pred, 99),
            "max_error": self.metrics_calculator.max_error(self.y_true, self.y_pred),
            "mae": self.metrics_calculator.mean_absolute_error(self.y_true, self.y_pred),
            "r_squared": self.metrics_calculator.r_squared(self.y_true, self.y_pred),
            "error_distribution": self.metrics_calculator.error_distribution(self.y_true, self.y_pred)
        }

    def get_summary(self) -> Dict[str, float]:
        """Get summary of key metrics."""
        metrics = self.calculate_all_metrics()
        return {
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "q_error_95": metrics["q_error_95"],
            "r_squared": metrics["r_squared"]
        }

    def compare_models(self, other_evaluator: 'ModelEvaluator') -> Dict[str, Dict[str, float]]:
        """Compare this model with another evaluator."""
        self_metrics = self.calculate_all_metrics()
        other_metrics = other_evaluator.calculate_all_metrics()

        comparison = {}
        for metric in self_metrics:
            if metric not in other_metrics:
                continue

            self_val = self_metrics[metric]
            other_val = other_metrics[metric]

            if metric in ["rmse", "mape", "mae", "max_error"]:
                # Lower is better
                improvement = ((other_val - self_val) / other_val) * 100
            elif metric == "r_squared":
                # Higher is better
                improvement = ((self_val - other_val) / other_val) * 100
            else:
                improvement = 0

            comparison[metric] = {
                "self": self_val,
                "other": other_val,
                "improvement": improvement
            }

        return comparison

    def print_report(self, model_name: str = "Model"):
        """Print evaluation report."""
        metrics = self.get_summary()

        print(f"\n=== {model_name} Evaluation Report ===")
        print(f"RMSE:        {metrics['rmse']:.4f}")
        print(f"MAPE:        {metrics['mape']:.2f}%")
        print(f"Q-Error(95): {metrics['q_error_95']:.4f}")
        print(f"R²:          {metrics['r_squared']:.4f}")
        print("=" * 40)


def evaluate_workload_results(model, workload_results: List[Dict]) -> Dict[str, float]:
    """Evaluate model on workload results."""
    if hasattr(model, 'evaluate'):
        return model.evaluate(workload_results)
    else:
        # Fallback for models that don't have evaluate method
        predictions = model.predict(workload_results)
        true_values = [r['avg_time_ms'] for r in workload_results if r.get('successful_executions', 0) > 0]

        evaluator = ModelEvaluator(true_values, predictions)
        return evaluator.get_summary()


def compare_models_results(model_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compare multiple model results."""
    comparison = {}
    model_names = list(model_results.keys())

    if len(model_names) < 2:
        return comparison

    # Use first model as baseline
    baseline_name = model_names[0]
    baseline_metrics = model_results[baseline_name]

    for model_name in model_names[1:]:
        model_metrics = model_results[model_name]

        comparison[model_name] = {}
        for metric in baseline_metrics:
            if metric in model_metrics:
                baseline_val = baseline_metrics[metric]
                model_val = model_metrics[metric]

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

                comparison[model_name][metric] = {
                    "baseline": baseline_val,
                    "model": model_val,
                    "improvement": improvement
                }

    return comparison