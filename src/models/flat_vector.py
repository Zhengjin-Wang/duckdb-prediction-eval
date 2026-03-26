"""
Flat Vector baseline model for cost estimation.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib


class FlatVectorModel:
    """Baseline flat vector model using traditional ML algorithms."""

    def __init__(self, model_type: str = 'random_forest', **model_params):
        """
        Initialize Flat Vector Model.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
            **model_params: Parameters for the underlying model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**model_params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**model_params)
        elif model_type == 'linear':
            self.model = LinearRegression(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def extract_features(self, query_plan: Dict) -> np.ndarray:
        """
        Extract flat features from query plan.

        Args:
            query_plan: Query execution plan with statistics

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Basic query statistics
        features.append(query_plan.get('avg_time_ms', 0))
        features.append(query_plan.get('min_time_ms', 0))
        features.append(query_plan.get('max_time_ms', 0))
        features.append(query_plan.get('successful_executions', 0))

        # Query complexity features
        query = query_plan.get('query', '')
        features.append(query.upper().count('JOIN'))
        features.append(query.upper().count('WHERE'))
        features.append(query.upper().count('GROUP BY'))
        features.append(query.upper().count('ORDER BY'))
        features.append(len(query.split()))

        # Table usage features
        tables = self._extract_tables_from_query(query)
        features.append(len(tables))

        # Add table-specific features
        table_features = self._extract_table_features(tables)
        features.extend(table_features)

        return np.array(features)

    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        query_upper = query.upper()

        # Simple regex to extract table names after FROM and JOIN
        import re

        # Extract FROM tables
        from_matches = re.findall(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_upper)
        tables.extend(from_matches)

        # Extract JOIN tables
        join_matches = re.findall(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_upper)
        tables.extend(join_matches)

        return list(set(tables))  # Remove duplicates

    def _extract_table_features(self, tables: List[str]) -> List[float]:
        """Extract table-specific features."""
        features = []

        # For each table, add some basic features
        # In a real implementation, these would come from database statistics
        for table in ['title', 'cast_info', 'company_name', 'movie_companies']:
            features.append(1.0 if table in tables else 0.0)

        return features

    def prepare_features(self, workload_results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from workload results.

        Args:
            workload_results: List of query execution results

        Returns:
            X: Feature matrix
            y: Target vector (execution times)
        """
        valid_results = [r for r in workload_results if r.get('successful_executions', 0) > 0]

        if not valid_results:
            raise ValueError("No valid query results found")

        X = []
        y = []

        for result in valid_results:
            # Extract features
            features = self.extract_features(result)
            X.append(features)

            # Use average execution time as target
            y.append(result['avg_time_ms'])

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train(self, workload_results: List[Dict], test_size: float = 0.2, random_state: int = 42):
        """
        Train the model on workload results.

        Args:
            workload_results: Training data
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        print(f"Training {self.model_type} model...")

        # Prepare data
        X, y = self.prepare_features(workload_results)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100

        print(f"Validation RMSE: {rmse:.2f}ms")
        print(f"Validation MAPE: {mape:.2f}%")

        self.is_trained = True

        return {
            'rmse': rmse,
            'mape': mape,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }

    def predict(self, workload_results: List[Dict]) -> List[float]:
        """
        Make predictions on workload results.

        Args:
            workload_results: Query execution results

        Returns:
            List of predicted execution times
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = []

        for result in workload_results:
            # Extract features
            features = self.extract_features(result)
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            predictions.append(max(0, prediction))  # Ensure non-negative predictions

        return predictions

    def evaluate(self, workload_results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            workload_results: Test workload results

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Get predictions
        predictions = self.predict(workload_results)
        true_values = [r['avg_time_ms'] for r in workload_results if r.get('successful_executions', 0) > 0]

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mape = mean_absolute_percentage_error(true_values, predictions) * 100

        # Calculate Q-Error (95th percentile)
        errors = np.abs(np.array(predictions) - np.array(true_values)) / np.array(true_values)
        q_error_95 = np.percentile(errors, 95)

        return {
            'rmse': rmse,
            'mape': mape,
            'q_error_95': q_error_95,
            'predictions': predictions,
            'true_values': true_values
        }

    def save(self, model_path: str):
        """Save model to disk."""
        model_data = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str):
        """Load model from disk."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(
            model_type=model_data['model_type'],
            **model_data['model_params']
        )
        model.model = model_data['model']
        model.scaler = model_data['scaler']
        model.is_trained = model_data['is_trained']

        return model


def create_flat_vector_model(model_type: str = 'random_forest', **kwargs) -> FlatVectorModel:
    """Factory function to create flat vector model."""
    return FlatVectorModel(model_type=model_type, **kwargs)