"""
Query Plan Prediction Network (QPPNet) baseline model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
import json


class QPPNet(nn.Module):
    """Query Plan Prediction Network for cost estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super(QPPNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )

        # Tree-like aggregation layer
        self.tree_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Tree-like aggregation (simplified)
        aggregated = self.tree_aggregator(features)

        # Output prediction
        output = self.output_layer(aggregated)

        return output.squeeze()


class QPPNetModel:
    """QPPNet Model wrapper with training and evaluation utilities."""

    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.model = QPPNet(input_dim).to(device)
        self.device = device
        self.scaler = None
        self.is_trained = False

        # Training parameters
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def extract_features(self, query_plan: Dict) -> np.ndarray:
        """
        Extract QPPNet-style features from query plan.

        Args:
            query_plan: Query execution plan with statistics

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Query execution statistics
        features.append(query_plan.get('avg_time_ms', 0))
        features.append(query_plan.get('min_time_ms', 0))
        features.append(query_plan.get('max_time_ms', 0))
        features.append(query_plan.get('std_time_ms', 0))

        # Query complexity metrics
        query = query_plan.get('query', '')
        query_upper = query.upper()

        # Basic SQL structure features
        features.append(query_upper.count('SELECT'))
        features.append(query_upper.count('FROM'))
        features.append(query_upper.count('JOIN'))
        features.append(query_upper.count('WHERE'))
        features.append(query_upper.count('GROUP BY'))
        features.append(query_upper.count('ORDER BY'))
        features.append(query_upper.count('HAVING'))

        # Query complexity indicators
        features.append(len(query.split()))  # Total tokens
        features.append(query_upper.count('UNION'))
        features.append(query_upper.count('INTERSECT'))
        features.append(query_upper.count('EXCEPT'))

        # Table usage patterns
        tables = self._extract_tables(query)
        features.append(len(tables))
        features.append(len(set(tables)))  # Unique tables

        # Join patterns
        join_types = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
        for join_type in join_types:
            features.append(query_upper.count(join_type))

        # Predicate patterns
        operators = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'BETWEEN', 'IS NULL', 'IS NOT NULL']
        for op in operators:
            features.append(query_upper.count(op))

        # Function usage
        functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'CASE', 'CAST']
        for func in functions:
            features.append(query_upper.count(func))

        # Aggregate features
        features.append(query_upper.count('GROUP BY') > 0)
        features.append(query_upper.count('HAVING') > 0)
        features.append(query_upper.count('ORDER BY') > 0)

        # Plan-specific features
        if 'plan' in query_plan:
            plan_text = str(query_plan['plan'])
            features.append(plan_text.count('Seq Scan'))
            features.append(plan_text.count('Index Scan'))
            features.append(plan_text.count('Hash Join'))
            features.append(plan_text.count('Merge Join'))
            features.append(plan_text.count('Nested Loop'))

        return np.array(features, dtype=np.float32)

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        import re
        tables = []
        query_upper = query.upper()

        # Extract FROM tables
        from_matches = re.findall(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_upper)
        tables.extend(from_matches)

        # Extract JOIN tables
        join_matches = re.findall(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_upper)
        tables.extend(join_matches)

        return list(set(tables))

    def prepare_data(self, workload_results: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        valid_results = [r for r in workload_results if r.get('successful_executions', 0) > 0]

        features_list = []
        labels_list = []

        for result in valid_results:
            features = self.extract_features(result)
            features_list.append(features)
            labels_list.append(result['avg_time_ms'])

        # Convert to numpy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_array)

        # Convert to tensors
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        labels_tensor = torch.FloatTensor(labels_array).to(self.device)

        return features_tensor, labels_tensor

    def train(self, workload_results: List[Dict], epochs: int = 200, batch_size: int = 32, val_split: float = 0.2):
        """Train the QPPNet model."""
        print(f"Training QPPNet model for {epochs} epochs...")

        # Prepare data
        features, labels = self.prepare_data(workload_results)

        # Split data
        n_samples = features.shape[0]
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        train_features = features[:n_train]
        train_labels = labels[:n_train]
        val_features = features[n_train:]
        val_labels = labels[n_train:]

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            # Shuffle training data
            indices = torch.randperm(n_train)
            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i:i+batch_size]

                # Get batch
                batch_features = train_features[batch_indices]
                batch_labels = train_labels[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_features)
                val_loss = self.criterion(val_outputs, val_labels).item()

            # Update learning rate
            self.scheduler.step(val_loss)

            avg_loss = epoch_loss / (n_train // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'temp_best_qppnet.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                self.model.load_state_dict(torch.load('temp_best_qppnet.pth'))
                import os
                os.remove('temp_best_qppnet.pth')
                break

        self.is_trained = True
        return best_loss

    def predict(self, workload_results: List[Dict]) -> List[float]:
        """Make predictions on workload results."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = []
        self.model.eval()

        with torch.no_grad():
            for result in workload_results:
                features = self.extract_features(result)
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)

                output = self.model(features_tensor)
                predictions.append(max(0, output.item()))

        return predictions

    def evaluate(self, workload_results: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

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
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'device': self.device,
            'is_trained': self.is_trained,
            'scaler': self.scaler
        }

        import pickle
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"QPPNet model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu'):
        """Load model from disk."""
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(model_data['input_dim'], device=device)
        model.model.load_state_dict(model_data['model_state_dict'])
        model.is_trained = model_data['is_trained']
        model.scaler = model_data['scaler']

        return model


def create_qppnet_model(device: str = 'cpu') -> QPPNetModel:
    """Factory function to create QPPNet model."""
    # Estimate input dimension based on feature extraction
    sample_features = np.zeros(50)  # Conservative estimate
    return QPPNetModel(len(sample_features), device=device)