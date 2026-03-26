"""
Multi-Set Convolutional Network (MSCN) baseline model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import json


class MSCN(nn.Module):
    """Multi-Set Convolutional Network for query cost estimation."""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256, output_dim: int = 1):
        super(MSCN, self).__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        # Feature extraction layers for different components
        self.table_layers = nn.Sequential(
            nn.Linear(input_dims['table'], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.join_layers = nn.Sequential(
            nn.Linear(input_dims['join'], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.filter_layers = nn.Sequential(
            nn.Linear(input_dims['filter'], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, table_features, join_features, filter_features):
        # Extract features from each component
        table_out = self.table_layers(table_features)
        join_out = self.join_layers(join_features)
        filter_out = self.filter_layers(filter_features)

        # Aggregate features
        combined = torch.cat([table_out, join_out, filter_out], dim=1)
        aggregated = self.aggregation(combined)

        # Output prediction
        output = self.output_layer(aggregated)

        return output.squeeze()


class MSCNModel:
    """MSCN Model wrapper with training and evaluation utilities."""

    def __init__(self, input_dims: Dict[str, int], device: str = 'cpu'):
        self.model = MSCN(input_dims).to(device)
        self.device = device
        self.scaler_table = StandardScaler()
        self.scaler_join = StandardScaler()
        self.scaler_filter = StandardScaler()
        self.is_trained = False

        # Training parameters
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def extract_features(self, query_plan: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract MSCN-style features from query plan.

        Returns:
            table_features, join_features, filter_features
        """
        query = query_plan.get('query', '')
        query_upper = query.upper()

        # Table features
        tables = self._extract_tables(query)
        table_features = np.zeros(self.input_dims['table'])
        for i, table in enumerate(tables[:self.input_dims['table']]):
            table_features[i] = 1.0

        # Join features
        join_count = query_upper.count('JOIN')
        join_features = np.array([join_count, join_count > 0, join_count > 1, join_count > 2])

        # Filter features
        where_count = query_upper.count('WHERE')
        filter_features = np.array([
            where_count,
            where_count > 0,
            where_count > 1,
            where_count > 2,
            'LIKE' in query_upper,
            '>' in query_upper,
            '<' in query_upper,
            'IS NOT NULL' in query_upper
        ])

        return table_features, join_features, filter_features

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

    def prepare_data(self, workload_results: List[Dict]) -> Tuple[Dict, np.ndarray]:
        """Prepare data for training."""
        valid_results = [r for r in workload_results if r.get('successful_executions', 0) > 0]

        table_features = []
        join_features = []
        filter_features = []
        labels = []

        for result in valid_results:
            t_feat, j_feat, f_feat = self.extract_features(result)
            table_features.append(t_feat)
            join_features.append(j_feat)
            filter_features.append(f_feat)
            labels.append(result['avg_time_ms'])

        # Convert to numpy arrays
        table_features = np.array(table_features)
        join_features = np.array(join_features)
        filter_features = np.array(filter_features)
        labels = np.array(labels)

        # Scale features
        table_features = self.scaler_table.fit_transform(table_features)
        join_features = self.scaler_join.fit_transform(join_features)
        filter_features = self.scaler_filter.fit_transform(filter_features)

        return {
            'table': torch.FloatTensor(table_features).to(self.device),
            'join': torch.FloatTensor(join_features).to(self.device),
            'filter': torch.FloatTensor(filter_features).to(self.device)
        }, torch.FloatTensor(labels).to(self.device)

    def train(self, workload_results: List[Dict], epochs: int = 100, batch_size: int = 32, val_split: float = 0.2):
        """Train the MSCN model."""
        print(f"Training MSCN model for {epochs} epochs...")

        # Prepare data
        features, labels = self.prepare_data(workload_results)

        # Split data
        n_samples = features['table'].shape[0]
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        # Simple train/val split
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_samples))

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Shuffle indices
            indices = np.random.permutation(train_indices)
            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]

                # Get batch
                batch_features = {
                    'table': features['table'][batch_indices],
                    'join': features['join'][batch_indices],
                    'filter': features['filter'][batch_indices]
                }
                batch_labels = labels[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    batch_features['table'],
                    batch_features['join'],
                    batch_features['filter']
                )
                loss = self.criterion(outputs, batch_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(
                    features['table'][val_indices],
                    features['join'][val_indices],
                    features['filter'][val_indices]
                )
                val_loss = self.criterion(val_outputs, labels[val_indices]).item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
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
                t_feat, j_feat, f_feat = self.extract_features(result)

                # Scale features
                t_feat = self.scaler_table.transform(t_feat.reshape(1, -1))
                j_feat = self.scaler_join.transform(j_feat.reshape(1, -1))
                f_feat = self.scaler_filter.transform(f_feat.reshape(1, -1))

                # Convert to tensors
                t_tensor = torch.FloatTensor(t_feat).to(self.device)
                j_tensor = torch.FloatTensor(j_feat).to(self.device)
                f_tensor = torch.FloatTensor(f_feat).to(self.device)

                # Predict
                output = self.model(t_tensor, j_tensor, f_tensor)
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
            'input_dims': self.model.input_dims,
            'device': self.device,
            'is_trained': self.is_trained,
            'scaler_table': self.scaler_table,
            'scaler_join': self.scaler_join,
            'scaler_filter': self.scaler_filter
        }

        import pickle
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"MSCN model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, device: str = 'cpu'):
        """Load model from disk."""
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(model_data['input_dims'], device=device)
        model.model.load_state_dict(model_data['model_state_dict'])
        model.is_trained = model_data['is_trained']
        model.scaler_table = model_data['scaler_table']
        model.scaler_join = model_data['scaler_join']
        model.scaler_filter = model_data['scaler_filter']

        return model


def create_mscn_model(device: str = 'cpu') -> MSCNModel:
    """Factory function to create MSCN model."""
    input_dims = {
        'table': 10,  # Max 10 tables
        'join': 4,    # Join features
        'filter': 8   # Filter features
    }
    return MSCNModel(input_dims, device=device)