"""
Data preprocessing and feature engineering utilities.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json


class DataPreprocessor:
    """Preprocess workload data for model training."""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}

    def extract_query_features(self, workload_results: List[Dict]) -> pd.DataFrame:
        """
        Extract features from workload results.

        Returns:
            DataFrame with extracted features
        """
        features_list = []

        for result in workload_results:
            features = self._extract_single_query_features(result)
            features_list.append(features)

        return pd.DataFrame(features_list)

    def _extract_single_query_features(self, result: Dict) -> Dict[str, Any]:
        """Extract features from a single query result."""
        features = {}

        # Execution statistics
        features['execution_time'] = result.get('avg_time_ms', 0)
        features['min_time'] = result.get('min_time_ms', 0)
        features['max_time'] = result.get('max_time_ms', 0)
        features['std_time'] = result.get('std_time_ms', 0)
        features['successful_executions'] = result.get('successful_executions', 0)

        # Query complexity features
        query = result.get('query', '').upper()
        features['join_count'] = query.count('JOIN')
        features['where_count'] = query.count('WHERE')
        features['group_by_count'] = query.count('GROUP BY')
        features['order_by_count'] = query.count('ORDER BY')
        features['select_count'] = query.count('SELECT')
        features['union_count'] = query.count('UNION')
        features['distinct_count'] = query.count('DISTINCT')

        # Token count
        features['token_count'] = len(query.split())

        # Table features
        tables = self._extract_tables(result.get('query', ''))
        features['table_count'] = len(tables)
        features['unique_table_count'] = len(set(tables))

        # Table presence features (one-hot for common tables)
        common_tables = ['title', 'cast_info', 'company_name', 'movie_companies',
                        'movie_info', 'movie_info_idx', 'movie_keyword', 'name',
                        'person_info', 'keyword', 'info_type', 'kind_type']

        for table in common_tables:
            features[f'table_{table}'] = 1 if table in tables else 0

        # Join type features
        join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN']
        for join_type in join_types:
            features[f'join_{join_type.lower().replace(" ", "_")}'] = query.count(join_type)

        # Predicate features
        operators = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'BETWEEN', 'IS NULL', 'IS NOT NULL']
        for op in operators:
            features[f'op_{op.lower().replace(" ", "_").replace("!", "not_")}'] = query.count(op)

        # Function features
        functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CASE', 'CAST', 'COALESCE']
        for func in functions:
            features[f'func_{func.lower()}'] = query.count(func)

        # Plan features if available
        plan = result.get('plan', '')
        if plan:
            features['plan_seq_scan'] = plan.count('Seq Scan')
            features['plan_index_scan'] = plan.count('Index Scan')
            features['plan_hash_join'] = plan.count('Hash Join')
            features['plan_merge_join'] = plan.count('Merge Join')
            features['plan_nested_loop'] = plan.count('Nested Loop')
        else:
            features['plan_seq_scan'] = 0
            features['plan_index_scan'] = 0
            features['plan_hash_join'] = 0
            features['plan_merge_join'] = 0
            features['plan_nested_loop'] = 0

        return features

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

    def prepare_training_data(self, workload_results: List[Dict],
                            target_column: str = 'execution_time',
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from workload results.

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Extract features
        df_features = self.extract_query_features(workload_results)

        # Separate features and target
        if target_column not in df_features.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")

        X = df_features.drop(columns=[target_column])
        y = df_features[target_column]

        # Handle missing values
        X = X.fillna(0)

        # Encode categorical features
        X_encoded = self._encode_categorical_features(X)

        # Scale features
        X_scaled = self._scale_features(X_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train.values, y_test.values

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        X_encoded = X.copy()

        # Identify categorical columns (object type)
        categorical_cols = X.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories
                known_categories = set(self.encoders[col].classes_)
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: self.encoders[col].transform([x])[0] if x in known_categories else -1
                )

        return X_encoded

    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale numerical features."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) > 0:
            X_numerical = X[numerical_cols]

            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                X_scaled = self.scalers['standard'].fit_transform(X_numerical)
            else:
                X_scaled = self.scalers['standard'].transform(X_numerical)

            return X_scaled
        else:
            return X.values

    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            return dict(zip(self.get_feature_names(), model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            return dict(zip(self.get_feature_names(), np.abs(model.coef_)))
        else:
            return {}

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        # This would need to be implemented based on the actual feature extraction
        # For now, return a placeholder
        return []

    def save_preprocessor(self, file_path: str):
        """Save preprocessor state."""
        preprocessor_state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats
        }

        with open(file_path, 'w') as f:
            json.dump(preprocessor_state, f, indent=2, default=str)

    def load_preprocessor(self, file_path: str):
        """Load preprocessor state."""
        with open(file_path, 'r') as f:
            preprocessor_state = json.load(f)

        self.scalers = preprocessor_state.get('scalers', {})
        self.encoders = preprocessor_state.get('encoders', {})
        self.feature_stats = preprocessor_state.get('feature_stats', {})


class FeatureEngineer:
    """Advanced feature engineering for query plans."""

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features."""
        df_new = df.copy()

        # Create some common interactions
        if 'join_count' in df.columns and 'table_count' in df.columns:
            df_new['join_table_interaction'] = df['join_count'] * df['table_count']

        if 'where_count' in df.columns and 'token_count' in df.columns:
            df_new['predicate_complexity'] = df['where_count'] / (df['token_count'] + 1)

        if 'execution_time' in df.columns:
            # Log transformation for skewed features
            df_new['log_execution_time'] = np.log1p(df['execution_time'])

        return df_new

    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[columns])

        # Create feature names
        feature_names = poly.get_feature_names_out(columns)

        # Create DataFrame with polynomial features
        df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        # Concatenate with original DataFrame
        return pd.concat([df, df_poly], axis=1)

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]