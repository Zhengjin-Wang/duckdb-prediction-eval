#!/usr/bin/env python3
"""
Test suite for DuckDB LCM Evaluation framework.

This module contains unit tests for the main components of the framework.
"""
import unittest
import tempfile
import os
from pathlib import Path
import json

# Import framework components to test
from src.database.duckdb_connection import create_database_connection, DatabaseConfig
from src.datasets.imdb_dataset import create_imdb_dataset
from src.workloads.generator import WorkloadGenerator, WorkloadConfig
from src.models.flat_vector import FlatVectorModel
from src.models.registry import get_available_models
from src.utils.config import Config, load_default_config
from src.utils.metrics import MetricsCalculator, ModelEvaluator


class TestDatabaseConnection(unittest.TestCase):
    """Test DuckDB connection functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")

    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_config(self):
        """Test database configuration."""
        config = DatabaseConfig(
            db_path=self.db_path,
            read_only=False,
            memory_limit="4GB",
            threads=2
        )

        self.assertEqual(config.db_path, self.db_path)
        self.assertEqual(config.memory_limit, "4GB")
        self.assertEqual(config.threads, 2)

    def test_connection_creation(self):
        """Test database connection creation."""
        config = DatabaseConfig(db_path=self.db_path)
        db = create_database_connection(config)

        self.assertIsNotNone(db)
        self.assertEqual(db.config.db_path, self.db_path)


class TestIMDBDataset(unittest.TestCase):
    """Test IMDB dataset functionality."""

    def test_dataset_creation(self):
        """Test IMDB dataset creation."""
        dataset = create_imdb_dataset()

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.schema.tables[0], "title")
        self.assertEqual(len(dataset.schema.tables), 15)

    def test_schema_relationships(self):
        """Test schema relationships."""
        dataset = create_imdb_dataset()
        relationships = dataset.schema.relationships

        self.assertGreater(len(relationships), 0)
        # Check that relationships have correct format
        for rel in relationships:
            self.assertEqual(len(rel), 4)
            self.assertEqual(rel[1], rel[3])  # Foreign key names should match


class TestWorkloadGenerator(unittest.TestCase):
    """Test workload generation functionality."""

    def test_workload_config(self):
        """Test workload configuration."""
        config = WorkloadConfig(
            num_queries=100,
            max_joins=3,
            max_predicates=2,
            seed=42
        )

        self.assertEqual(config.num_queries, 100)
        self.assertEqual(config.max_joins, 3)
        self.assertEqual(config.max_predicates, 2)
        self.assertEqual(config.seed, 42)

    def test_workload_generator_creation(self):
        """Test workload generator creation."""
        dataset = create_imdb_dataset()
        config = WorkloadConfig(num_queries=10, max_joins=1, seed=42)
        generator = WorkloadGenerator(dataset, config)

        self.assertIsNotNone(generator)
        self.assertEqual(generator.config.num_queries, 10)
        self.assertEqual(generator.config.max_joins, 1)


class TestModels(unittest.TestCase):
    """Test model functionality."""

    def test_available_models(self):
        """Test available models."""
        models = get_available_models()
        self.assertIn('flat_vector', models)
        self.assertIn('mscn', models)
        self.assertIn('qppnet', models)

    def test_flat_vector_model_creation(self):
        """Test Flat Vector model creation."""
        model = FlatVectorModel(model_type='random_forest')

        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, 'random_forest')
        self.assertFalse(model.is_trained)

    def test_flat_vector_model_features(self):
        """Test Flat Vector model feature extraction."""
        model = FlatVectorModel(model_type='random_forest')

        # Create mock query plan
        query_plan = {
            'query': 'SELECT COUNT(*) FROM title WHERE id = 1',
            'avg_time_ms': 10.0,
            'successful_executions': 3
        }

        features = model.extract_features(query_plan)

        self.assertIsNotNone(features)
        self.assertIsInstance(features, list)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_config_creation(self):
        """Test configuration creation."""
        config = Config()

        # Test setting values
        config.set('database.path', 'test.db')
        config.set('database.memory_limit', '8GB')

        # Test getting values
        self.assertEqual(config.get('database.path'), 'test.db')
        self.assertEqual(config.get('database.memory_limit'), '8GB')
        self.assertEqual(config.get('nonexistent.key', 'default'), 'default')

    def test_metrics_calculator(self):
        """Test metrics calculator."""
        calc = MetricsCalculator()

        # Create test data
        y_true = [10, 20, 30, 40, 50]
        y_pred = [12, 18, 32, 38, 52]

        # Test RMSE
        rmse = calc.rmse(y_true, y_pred)
        self.assertIsInstance(rmse, float)
        self.assertGreater(rmse, 0)

        # Test MAPE
        mape = calc.mape(y_true, y_pred)
        self.assertIsInstance(mape, float)
        self.assertGreater(mape, 0)

        # Test Q-Error
        q_error = calc.q_error(y_true, y_pred, percentile=95)
        self.assertIsInstance(q_error, float)
        self.assertGreater(q_error, 0)

    def test_model_evaluator(self):
        """Test model evaluator."""
        y_true = [10, 20, 30, 40, 50]
        y_pred = [12, 18, 32, 38, 52]

        evaluator = ModelEvaluator(y_true, y_pred)
        summary = evaluator.get_summary()

        self.assertIn('rmse', summary)
        self.assertIn('mape', summary)
        self.assertIn('q_error_95', summary)
        self.assertIn('r_squared', summary)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_components(self):
        """Test that all components can work together."""
        # Test that we can create all necessary components
        dataset = create_imdb_dataset()
        self.assertIsNotNone(dataset)

        config = WorkloadConfig(num_queries=10, max_joins=1, seed=42)
        generator = WorkloadGenerator(dataset, config)
        self.assertIsNotNone(generator)

        models = get_available_models()
        self.assertGreater(len(models), 0)

        for model_type in models[:1]:  # Test first model
            if model_type == 'flat_vector':
                model = FlatVectorModel(model_type='random_forest')
                self.assertIsNotNone(model)

        calc = MetricsCalculator()
        self.assertIsNotNone(calc)

        config_obj = load_default_config()
        self.assertIsNotNone(config_obj)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)