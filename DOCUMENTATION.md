# DuckDB LCM Evaluation

## Overview

This project provides a comprehensive framework for evaluating Learned Cost Models (LCMs) on the IMDB dataset using DuckDB. It's designed to be lightweight, extensible, and easy to use for research and experimentation in query optimization.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IMDB Dataset  │    │   Workload      │    │   DuckDB        │
│                 │    │   Generation    │    │   Database      │
│  • title        │───▶│  • Random       │───▶│  • Connection   │
│  • cast_info    │    │  • Benchmark    │    │  • Execution    │
│  • movie_info   │    │  • Custom       │    │  • Statistics   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Training│    │   Model         │    │   Evaluation    │
│                 │    │   Registry      │    │                 │
│  • Flat Vector  │◀───│  • Factory      │───▶│  • Metrics      │
│  • MSCN         │    │  • Loading      │    │  • Comparison   │
│  • QPPNet       │    │  • Saving       │    │  • Visualization│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd duckdb-lcm-eval

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### 2. Dataset Preparation

The framework supports the IMDB dataset. You can either:

**Option A: Use provided dummy data for testing**
```bash
python src/main.py setup --data_dir data/datasets/imdb --db_path data/imdb.duckdb
```

**Option B: Download real IMDB dataset**
```bash
# Download from official IMDB datasets
wget https://datasets.imdbws.com/title.basics.tsv.gz
wget https://datasets.imdbws.com/name.basics.tsv.gz
# ... download other required files

# Extract and organize in data/datasets/imdb/
```

### 3. Workload Generation

Generate SQL workloads with configurable complexity:

```bash
# Generate 1000 queries with up to 3 joins
python src/main.py generate --num_queries 1000 --max_joins 3 --output workloads/imdb_workload.sql

# Generate workload with specific parameters
python src/main.py generate \
    --num_queries 500 \
    --max_joins 2 \
    --max_predicates 3 \
    --seed 42 \
    --output workloads/custom_workload.sql
```

### 4. Running Workloads

Execute workloads on DuckDB and collect performance metrics:

```bash
# Run workload and collect execution statistics
python src/main.py run \
    --db_path data/imdb.duckdb \
    --workload workloads/imdb_workload.sql \
    --output data/runs/imdb_runs.json \
    --min_runtime 100 \
    --max_runtime 300000

# Analyze workload results
python src/main.py analyze --workload_results data/runs/imdb_runs.json --output_dir output/
```

### 5. Model Training

Train baseline models on collected workload data:

```bash
# Train a specific model
python src/main.py train \
    --model_type flat_vector \
    --workload data/runs/imdb_runs.json \
    --device cpu \
    --epochs 100 \
    --output_dir models/

# Train all available models
python src/main.py train \
    --train_all \
    --workload data/runs/imdb_runs.json \
    --device cpu \
    --output_dir models/
```

### 6. Model Evaluation

Evaluate trained models on test workloads:

```bash
# Evaluate a specific model
python src/main.py evaluate \
    --model_type flat_vector \
    --model_path models/flat_vector_model.pkl \
    --test_workload data/runs/imdb_runs.json \
    --device cpu \
    --output output/evaluation_results.json

# Compare multiple models
python src/main.py evaluate \
    --model_type flat_vector \
    --model_path models/flat_vector_model.pkl \
    --test_workload data/runs/imdb_runs.json \
    --compare models/mscn_model.pkl models/qppnet_model.pkl \
    --baseline flat_vector
```

### 7. Complete Pipeline

Run the entire pipeline with a single command:

```bash
python src/main.py pipeline \
    --num_queries 1000 \
    --max_joins 3 \
    --train_all \
    --device cpu \
    --output_dir output/
```

## Model Documentation

### Flat Vector Model

A simple baseline model that uses traditional machine learning algorithms on flattened query features.

**Features:**
- Query execution statistics (min, max, avg, std times)
- Query complexity metrics (joins, predicates, aggregations)
- Table usage patterns
- Plan-specific features

**Algorithms Supported:**
- Random Forest
- Gradient Boosting
- Linear Regression

**Usage:**
```python
from src.models.registry import create_model

model = create_model('flat_vector', model_type='random_forest', n_estimators=100)
model.train(workload_results)
predictions = model.predict(test_workload)
```

### MSCN (Multi-Set Convolutional Networks)

A neural network model that learns from query plan structures.

**Architecture:**
- Separate feature extraction for tables, joins, and filters
- Multi-set convolution operations
- Aggregation layer for combining features
- Fully connected output layer

**Features:**
- Table features (presence, statistics)
- Join features (count, types)
- Filter features (count, operators)

**Usage:**
```python
from src.models.registry import create_model

model = create_model('mscn', device='cuda')
model.train(workload_results, epochs=200, batch_size=32)
```

### QPPNet (Query Plan Prediction Network)

A neural network designed for query plan analysis and cost prediction.

**Architecture:**
- Feature extraction layers
- Tree-like aggregation
- Hierarchical feature learning

**Features:**
- Comprehensive query structure analysis
- Plan-specific feature extraction
- Query complexity metrics

**Usage:**
```python
from src.models.registry import create_model

model = create_model('qppnet', device='cuda', hidden_dim=256)
model.train(workload_results, epochs=150)
```

## Advanced Usage

### Custom Models

To add a new model:

1. **Create model class** in `src/models/`:

```python
from src.models.registry import BaseModel

class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        # Initialize model
        pass

    def extract_features(self, query_plan):
        # Extract features from query plan
        return features

    def train(self, workload_results, **kwargs):
        # Train the model
        return training_results

    def predict(self, workload_results):
        # Make predictions
        return predictions

    def evaluate(self, workload_results):
        # Evaluate the model
        return evaluation_metrics

    def save(self, model_path):
        # Save the model
        pass

    @classmethod
    def load(cls, model_path, **kwargs):
        # Load the model
        return cls()
```

2. **Register the model** in `src/models/registry.py`:

```python
model_registry.register('custom_model', create_custom_model)
```

3. **Use the model**:

```python
model = create_model('custom_model', param1=value1, param2=value2)
```

### Custom Datasets

To support a new dataset:

1. **Create dataset class** in `src/datasets/`:

```python
from src.datasets.imdb_dataset import IMDBSchema

class CustomDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.schema = self._load_schema()

    def _load_schema(self):
        # Define schema
        return schema

    def get_table_files(self):
        # Return table file paths
        return table_files

    def validate_dataset(self):
        # Validate dataset
        return validation_results
```

2. **Update data loading** in `src/database/loader.py`:

```python
def load_custom_dataset(db_path, data_dir, force=False):
    # Implementation
    pass
```

### Custom Workloads

To generate custom workloads:

1. **Extend WorkloadGenerator** in `src/workloads/generator.py`:

```python
class CustomWorkloadGenerator(WorkloadGenerator):
    def _generate_single_query(self):
        # Custom query generation logic
        return query
```

2. **Use the custom generator**:

```python
generator = CustomWorkloadGenerator(dataset, config)
queries = generator.generate_workload()
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Basic Metrics
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **Max Error**: Maximum absolute error

### Advanced Metrics
- **Q-Error**: Quantile-based error at different percentiles (50, 95, 99)
- **R²**: Coefficient of determination
- **Error Distribution**: Detailed error statistics

### Usage

```python
from src.utils.metrics import ModelEvaluator

evaluator = ModelEvaluator(y_true, y_pred)
metrics = evaluator.calculate_all_metrics()

# Get specific metrics
summary = evaluator.get_summary()
comparison = evaluator.compare_models(other_evaluator)
```

## Configuration

### Command Line Arguments

All commands support extensive configuration through command-line arguments:

```bash
# Common arguments
--data_dir DATA_DIR     Dataset directory
--db_path DB_PATH       Database path
--model_type MODEL_TYPE Model type
--device DEVICE         Device (cpu/cuda)
--output OUTPUT         Output file/directory

# Workload-specific
--num_queries NUM_QUERIES   Number of queries
--max_joins MAX_JOINS       Maximum joins
--max_predicates MAX_PREDICATES Maximum predicates
--seed SEED                 Random seed

# Training-specific
--epochs EPOCHS         Number of training epochs
--batch_size BATCH_SIZE Batch size
--learning_rate LR      Learning rate
```

### Configuration Files

You can also use configuration files:

```python
from src.utils.config import Config

config = Config('config.yaml')
model_type = config.get('models.type', 'flat_vector')
epochs = config.get('training.epochs', 100)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   pip install -e .
   ```

2. **DuckDB Connection Issues**
   ```bash
   # Check if DuckDB is properly installed
   python -c "import duckdb; print(duckdb.__version__)"
   ```

3. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size or use CPU
   --device cpu --batch_size 16
   ```

### Debug Mode

Enable debug logging for detailed information:

```bash
python src/main.py --log_level DEBUG --log_file debug.log
```

### Performance Optimization

1. **Increase DuckDB memory limit**:
   ```python
   config = DatabaseConfig(memory_limit="16GB", threads=8)
   ```

2. **Use CUDA for neural networks**:
   ```bash
   --device cuda
   ```

3. **Optimize workload execution**:
   ```python
   config = WorkloadConfig(
       min_runtime_ms=50,
       max_runtime_ms=600000,  # 10 minutes
       query_timeout=300000    # 5 minutes
   )
   ```

## Contributing

### Development Setup

1. **Fork and clone the repository**
2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install pytest coverage
   ```
4. **Run tests**:
   ```bash
   python -m pytest tests/ -v
   ```
5. **Run linting** (if available):
   ```bash
   # Add linting tools as needed
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Write docstrings for all public functions and classes
- Include unit tests for new functionality

### Testing

All code should be tested:

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage
coverage run -m pytest tests/
coverage report -m
coverage html

# Run specific test
python -m pytest tests/test_models.py::TestFlatVectorModel -v
```

### Documentation

Update documentation for:
- New models
- New datasets
- New features
- API changes

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{duckdb_lcm_eval,
  title = {DuckDB LCM Evaluation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/duckdb-lcm-eval},
}
```

## Contact

For questions, issues, or contributions, please use:

- **GitHub Issues**: [github.com/yourusername/duckdb-lcm-eval/issues](https://github.com/yourusername/duckdb-lcm-eval/issues)
- **Email**: your.email@example.com

## Acknowledgments

This project builds upon research in learned cost models and query optimization. We thank the authors of the original papers and implementations that inspired this work.