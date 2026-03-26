# DuckDB LCM Evaluation

A lightweight, DuckDB-focused implementation for evaluating Learned Cost Models (LCMs) on the IMDB dataset.

## Features

- **DuckDB Support**: Optimized for DuckDB database engine
- **IMDB Dataset**: Pre-configured for the Internet Movie Database benchmark
- **Baseline Models**: Implementation of key LCM baselines from research literature
- **Workload Management**: Generate and execute SQL workloads
- **Model Evaluation**: Comprehensive evaluation metrics and visualization

## Quick Start

### Installation

```bash
git clone <repository-url>
cd duckdb-lcm-eval
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

1. **Setup IMDB dataset**:
```bash
python src/main.py setup --data_dir data/datasets/imdb --db_path data/imdb.duckdb
```

2. **Generate workload**:
```bash
python src/main.py generate --num_queries 1000 --max_joins 3 --output workloads/imdb_workload.sql
```

3. **Run workload on DuckDB**:
```bash
python src/main.py run --db_path data/imdb.duckdb --workload workloads/imdb_workload.sql --output data/runs/imdb_runs.json
```

4. **Train baseline model**:
```bash
python src/main.py train --model_type flat_vector --workload data/runs/imdb_runs.json --device cpu
```

5. **Evaluate model**:
```bash
python src/main.py evaluate --model_type flat_vector --model_path models/flat_vector_model.pkl --test_workload data/runs/imdb_runs.json
```

6. **Run complete pipeline**:
```bash
python src/main.py pipeline --num_queries 500 --max_joins 2 --train_all
```

## Project Structure

```
duckdb-lcm-eval/
├── src/
│   ├── database/          # DuckDB connection and management
│   ├── datasets/          # Dataset handling and preprocessing
│   ├── models/            # Baseline model implementations
│   ├── workloads/         # Workload generation and execution
│   ├── utils/             # Utility functions
│   ├── scripts/           # Command-line scripts
│   └── main.py            # Main CLI entry point
├── data/                  # Raw and processed data
├── models/                # Trained model files
├── workloads/             # SQL workload files
├── output/                # Evaluation results
├── examples/              # Example usage scripts
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── setup.py             # Package setup
```

## Supported Models

### Baseline Models
- **Flat Vector**: Simple feature vector model
- **MSCN**: Multi-Set Convolutional Networks
- **QPPNet**: Query Plan Prediction Network

### Model Types
- **Workload-Agnostic**: Train on multiple datasets
- **Workload-Driven**: Train on target database

## Dataset Support

Currently supports:
- **IMDB**: Internet Movie Database (default)

## Workload Types

- **Random Workloads**: Generated with configurable complexity
- **Benchmark Workloads**: Standard IMDB query sets
- **Custom Workloads**: User-defined SQL queries

## CLI Commands

### Setup
```bash
python src/main.py setup --data_dir data/datasets/imdb --db_path data/imdb.duckdb
```

### Generate Workload
```bash
python src/main.py generate --num_queries 1000 --max_joins 3 --output workloads/imdb_workload.sql
```

### Run Workload
```bash
python src/main.py run --db_path data/imdb.duckdb --workload workloads/imdb_workload.sql --output data/runs/imdb_runs.json
```

### Train Models
```bash
python src/main.py train --model_type flat_vector --workload data/runs/imdb_runs.json --device cpu
python src/main.py train --train_all  # Train all available models
```

### Evaluate Models
```bash
python src/main.py evaluate --model_type flat_vector --model_path models/flat_vector_model.pkl --test_workload data/runs/imdb_runs.json
```

### Run Complete Pipeline
```bash
python src/main.py pipeline --num_queries 500 --max_joins 2 --train_all
```

### Analyze Results
```bash
python src/main.py analyze --workload_results data/runs/imdb_runs.json --output_dir output/
```

## Configuration

Configuration is managed through command-line arguments or can be extended to support configuration files.

### Common Parameters
- `--data_dir`: Dataset directory (default: data/datasets/imdb)
- `--db_path`: DuckDB database path (default: data/imdb.duckdb)
- `--model_type`: Model type (flat_vector, mscn, qppnet)
- `--device`: Training device (cpu, cuda)
- `--num_queries`: Number of queries to generate
- `--max_joins`: Maximum number of joins per query

## Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Q-Error**: Quantile-based error metrics
- **R²**: Coefficient of determination

## Development

### Adding New Models

1. Create model class in `src/models/`
2. Implement required methods: `train()`, `predict()`, `save()`, `load()`
3. Add model to model registry in `src/models/registry.py`

### Adding New Datasets

1. Create dataset configuration in `src/datasets/`
2. Implement data loading and preprocessing
3. Add schema and relationship information

### Adding New Workloads

1. Create workload generator in `src/workloads/`
2. Implement query generation logic
3. Add workload validation

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_framework.py::TestDatabaseConnection

# Run tests with coverage
coverage run -m pytest tests/
coverage report
coverage html
```

### Running Examples

```bash
# Run example usage
python examples/example_usage.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

Apache License 2.0

## Acknowledgments

This project is inspired by research in learned cost models and query optimization. We thank the original authors of the baseline models implemented here.