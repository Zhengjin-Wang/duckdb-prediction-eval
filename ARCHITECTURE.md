# DuckDB LCM Evaluation Framework

## Project Structure

```
duckdb-lcm-eval/
├── src/                          # Main source code
│   ├── database/                 # DuckDB database operations
│   │   ├── __init__.py
│   │   ├── duckdb_connection.py  # DuckDB connection and query execution
│   │   └── loader.py            # Dataset loading utilities
│   ├── datasets/                # Dataset handling
│   │   ├── __init__.py
│   │   └── imdb_dataset.py      # IMDB dataset configuration
│   ├── models/                  # Machine learning models
│   │   ├── __init__.py
│   │   ├── flat_vector.py       # Flat Vector baseline model
│   │   ├── mscn.py             # MSCN baseline model
│   │   ├── qppnet.py           # QPPNet baseline model
│   │   └── registry.py         # Model factory and registry
│   ├── workloads/              # Workload generation and execution
│   │   ├── __init__.py
│   │   ├── generator.py        # Workload generation
│   │   └── analyzer.py         # Workload analysis
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── preprocessing.py    # Data preprocessing
│   ├── scripts/                # Command-line scripts
│   │   ├── __init__.py
│   │   ├── download_imdb.py    # Dataset download
│   │   ├── generate_workload.py # Workload generation
│   │   ├── run_workload.py     # Workload execution
│   │   ├── train_model.py      # Model training
│   │   └── evaluate_model.py   # Model evaluation
│   └── main.py                 # Main CLI entry point
├── data/                       # Data directory
│   ├── datasets/               # Raw dataset files
│   │   └── imdb/              # IMDB dataset
│   ├── runs/                   # Query execution results
│   └── workloads/             # SQL workload files
├── models/                     # Trained model files
├── workloads/                  # Generated workload files
├── output/                     # Evaluation results and reports
├── examples/                   # Example usage scripts
│   └── example_usage.py       # Comprehensive example
├── tests/                      # Unit tests
│   └── test_framework.py      # Framework tests
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project configuration
├── README.md                 # Project documentation
├── DOCUMENTATION.md          # Detailed documentation
└── LICENSE                   # License file
```

## Key Components

### Core Modules

1. **Database Layer** (`src/database/`)
   - `duckdb_connection.py`: DuckDB connection management, query execution, and statistics collection
   - `loader.py`: Dataset loading and database setup

2. **Dataset Layer** (`src/datasets/`)
   - `imdb_dataset.py`: IMDB dataset schema, configuration, and validation

3. **Models Layer** (`src/models/`)
   - `flat_vector.py`: Traditional ML models (Random Forest, Gradient Boosting, Linear Regression)
   - `mscn.py`: Multi-Set Convolutional Networks
   - `qppnet.py`: Query Plan Prediction Networks
   - `registry.py`: Model factory and management

4. **Workloads Layer** (`src/workloads/`)
   - `generator.py`: SQL workload generation with configurable complexity
   - `analyzer.py`: Workload analysis and validation

5. **Utilities** (`src/utils/`)
   - `config.py`: Configuration management and environment setup
   - `metrics.py`: Comprehensive evaluation metrics
   - `preprocessing.py`: Data preprocessing and feature engineering

### Scripts

- **Command-line Interface**: `src/main.py` provides the main CLI with subcommands
- **Individual Scripts**: Each major operation has a dedicated script in `src/scripts/`
- **Examples**: `examples/example_usage.py` demonstrates complete usage patterns

### Testing

- **Unit Tests**: `tests/test_framework.py` contains comprehensive tests for all components
- **Integration Tests**: Examples serve as integration test scenarios
- **Coverage**: Configured for code coverage analysis

## Data Flow

```
1. Dataset Setup
   ↓
2. Workload Generation
   ↓
3. Workload Execution (DuckDB)
   ↓
4. Result Collection & Analysis
   ↓
5. Model Training
   ↓
6. Model Evaluation
   ↓
7. Results Visualization
```

## Configuration

### Environment Variables

```bash
# Database configuration
DATABASE_PATH="data/imdb.duckdb"
MODEL_DIR="models/"
WORKLOAD_DIR="workloads/"
OUTPUT_DIR="output/"

# Training configuration
DEVICE="cpu"  # or "cuda"
EPOCHS=100
BATCH_SIZE=32
```

### Command-line Arguments

All major operations support extensive configuration:

```bash
# General
--data_dir, --db_path, --model_type, --device, --output

# Workload generation
--num_queries, --max_joins, --max_predicates, --seed

# Training
--epochs, --batch_size, --learning_rate, --train_all

# Evaluation
--model_path, --test_workload, --compare, --baseline
```

## Dependencies

### Core Dependencies

- **duckdb**: Database engine
- **pandas, numpy**: Data manipulation
- **scikit-learn**: Traditional ML models
- **torch**: Deep learning framework
- **lightgbm**: Gradient boosting

### Optional Dependencies

- **matplotlib, seaborn**: Visualization
- **pytest, coverage**: Testing
- **yaml**: Configuration files

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage Patterns

### Research Workflow

```bash
# 1. Setup experiment
python src/main.py setup --data_dir experiment/data --db_path experiment/db.duckdb

# 2. Generate workload
python src/main.py generate --num_queries 5000 --max_joins 4 --output experiment/workload.sql

# 3. Run workload
python src/main.py run --db_path experiment/db.duckdb --workload experiment/workload.sql --output experiment/results.json

# 4. Train models
python src/main.py train --train_all --workload experiment/results.json --device cuda

# 5. Evaluate and compare
python src/main.py evaluate --model_type all --test_workload experiment/results.json --output experiment/evaluation.json

# 6. Analyze results
python src/main.py analyze --workload_results experiment/results.json --output_dir experiment/
```

### Development Workflow

```bash
# Run tests
python -m pytest tests/ -v

# Run examples
python examples/example_usage.py

# Check coverage
coverage run -m pytest tests/
coverage report

# Development CLI
python src/main.py --log_level DEBUG pipeline --num_queries 100 --train_all
```

### Production Workflow

```bash
# Single command pipeline
python src/main.py pipeline \
    --num_queries 10000 \
    --max_joins 3 \
    --train_all \
    --device cuda \
    --output_dir production/

# Specific model training
python src/main.py train \
    --model_type qppnet \
    --workload production/workload_results.json \
    --device cuda \
    --epochs 200 \
    --output_dir production/models/
```

## Extensibility

### Adding Models

1. Create model class in `src/models/`
2. Implement `BaseModel` interface
3. Register in `src/models/registry.py`
4. Add to CLI in `src/main.py`

### Adding Datasets

1. Create dataset class in `src/datasets/`
2. Define schema and relationships
3. Implement data loading
4. Update database loader

### Adding Workloads

1. Extend `WorkloadGenerator` in `src/workloads/generator.py`
2. Implement custom query generation
3. Add validation logic
4. Update CLI options

## Performance Considerations

### Database Performance

- Use appropriate memory limits for DuckDB
- Enable parallel execution with multiple threads
- Create indexes on frequently queried columns
- Use appropriate data types and compression

### Model Training Performance

- Use CUDA when available for neural networks
- Optimize batch sizes for memory constraints
- Implement early stopping to prevent overfitting
- Use appropriate learning rates and optimizers

### Workload Execution Performance

- Set appropriate query timeouts
- Use workload validation to filter slow queries
- Implement parallel query execution
- Monitor memory usage during execution

## Best Practices

### Code Organization

- Follow the existing module structure
- Use type hints for all function parameters
- Write comprehensive docstrings
- Implement proper error handling

### Testing

- Write unit tests for all new functionality
- Use integration tests for end-to-end workflows
- Test with different dataset sizes
- Validate model performance metrics

### Documentation

- Update README for major changes
- Add examples for new features
- Document configuration options
- Provide troubleshooting guides

### Version Control

- Use meaningful commit messages
- Create feature branches for development
- Use pull requests for code review
- Tag releases appropriately

This architecture provides a solid foundation for evaluating learned cost models while remaining flexible and extensible for future research and development.