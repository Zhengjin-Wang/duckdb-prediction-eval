"""
Utility functions and configuration management.
"""
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class Config:
    """Configuration management class."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path:
            self.load(config_path)

    def load(self, config_path: str):
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def save(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            self.set(key, value)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def ensure_directory(path: Union[str, Path]):
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: Union[str, Path]):
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: Union[str, Path]):
    """Save data to YAML file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def get_environment_variable(name: str, default: Any = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(name)
    if value is None and default is not None:
        return default
    elif value is None:
        raise ValueError(f"Environment variable {name} not set")
    return value


def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validate and convert path to Path object."""
    path = Path(path)

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.2f}{size_names[i]}"


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information."""
    file_path = Path(file_path)
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}

    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": stat.st_size,
        "size_formatted": format_size(stat.st_size),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "extension": file_path.suffix
    }


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "database": {
            "path": "data/imdb.duckdb",
            "read_only": False,
            "memory_limit": "8GB",
            "threads": 4
        },
        "dataset": {
            "name": "imdb",
            "data_dir": "data/datasets/imdb"
        },
        "workload": {
            "num_queries": 1000,
            "max_joins": 3,
            "max_predicates": 3,
            "max_aggregates": 1,
            "seed": 42,
            "timeout": 30000
        },
        "models": {
            "output_dir": "models/",
            "device": "cpu"
        },
        "evaluation": {
            "output_dir": "output/",
            "metrics": ["rmse", "mape", "q_error_95"]
        }
    }


def load_default_config() -> Config:
    """Load default configuration."""
    default_config = create_default_config()
    config = Config()
    config.config = default_config
    return config