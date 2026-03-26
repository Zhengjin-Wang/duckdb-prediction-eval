"""
IMDB dataset configuration and handling module.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class IMDBSchema:
    """IMDB dataset schema definition."""

    def __init__(self):
        self.tables = [
            "title", "cast_info", "company_name", "company_type",
            "info_type", "keyword", "movie_companies", "movie_info_idx",
            "movie_keyword", "movie_info", "person_info", "kind_type",
            "char_name", "aka_name", "name"
        ]

        # Table relationships (foreign key constraints)
        self.relationships = [
            ("cast_info", "movie_id", "title", "id"),
            ("movie_companies", "company_id", "company_name", "id"),
            ("movie_companies", "company_type_id", "company_type", "id"),
            ("movie_info_idx", "info_type_id", "info_type", "id"),
            ("movie_keyword", "keyword_id", "keyword", "id"),
            ("movie_companies", "movie_id", "title", "id"),
            ("movie_info_idx", "movie_id", "title", "id"),
            ("cast_info", "person_role_id", "char_name", "id"),
            ("movie_keyword", "movie_id", "title", "id"),
            ("movie_info", "movie_id", "title", "id"),
            ("person_info", "person_id", "name", "id"),
            ("title", "kind_id", "kind_type", "id"),
            ("cast_info", "person_id", "aka_name", "id"),
            ("aka_name", "person_id", "name", "id")
        ]

        # Primary keys for each table
        self.primary_keys = {
            "title": "id",
            "cast_info": "id",
            "company_name": "id",
            "company_type": "id",
            "info_type": "id",
            "keyword": "id",
            "movie_companies": "id",
            "movie_info_idx": "id",
            "movie_keyword": "id",
            "movie_info": "id",
            "person_info": "id",
            "kind_type": "id",
            "char_name": "id",
            "aka_name": "id",
            "name": "id"
        }

        # Data types for schema generation
        self.column_types = {
            "title": {
                "id": "INTEGER PRIMARY KEY",
                "title": "VARCHAR",
                "imdb_index": "VARCHAR(5)",
                "kind_id": "INTEGER",
                "production_year": "INTEGER",
                "imdb_id": "INTEGER",
                "phonetic_code": "VARCHAR(5)",
                "episode_of_id": "INTEGER",
                "season_nr": "INTEGER",
                "episode_nr": "INTEGER",
                "series_years": "VARCHAR(49)",
                "md5sum": "VARCHAR(32)"
            },
            "cast_info": {
                "id": "INTEGER PRIMARY KEY",
                "person_id": "INTEGER",
                "movie_id": "INTEGER",
                "person_role_id": "INTEGER",
                "note": "VARCHAR",
                "nr_order": "INTEGER",
                "role_id": "INTEGER"
            },
            "company_name": {
                "id": "INTEGER PRIMARY KEY",
                "name": "VARCHAR",
                "country_code": "VARCHAR(6)",
                "imdb_id": "INTEGER",
                "name_pcode_nf": "VARCHAR(5)",
                "name_pcode_sf": "VARCHAR(5)",
                "md5sum": "VARCHAR(32)"
            },
            "company_type": {
                "id": "INTEGER PRIMARY KEY",
                "kind": "VARCHAR(32)"
            },
            "info_type": {
                "id": "INTEGER PRIMARY KEY",
                "info": "VARCHAR(32)"
            },
            "keyword": {
                "id": "INTEGER PRIMARY KEY",
                "keyword": "VARCHAR",
                "phonetic_code": "VARCHAR(5)"
            },
            "movie_companies": {
                "id": "INTEGER PRIMARY KEY",
                "movie_id": "INTEGER",
                "company_id": "INTEGER",
                "company_type_id": "INTEGER",
                "note": "VARCHAR"
            },
            "movie_info_idx": {
                "id": "INTEGER PRIMARY KEY",
                "movie_id": "INTEGER",
                "info_type_id": "INTEGER",
                "info": "VARCHAR",
                "note": "VARCHAR(1)"
            },
            "movie_keyword": {
                "id": "INTEGER PRIMARY KEY",
                "movie_id": "INTEGER",
                "keyword_id": "INTEGER"
            },
            "movie_info": {
                "id": "INTEGER PRIMARY KEY",
                "movie_id": "INTEGER",
                "info_type_id": "INTEGER",
                "info": "VARCHAR",
                "note": "VARCHAR"
            },
            "person_info": {
                "id": "INTEGER PRIMARY KEY",
                "person_id": "INTEGER",
                "info_type_id": "INTEGER",
                "info": "VARCHAR",
                "note": "VARCHAR"
            },
            "kind_type": {
                "id": "INTEGER PRIMARY KEY",
                "kind": "VARCHAR(15)"
            },
            "char_name": {
                "id": "INTEGER PRIMARY KEY",
                "name": "VARCHAR",
                "imdb_index": "VARCHAR(2)",
                "imdb_id": "INTEGER",
                "name_pcode_nf": "VARCHAR(5)",
                "surname_pcode": "VARCHAR(5)",
                "md5sum": "VARCHAR(32)"
            },
            "aka_name": {
                "id": "INTEGER PRIMARY KEY",
                "person_id": "INTEGER",
                "name": "VARCHAR",
                "imdb_index": "VARCHAR(3)",
                "name_pcode_cf": "VARCHAR(11)",
                "name_pcode_nf": "VARCHAR(11)",
                "surname_pcode": "VARCHAR(11)",
                "md5sum": "VARCHAR(65)"
            },
            "name": {
                "id": "INTEGER PRIMARY KEY",
                "name": "VARCHAR",
                "imdb_index": "VARCHAR(9)",
                "imdb_id": "INTEGER",
                "gender": "VARCHAR(1)",
                "name_pcode_cf": "VARCHAR(5)",
                "name_pcode_nf": "VARCHAR(5)",
                "surname_pcode": "VARCHAR(5)",
                "md5sum": "VARCHAR(32)"
            }
        }

    def get_create_table_sql(self, table_name: str) -> str:
        """Generate CREATE TABLE SQL for a given table."""
        if table_name not in self.column_types:
            raise ValueError(f"Unknown table: {table_name}")

        columns = []
        for col_name, col_type in self.column_types[table_name].items():
            columns.append(f'"{col_name}" {col_type}')

        return f"CREATE TABLE {table_name} ({', '.join(columns)});"

    def get_all_create_tables_sql(self) -> str:
        """Generate CREATE TABLE SQL for all tables."""
        sql_statements = []
        for table in self.tables:
            sql_statements.append(self.get_create_table_sql(table))
        return '\n\n'.join(sql_statements)

    def get_foreign_key_constraints(self) -> List[str]:
        """Get all foreign key constraint SQL statements."""
        constraints = []
        for table_left, col_left, table_right, col_right in self.relationships:
            constraint = f"""
            ALTER TABLE {table_left}
            ADD FOREIGN KEY ({col_left}) REFERENCES {table_right}({col_right});
            """
            constraints.append(constraint.strip())
        return constraints


class IMDBDataset:
    """IMDB dataset management and utilities."""

    def __init__(self, data_dir: str = "data/datasets/imdb"):
        self.data_dir = Path(data_dir)
        self.schema = IMDBSchema()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_table_files(self) -> Dict[str, Path]:
        """Get paths to all table CSV files."""
        table_files = {}
        for table in self.schema.tables:
            csv_file = self.data_dir / f"{table}.csv"
            if csv_file.exists():
                table_files[table] = csv_file
        return table_files

    def validate_dataset(self) -> Dict[str, bool]:
        """Validate that all required table files exist."""
        table_files = self.get_table_files()
        validation = {}
        for table in self.schema.tables:
            validation[table] = table in table_files
        return validation

    def get_dataset_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get basic statistics about the dataset."""
        stats = {}
        table_files = self.get_table_files()

        for table, file_path in table_files.items():
            try:
                df = pd.read_csv(file_path, nrows=0)  # Just get column info
                stats[table] = {
                    'columns': len(df.columns),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'has_data': file_path.stat().st_size > 0
                }
            except Exception as e:
                stats[table] = {'error': str(e)}

        return stats

    def load_schema_from_json(self, schema_file: Optional[str] = None) -> Dict[str, Any]:
        """Load schema from JSON file if available."""
        if schema_file is None:
            schema_file = self.data_dir / "schema.json"

        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                return json.load(f)
        else:
            # Return default schema structure
            return {
                "name": "imdb",
                "tables": self.schema.tables,
                "relationships": self.schema.relationships,
                "primary_keys": self.schema.primary_keys
            }

    def save_schema_to_json(self, output_file: Optional[str] = None):
        """Save schema to JSON file."""
        if output_file is None:
            output_file = self.data_dir / "schema.json"

        schema_data = {
            "name": "imdb",
            "tables": self.schema.tables,
            "relationships": self.schema.relationships,
            "primary_keys": self.schema.primary_keys,
            "column_types": self.schema.column_types
        }

        with open(output_file, 'w') as f:
            json.dump(schema_data, f, indent=2)


def create_imdb_dataset(data_dir: str = "data/datasets/imdb") -> IMDBDataset:
    """Factory function to create IMDB dataset instance."""
    return IMDBDataset(data_dir=data_dir)