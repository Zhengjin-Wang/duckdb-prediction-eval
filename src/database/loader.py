"""
Database loader for IMDB dataset into DuckDB.
"""
import os
from pathlib import Path
from typing import Optional

from database.duckdb_connection import create_database_connection
from datasets.imdb_dataset import create_imdb_dataset


class DatabaseLoader:
    """Load IMDB dataset into DuckDB database."""

    def __init__(self, db_path: str, data_dir: str):
        self.db_path = db_path
        self.data_dir = data_dir
        self.db = create_database_connection(db_path)
        self.dataset = create_imdb_dataset(data_dir)

    def load_dataset(self, force: bool = False):
        """Load IMDB dataset into DuckDB."""
        if os.path.exists(self.db_path) and not force:
            print(f"Database {self.db_path} already exists. Use force=True to overwrite.")
            return

        print(f"Loading IMDB dataset into {self.db_path}...")

        # Create schema
        self._create_schema()

        # Load tables
        table_files = self.dataset.get_table_files()
        for table_name, file_path in table_files.items():
            self._load_table(table_name, str(file_path))

        # Create indexes
        self._create_indexes()

        # Vacuum and analyze
        self.db.vacuum_analyze()

        print("Dataset loading completed successfully!")

    def _create_schema(self):
        """Create database schema."""
        schema = self.dataset.schema
        print("Creating database schema...")

        # Create all tables
        create_tables_sql = schema.get_all_create_tables_sql()
        self.db.execute_script(create_tables_sql)

        # Add foreign key constraints
        constraints = schema.get_foreign_key_constraints()
        for constraint in constraints:
            try:
                self.db.execute_query(constraint)
            except Exception as e:
                print(f"Warning: Could not create constraint: {e}")

    def _load_table(self, table_name: str, csv_path: str):
        """Load a single table from CSV file."""
        print(f"Loading table {table_name} from {csv_path}...")

        start_time = self.db.connect().execute("SELECT current_timestamp").fetchone()[0]

        try:
            # Use DuckDB's optimized CSV loading
            self.db.create_table_from_csv(table_name, csv_path)

            end_time = self.db.connect().execute("SELECT current_timestamp").fetchone()[0]
            load_time = (end_time - start_time).total_seconds()

            stats = self.db.get_table_stats(table_name)
            print(f"  Loaded {stats['row_count']} rows in {load_time:.2f} seconds")

        except Exception as e:
            print(f"  Error loading {table_name}: {e}")

    def _create_indexes(self):
        """Create indexes for better query performance."""
        print("Creating indexes...")
        schema = self.dataset.schema

        # Create indexes on foreign key columns
        for table_left, col_left, table_right, col_right in schema.relationships:
            try:
                # Index on left table's foreign key
                index_sql = f"CREATE INDEX idx_{table_left}_{col_left} ON {table_left}({col_left})"
                self.db.execute_query(index_sql)

                # Index on right table's primary key (usually already indexed)
                # But we'll ensure it for completeness
                if col_right == schema.primary_keys.get(table_right):
                    index_sql = f"CREATE INDEX idx_{table_right}_{col_right} ON {table_right}({col_right})"
                    self.db.execute_query(index_sql)

            except Exception as e:
                print(f"Warning: Could not create index for {table_left}.{col_left}: {e}")


def load_imdb_to_duckdb(db_path: str, data_dir: str, force: bool = False):
    """Convenience function to load IMDB dataset into DuckDB."""
    loader = DatabaseLoader(db_path, data_dir)
    loader.load_dataset(force=force)