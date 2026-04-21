"""
DuckDB database connection and management module.
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import duckdb
import pandas as pd
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Configuration for DuckDB database connection."""
    db_path: str = "imdb.duckdb"
    read_only: bool = False
    memory_limit: str = "8GB"
    threads: int = 4


class DuckDBConnection:
    """DuckDB database connection wrapper with enhanced functionality."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish connection to DuckDB database."""
        if self.connection is None:
            self.connection = duckdb.connect(
                database=self.config.db_path,
                read_only=self.config.read_only
            )
            # Set performance parameters
            self.connection.execute(f"SET memory_limit='{self.config.memory_limit}'")
            self.connection.execute(f"SET threads={self.config.threads}")
        return self.connection

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        conn = self.connect()
        if params:
            return conn.execute(query, params).df()
        else:
            return conn.execute(query).df()

    def execute_script(self, script: str):
        """Execute a SQL script file."""
        conn = self.connect()
        conn.execute(script)

    def create_table_from_csv(self, table_name: str, csv_path: str, **kwargs):
        """Create table from CSV file, or insert into it if it already exists."""
        conn = self.connect()
        csv_params = {
            'header': True,
            'escape': '\\',
            'quote': '"',
            'encoding': 'utf-8',
            **kwargs
        }

        read_csv_expr = (
            f"read_csv_auto('{csv_path}',"
            f" header={str(csv_params['header']).lower()},"
            f" escape='{csv_params['escape']}',"
            f" quote='{csv_params['quote']}',"
            f" encoding='{csv_params['encoding']}')"
        )

        # If the table was pre-created by the schema step, INSERT into it.
        # Otherwise, create the table from the CSV.
        exists = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()

        if exists:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM {read_csv_expr}")
        else:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {read_csv_expr}")

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table schema information."""
        query = f"DESCRIBE {table_name}"
        return self.execute_query(query)

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics."""
        conn = self.connect()
        stats = {}

        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        stats['row_count'] = conn.execute(count_query).fetchone()[0]

        # Get table size (approximate)
        size_query = f"SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as size"
        try:
            stats['size'] = conn.execute(size_query).fetchone()[0]
        except:
            stats['size'] = "N/A"

        return stats

    def get_column_stats(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get column statistics."""
        conn = self.connect()
        stats = {}

        # Basic stats
        query = f"""
        SELECT
            COUNT(*) as count,
            COUNT(DISTINCT "{column_name}") as distinct_count,
            MIN("{column_name}") as min_val,
            MAX("{column_name}") as max_val
        FROM {table_name}
        """

        result = conn.execute(query).fetchone()
        stats.update({
            'count': result[0],
            'distinct_count': result[1],
            'min': result[2],
            'max': result[3]
        })

        # Calculate null percentage
        null_query = f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE "{column_name}" IS NULL
        """
        null_count = conn.execute(null_query).fetchone()[0]
        stats['null_percentage'] = (null_count / stats['count']) * 100

        return stats

    def vacuum_analyze(self):
        """Run vacuum and analyze operations."""
        conn = self.connect()
        # DuckDB doesn't have VACUUM/`PRAGMA optimize`. ANALYZE refreshes
        # statistics and CHECKPOINT flushes the WAL, which is the closest
        # equivalent to a post-load tidy-up.
        conn.execute("ANALYZE")
        conn.execute("CHECKPOINT")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        conn = self.connect()
        stats = {}

        # Get all tables
        tables_query = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
        """
        tables = conn.execute(tables_query).fetchall()
        table_names = [t[0] for t in tables]

        stats['tables'] = {}
        for table in table_names:
            stats['tables'][table] = self.get_table_stats(table)

        return stats


class QueryExecutor:
    """Enhanced query execution with timing and statistics collection."""

    def __init__(self, db_connection: DuckDBConnection):
        self.db = db_connection

    def execute_with_timing(self, query: str, repetitions: int = 3) -> Dict[str, Any]:
        """Execute query with timing measurements."""
        results = []
        execution_times = []

        for _ in range(repetitions):
            start_time = time.perf_counter()
            try:
                result = self.db.execute_query(query)
                end_time = time.perf_counter()

                execution_time = (end_time - start_time) * 1000  # Convert to ms
                execution_times.append(execution_time)
                results.append({
                    'success': True,
                    'execution_time_ms': execution_time,
                    'row_count': len(result),
                    'error': None
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'execution_time_ms': 0,
                    'row_count': 0,
                    'error': str(e)
                })

        # Calculate statistics
        successful_times = [r['execution_time_ms'] for r in results if r['success']]
        stats = {
            'query': query,
            'repetitions': repetitions,
            'successful_executions': len(successful_times),
            'failed_executions': len(results) - len(successful_times)
        }

        if successful_times:
            stats.update({
                'min_time_ms': min(successful_times),
                'max_time_ms': max(successful_times),
                'avg_time_ms': sum(successful_times) / len(successful_times),
                'std_time_ms': pd.Series(successful_times).std() if len(successful_times) > 1 else 0
            })

        return stats

    def explain_query(self, query: str) -> str:
        """Get query execution plan."""
        conn = self.db.connect()
        plan = conn.execute(f"EXPLAIN {query}").fetchall()
        return '\n'.join([str(p) for p in plan])

    def get_query_statistics(self, query: str, repetitions: int = 3) -> Dict[str, Any]:
        """Get comprehensive query statistics including plan and timing."""
        stats = self.execute_with_timing(query, repetitions)
        stats['plan'] = self.explain_query(query)
        return stats


def create_database_connection(db_path: str = "imdb.duckdb", **kwargs) -> DuckDBConnection:
    """Factory function to create database connection."""
    config = DatabaseConfig(db_path=db_path, **kwargs)
    return DuckDBConnection(config)