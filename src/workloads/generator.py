"""
Workload generation and execution for IMDB benchmark.
"""
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import json

from database.duckdb_connection import DuckDBConnection, QueryExecutor
from datasets.imdb_dataset import IMDBDataset


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    num_queries: int = 100
    max_joins: int = 3
    max_predicates: int = 3
    max_aggregates: int = 1
    max_group_by: int = 1
    seed: int = 42
    query_timeout: int = 30000  # in milliseconds
    min_runtime_ms: int = 100
    max_runtime_ms: int = 300000


class WorkloadGenerator:
    """Generate SQL workloads for IMDB dataset."""

    def __init__(self, dataset: IMDBDataset, config: WorkloadConfig):
        self.dataset = dataset
        self.config = config
        self.schema = dataset.schema

        # Set random seed for reproducibility
        random.seed(config.seed)

        # Query templates
        self.select_templates = [
            "SELECT COUNT(*) FROM {tables}",
            "SELECT COUNT(DISTINCT {columns}) FROM {tables}",
            "SELECT {aggregates} FROM {tables}",
            "SELECT {columns} FROM {tables}",
        ]

        self.join_templates = [
            "JOIN {table_right} ON {table_left}.{col_left} = {table_right}.{col_right}",
            "LEFT JOIN {table_right} ON {table_left}.{col_left} = {table_right}.{col_right}",
        ]

        self.where_templates = [
            "{table}.{column} = '{value}'",
            "{table}.{column} > {value}",
            "{table}.{column} < {value}",
            "{table}.{column} LIKE '%{value}%'",
            "{table}.{column} IS NOT NULL",
        ]

    def generate_workload(self) -> List[str]:
        """Generate a workload of SQL queries."""
        queries = []
        for i in range(self.config.num_queries):
            query = self._generate_single_query()
            queries.append(query)
        return queries

    def _generate_single_query(self) -> str:
        """Generate a single SQL query."""
        # Determine query complexity
        num_joins = random.randint(1, min(self.config.max_joins, len(self.schema.tables)))
        num_predicates = random.randint(0, self.config.max_predicates)
        num_aggregates = random.randint(0, self.config.max_aggregates)
        num_group_by = random.randint(0, self.config.max_group_by)

        # Select base table
        tables = self._select_tables(num_joins)
        joins = self._generate_joins(tables)

        # Generate SELECT clause
        select_clause = self._generate_select_clause(tables, num_aggregates, num_group_by)

        # Generate WHERE clause
        where_clause = self._generate_where_clause(tables, num_predicates)

        # Generate GROUP BY clause
        group_by_clause = self._generate_group_by_clause(tables, num_group_by)

        # Construct query
        query_parts = [select_clause, f"FROM {tables[0]}"]
        if joins:
            query_parts.extend(joins)
        if where_clause:
            query_parts.append(f"WHERE {where_clause}")
        if group_by_clause:
            query_parts.append(f"GROUP BY {group_by_clause}")

        return " ".join(query_parts) + ";"

    def _select_tables(self, num_joins: int) -> List[str]:
        """Select tables for the query."""
        # Start with a random table
        selected_tables = [random.choice(self.schema.tables)]

        # Add tables through relationships
        for _ in range(num_joins - 1):
            current_table = selected_tables[-1]
            related_tables = self._get_related_tables(current_table)

            if related_tables:
                next_table = random.choice(related_tables)
                if next_table not in selected_tables:
                    selected_tables.append(next_table)
            else:
                break

        return selected_tables

    def _get_related_tables(self, table_name: str) -> List[str]:
        """Get tables related to the given table."""
        related = []
        for table_left, col_left, table_right, col_right in self.schema.relationships:
            if table_left == table_name and table_right not in related:
                related.append(table_right)
            elif table_right == table_name and table_left not in related:
                related.append(table_left)
        return related

    def _generate_joins(self, tables: List[str]) -> List[str]:
        """Generate JOIN clauses."""
        joins = []
        for i in range(1, len(tables)):
            table_left = tables[i-1]
            table_right = tables[i]

            # Find relationship between tables
            relationship = self._find_relationship(table_left, table_right)
            if relationship:
                join_type = random.choice(self.join_templates)
                joins.append(join_type.format(
                    table_left=table_left,
                    col_left=relationship[1],
                    table_right=table_right,
                    col_right=relationship[3]
                ))

        return joins

    def _find_relationship(self, table_left: str, table_right: str) -> Optional[Tuple]:
        """Find relationship between two tables."""
        for rel in self.schema.relationships:
            if (rel[0] == table_left and rel[2] == table_right) or \
               (rel[0] == table_right and rel[2] == table_left):
                return rel
        return None

    def _generate_select_clause(self, tables: List[str], num_aggregates: int, num_group_by: int) -> str:
        """Generate SELECT clause."""
        if num_aggregates > 0:
            aggregates = self._generate_aggregates(tables, num_aggregates)
            return f"SELECT {', '.join(aggregates)}"
        elif num_group_by > 0:
            columns = self._generate_columns(tables, num_group_by)
            return f"SELECT {', '.join(columns)}"
        else:
            template = random.choice(self.select_templates)
            if "{columns}" in template:
                columns = self._generate_columns(tables, random.randint(1, 3))
                return template.format(tables=', '.join(tables), columns=', '.join(columns))
            else:
                return template.format(tables=', '.join(tables))

    def _generate_aggregates(self, tables: List[str], num_aggregates: int) -> List[str]:
        """Generate aggregate functions."""
        aggregates = []
        functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']

        for _ in range(num_aggregates):
            func = random.choice(functions)
            table = random.choice(tables)
            numeric_only = func in ('SUM', 'AVG')
            column = self._get_random_column(table, numeric_only=numeric_only)
            aggregates.append(f"{func}(\"{table}\".\"{column}\")")

        return aggregates

    def _generate_columns(self, tables: List[str], num_columns: int) -> List[str]:
        """Generate column references."""
        columns = []
        for _ in range(num_columns):
            table = random.choice(tables)
            column = self._get_random_column(table)
            columns.append(f"\"{table}\".\"{column}\"")
        return columns

    def _generate_where_clause(self, tables: List[str], num_predicates: int) -> str:
        """Generate WHERE clause."""
        predicates = []
        for _ in range(num_predicates):
            table = random.choice(tables)
            column = self._get_random_column(table)
            template = random.choice(self.where_templates)

            # Generate appropriate value for the predicate
            value = self._generate_predicate_value(table, column, template)

            predicate = template.format(
                table=table,
                column=column,
                value=value
            )
            predicates.append(predicate)

        return " AND ".join(predicates) if predicates else ""

    def _generate_group_by_clause(self, tables: List[str], num_group_by: int) -> str:
        """Generate GROUP BY clause."""
        if num_group_by == 0:
            return ""

        columns = self._generate_columns(tables, num_group_by)
        return ", ".join(columns)

    def _get_random_column(self, table_name: str, numeric_only: bool = False) -> str:
        """Get a random column from a table."""
        col_types = self.schema.column_types[table_name]
        if numeric_only:
            columns = [c for c, t in col_types.items() if 'INT' in t.upper() or 'DECIMAL' in t.upper() or 'FLOAT' in t.upper() or 'DOUBLE' in t.upper()]
            if columns:
                return random.choice(columns)
        columns = list(col_types.keys())
        if len(columns) > 1:
            return random.choice(columns[1:])
        return columns[0]

    def _generate_predicate_value(self, table: str, column: str, template: str) -> Union[str, int]:
        """Generate appropriate value for WHERE clause predicate."""
        # This is a simplified version - in practice, you might want to
        # sample actual values from the dataset
        if "LIKE" in template:
            return f"text{random.randint(1, 100)}"
        elif ">" in template or "<" in template:
            return random.randint(1, 1000)
        elif "IS NOT NULL" in template:
            return ""
        else:
            return f"value{random.randint(1, 100)}"


class WorkloadExecutor:
    """Execute workload on DuckDB database."""

    def __init__(self, db_connection: DuckDBConnection, config: WorkloadConfig):
        self.db = db_connection
        self.config = config
        self.executor = QueryExecutor(db_connection)

    def execute_workload(self, queries: List[str]) -> List[Dict]:
        """Execute a workload and collect statistics."""
        results = []
        valid_queries = 0

        print(f"Executing {len(queries)} queries...")

        for i, query in enumerate(queries):
            print(f"Query {i+1}/{len(queries)}: ", end="")

            try:
                stats = self.executor.get_query_statistics(
                    query,
                    repetitions=3
                )

                # Check if query meets runtime criteria
                if self._is_valid_query(stats):
                    results.append(stats)
                    valid_queries += 1
                    print(f"OK ({stats['avg_time_ms']:.2f}ms)")

                    # Save intermediate results
                    if valid_queries % 50 == 0:
                        self._save_intermediate_results(results, f"workload_results_{valid_queries}.json")
                else:
                    print("SKIPPED (runtime criteria not met)")

            except Exception as e:
                print(f"ERROR: {str(e)}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })

        print(f"Successfully executed {valid_queries} valid queries")
        return results

    def _is_valid_query(self, stats: Dict) -> bool:
        """Check if query meets runtime criteria."""
        if not stats.get('successful_executions', 0) > 0:
            return False

        avg_time = stats.get('avg_time_ms', 0)
        return self.config.min_runtime_ms <= avg_time <= self.config.max_runtime_ms

    def _save_intermediate_results(self, results: List[Dict], filename: str):
        """Save intermediate results to JSON file."""
        output_path = Path("data/runs") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved {len(results)} results to {output_path}")


def generate_and_execute_workload(
    db_path: str,
    data_dir: str,
    config: WorkloadConfig,
    output_file: Optional[str] = None
) -> List[Dict]:
    """Generate and execute workload on IMDB dataset."""
    # Initialize components
    dataset = IMDBDataset(data_dir)
    db = create_database_connection(db_path)
    generator = WorkloadGenerator(dataset, config)
    executor = WorkloadExecutor(db, config)

    # Generate workload
    print("Generating workload...")
    queries = generator.generate_workload()

    # Execute workload
    results = executor.execute_workload(queries)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'config': config.__dict__,
                'results': results
            }, f, indent=2)

    db.close()
    return results