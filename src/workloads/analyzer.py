"""
Workload validation and analysis utilities.
"""
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class WorkloadAnalyzer:
    """Analyze and validate generated workloads."""

    def __init__(self, workload_results: List[Dict]):
        self.results = workload_results
        self.valid_results = [r for r in workload_results if r.get('successful_executions', 0) > 0]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the workload."""
        if not self.valid_results:
            return {"error": "No valid queries found"}

        execution_times = [r['avg_time_ms'] for r in self.valid_results]

        return {
            "total_queries": len(self.results),
            "valid_queries": len(self.valid_results),
            "success_rate": len(self.valid_results) / len(self.results) * 100,
            "execution_times": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": sum(execution_times) / len(execution_times),
                "median": sorted(execution_times)[len(execution_times) // 2],
                "std": pd.Series(execution_times).std()
            },
            "query_complexity": {
                "avg_joins": self._calculate_avg_joins(),
                "avg_predicates": self._calculate_avg_predicates()
            }
        }

    def _calculate_avg_joins(self) -> float:
        """Calculate average number of joins in queries."""
        join_counts = []
        for result in self.valid_results:
            query = result['query'].upper()
            join_count = query.count('JOIN')
            join_counts.append(join_count)
        return sum(join_counts) / len(join_counts) if join_counts else 0

    def _calculate_avg_predicates(self) -> float:
        """Calculate average number of WHERE predicates."""
        predicate_counts = []
        for result in self.valid_results:
            query = result['query'].upper()
            where_count = query.count('WHERE')
            predicate_counts.append(where_count)
        return sum(predicate_counts) / len(predicate_counts) if predicate_counts else 0

    def plot_execution_times(self, save_path: str = None):
        """Plot distribution of execution times."""
        if not self.valid_results:
            print("No valid queries to plot")
            return

        execution_times = [r['avg_time_ms'] for r in self.valid_results]

        plt.figure(figsize=(10, 6))
        sns.histplot(execution_times, bins=30, kde=True)
        plt.title('Distribution of Query Execution Times')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Frequency')
        plt.yscale('log')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_query_complexity(self, save_path: str = None):
        """Plot query complexity metrics."""
        if not self.valid_results:
            print("No valid queries to plot")
            return

        join_counts = []
        predicate_counts = []

        for result in self.valid_results:
            query = result['query'].upper()
            join_count = query.count('JOIN')
            where_count = query.count('WHERE')
            join_counts.append(join_count)
            predicate_counts.append(where_count)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot join counts
        join_series = pd.Series(join_counts)
        join_series.value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_title('Distribution of JOINs per Query')
        ax1.set_xlabel('Number of JOINs')
        ax1.set_ylabel('Frequency')

        # Plot predicate counts
        pred_series = pd.Series(predicate_counts)
        pred_series.value_counts().sort_index().plot(kind='bar', ax=ax2)
        ax2.set_title('Distribution of WHERE Predicates per Query')
        ax2.set_xlabel('Number of WHERE Predicates')
        ax2.set_ylabel('Frequency')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def validate_workload(self, min_queries: int = 50, min_runtime: float = 10.0) -> Dict[str, Any]:
        """Validate workload quality."""
        validation = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }

        # Check minimum number of valid queries
        if len(self.valid_results) < min_queries:
            validation["valid"] = False
            validation["issues"].append(f"Too few valid queries: {len(self.valid_results)} < {min_queries}")

        # Check minimum average runtime
        if self.valid_results:
            avg_runtime = sum(r['avg_time_ms'] for r in self.valid_results) / len(self.valid_results)
            if avg_runtime < min_runtime:
                validation["valid"] = False
                validation["issues"].append(f"Average runtime too low: {avg_runtime:.2f}ms < {min_runtime}ms")

        # Check query diversity
        unique_queries = len(set(r['query'] for r in self.valid_results))
        if unique_queries < len(self.valid_results) * 0.8:
            validation["recommendations"].append("Consider increasing query diversity")

        # Check for timeout issues
        timeouts = sum(1 for r in self.results if r.get('error', '').lower().find('timeout') != -1)
        if timeouts > len(self.results) * 0.1:
            validation["recommendations"].append("Many timeouts detected - consider reducing query complexity")

        return validation

    def export_workload_summary(self, output_file: str):
        """Export workload summary to file."""
        summary = {
            "summary_stats": self.get_summary_stats(),
            "validation": self.validate_workload()
        }

        import json
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Workload summary exported to {output_file}")


def analyze_workload(workload_results: List[Dict], output_dir: str = "output/"):
    """Analyze workload and generate reports."""
    analyzer = WorkloadAnalyzer(workload_results)

    # Print summary
    summary = analyzer.get_summary_stats()
    print("Workload Summary:")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Valid queries: {summary['valid_queries']}")
    print(f"  Success rate: {summary['success_rate']:.2f}%")
    print(f"  Average execution time: {summary['execution_times']['mean']:.2f}ms")

    # Validate workload
    validation = analyzer.validate_workload()
    print("\nValidation Results:")
    print(f"  Valid: {validation['valid']}")
    if validation['issues']:
        print("  Issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    if validation['recommendations']:
        print("  Recommendations:")
        for rec in validation['recommendations']:
            print(f"    - {rec}")

    # Generate plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer.plot_execution_times(output_dir / "execution_times.png")
    analyzer.plot_query_complexity(output_dir / "query_complexity.png")
    analyzer.export_workload_summary(output_dir / "workload_summary.json")