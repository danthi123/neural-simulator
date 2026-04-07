#!/usr/bin/env python
"""
Parse and analyze validation results from validate_gpu.py output.

Usage:
    python tests/parse_validation_results.py                          # Parse latest results
    python tests/parse_validation_results.py --file custom_results.json
    python tests/parse_validation_results.py --compare baseline.json current.json
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


def load_results(filepath):
    """Load validation results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def print_summary(results, verbose=False):
    """Print human-readable summary of validation results."""
    print(f"\nValidation Results")
    print(f"{'='*70}")
    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    print(f"GPU Info:  {results.get('gpu_info', 'Unknown')}")
    print(f"\n{'Test Name':<30} {'Status':<10} {'Details':<30}")
    print(f"{'-'*70}")

    passed_count = 0
    total_count = 0

    for test_name, test_result in results.get('tests', {}).items():
        passed = test_result.get('passed', False)
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"

        details = test_result.get('details', '')
        metrics = test_result.get('metrics', {})

        if passed:
            passed_count += 1
        total_count += 1

        # Create details string
        if details:
            detail_str = details[:27] + "..." if len(details) > 27 else details
        elif metrics:
            # Show first metric as summary
            first_key = list(metrics.keys())[0]
            first_val = metrics[first_key]
            detail_str = f"{first_key}: {first_val}"[:27]
        else:
            detail_str = ""

        print(f"{symbol} {test_name:<28} {status:<10} {detail_str:<30}")

    print(f"{'-'*70}")
    print(f"Summary: {passed_count}/{total_count} tests passed")
    print(f"{'='*70}\n")

    if verbose:
        print("Detailed Metrics")
        print(f"{'='*70}")
        for test_name, test_result in results.get('tests', {}).items():
            metrics = test_result.get('metrics', {})
            if metrics:
                print(f"\n{test_name}:")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
        print()

    return passed_count == total_count


def print_performance_summary(results):
    """Extract and summarize performance benchmark results."""
    perf = results.get('tests', {}).get('Performance Benchmark', {}).get('metrics', {})

    if not perf:
        print("No performance data found")
        return

    print(f"\nPerformance Benchmark Summary")
    print(f"{'='*70}")

    for model_name, configs in perf.items():
        print(f"\n{model_name}:")
        print(f"  {'Neurons':<15} {'ms/step':<15} {'Total Time':<15}")
        print(f"  {'-'*45}")

        items = sorted(configs.items(), key=lambda x: int(x[0].split('_')[0]))
        for config_key, metrics in items:
            if isinstance(metrics, dict) and 'ms_per_step' in metrics:
                neuron_count = config_key.split('_')[0]
                ms_per_step = metrics['ms_per_step']
                total_time = metrics.get('total_time_ms', 0)
                print(f"  {neuron_count:<15} {ms_per_step:<15} {total_time:.1f}ms")

    print(f"{'='*70}\n")


def compare_results(baseline_path, current_path):
    """Compare baseline and current validation results."""
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    print(f"\nValidation Comparison")
    print(f"{'='*70}")
    print(f"Baseline: {baseline.get('timestamp', 'Unknown')}")
    print(f"Current:  {current.get('timestamp', 'Unknown')}")
    print(f"\n{'Test Name':<30} {'Baseline':<15} {'Current':<15} {'Change':<10}")
    print(f"{'-'*70}")

    for test_name in baseline.get('tests', {}).keys():
        baseline_test = baseline['tests'].get(test_name, {})
        current_test = current['tests'].get(test_name, {})

        baseline_passed = baseline_test.get('passed', False)
        current_passed = current_test.get('passed', False)

        baseline_str = "PASS" if baseline_passed else "FAIL"
        current_str = "PASS" if current_passed else "FAIL"

        if baseline_passed == current_passed:
            change = "No change"
        elif current_passed and not baseline_passed:
            change = "✓ Fixed!"
        else:
            change = "✗ Regression"

        print(f"{test_name:<30} {baseline_str:<15} {current_str:<15} {change:<10}")

    # Compare performance metrics
    baseline_perf = baseline.get('tests', {}).get('Performance Benchmark', {}).get('metrics', {})
    current_perf = current.get('tests', {}).get('Performance Benchmark', {}).get('metrics', {})

    if baseline_perf and current_perf:
        print(f"\nPerformance Change (ms/step):")
        print(f"{'-'*70}")

        for model in baseline_perf.keys():
            if model in current_perf:
                print(f"\n{model}:")
                for config in baseline_perf[model].keys():
                    if config in current_perf[model]:
                        baseline_ms = baseline_perf[model][config].get('ms_per_step', 0)
                        current_ms = current_perf[model][config].get('ms_per_step', 0)

                        if baseline_ms > 0:
                            change_pct = ((current_ms - baseline_ms) / baseline_ms) * 100
                            change_str = f"{change_pct:+.1f}%"
                            symbol = "↓" if change_pct < 0 else "↑"
                        else:
                            change_str = "N/A"
                            symbol = ""

                        print(f"  {config:<20} {baseline_ms:>8.3f} → {current_ms:>8.3f} ms/step  {symbol} {change_str}")

    print(f"{'='*70}\n")


def export_csv(results, output_path):
    """Export results to CSV format."""
    try:
        with open(output_path, 'w') as f:
            f.write("Test Name,Status,Details\n")
            for test_name, test_result in results.get('tests', {}).items():
                passed = "PASS" if test_result.get('passed') else "FAIL"
                details = test_result.get('details', '').replace(',', ';')
                f.write(f'"{test_name}","{passed}","{details}"\n')
        print(f"Results exported to {output_path}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse and analyze GPU validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/parse_validation_results.py
  python tests/parse_validation_results.py --verbose
  python tests/parse_validation_results.py --file custom_results.json
  python tests/parse_validation_results.py --perf-only
  python tests/parse_validation_results.py --compare baseline.json current.json
  python tests/parse_validation_results.py --export results.csv
        """
    )

    parser.add_argument(
        '--file', '-f',
        default='tests/validation_results.json',
        help='Path to validation results JSON (default: tests/validation_results.json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed metrics'
    )
    parser.add_argument(
        '--perf-only',
        action='store_true',
        help='Print only performance benchmark results'
    )
    parser.add_argument(
        '--compare', '-c',
        nargs=2,
        metavar=('BASELINE', 'CURRENT'),
        help='Compare two validation result files'
    )
    parser.add_argument(
        '--export', '-e',
        metavar='OUTPUT',
        help='Export results to CSV file'
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results = load_results(args.file)

        if args.perf_only:
            print_performance_summary(results)
        else:
            all_passed = print_summary(results, verbose=args.verbose)

            if args.perf_only is False:  # Default behavior
                print_performance_summary(results)

            if args.export:
                export_csv(results, args.export)

            sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
