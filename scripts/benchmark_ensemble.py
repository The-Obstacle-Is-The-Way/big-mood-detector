#!/usr/bin/env python3
"""
Benchmark ensemble prediction latency.

Measures p50, p95, p99 latencies for ensemble predictions.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from rich.console import Console
from rich.table import Table

from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor

console = Console()


def generate_test_data():
    """Generate test data for benchmarking."""
    # Features
    features = np.random.randn(36).astype(np.float32)
    
    # Activity records for PAT
    records = []
    base_date = date.today()
    for hour in range(24 * 7):  # 7 days
        start_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour)
        records.append(
            ActivityRecord(
                source_name="benchmark_test",
                start_date=start_time,
                end_date=start_time + timedelta(hours=1),
                activity_type=ActivityType.STEP_COUNT,
                value=500 + 300 * np.sin(hour / 24 * 2 * np.pi),  # Simulated step count
                unit="count",
            )
        )
    
    return features, records


def benchmark_predictions(orchestrator, features, activity_records, n_runs=100):
    """Run predictions and measure latency."""
    latencies = []
    
    # Warmup
    console.print("[yellow]Warming up...[/yellow]")
    for _ in range(10):
        orchestrator.predict(features, activity_records)
    
    # Benchmark
    console.print(f"[green]Running {n_runs} predictions...[/green]")
    
    with console.status("[bold blue]Benchmarking..."):
        for i in range(n_runs):
            start = time.perf_counter()
            result = orchestrator.predict(
                statistical_features=features,
                activity_records=activity_records,
                prediction_date=None,
            )
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 10 == 0:
                console.print(f"  Progress: {i + 1}/{n_runs}")
    
    return latencies, result


def calculate_percentiles(latencies):
    """Calculate percentile statistics."""
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "min": sorted_latencies[0],
        "p50": sorted_latencies[int(n * 0.50)],
        "p90": sorted_latencies[int(n * 0.90)],
        "p95": sorted_latencies[int(n * 0.95)],
        "p99": sorted_latencies[int(n * 0.99)],
        "max": sorted_latencies[-1],
        "mean": mean(sorted_latencies),
        "stdev": stdev(sorted_latencies) if n > 1 else 0,
    }


def main():
    """Run the benchmark."""
    console.print("[bold cyan]Big Mood Detector - Ensemble Prediction Benchmark[/bold cyan]\n")
    
    # Initialize models
    console.print("Loading models...")
    
    xgboost_predictor = XGBoostMoodPredictor()
    if not xgboost_predictor.load_models(Path("model_weights/xgboost/pretrained")):
        console.print("[red]Failed to load XGBoost models![/red]")
        return
    
    pat_model = None
    if PAT_AVAILABLE:
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
        
        pat_model = PATModel(model_size="medium")
        if pat_model.load_pretrained_weights():
            console.print("[green]✓ PAT model loaded[/green]")
        else:
            console.print("[yellow]⚠ PAT weights not found[/yellow]")
            pat_model = None
    else:
        console.print("[yellow]⚠ TensorFlow not installed - PAT unavailable[/yellow]")
    
    # Create orchestrator
    config = EnsembleConfig()
    orchestrator = EnsembleOrchestrator(
        xgboost_predictor=xgboost_predictor,
        pat_model=pat_model,
        config=config,
    )
    
    # Generate test data
    features, activity_records = generate_test_data()
    
    # Run benchmarks
    console.print("\n[bold]Benchmark Results:[/bold]")
    
    # 1. XGBoost only
    orchestrator_xgb = EnsembleOrchestrator(
        xgboost_predictor=xgboost_predictor,
        pat_model=None,
        config=config,
    )
    
    latencies_xgb, _ = benchmark_predictions(
        orchestrator_xgb, features, None, n_runs=100
    )
    stats_xgb = calculate_percentiles(latencies_xgb)
    
    # 2. Full ensemble (if PAT available)
    if pat_model:
        latencies_ensemble, result = benchmark_predictions(
            orchestrator, features, activity_records, n_runs=100
        )
        stats_ensemble = calculate_percentiles(latencies_ensemble)
    
    # Display results
    table = Table(title="Latency Statistics (milliseconds)")
    table.add_column("Model", style="cyan")
    table.add_column("Min", justify="right")
    table.add_column("P50", justify="right", style="bold")
    table.add_column("P90", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right", style="yellow")
    table.add_column("Max", justify="right")
    table.add_column("Mean ± SD", justify="right")
    
    # XGBoost row
    table.add_row(
        "XGBoost Only",
        f"{stats_xgb['min']:.1f}",
        f"{stats_xgb['p50']:.1f}",
        f"{stats_xgb['p90']:.1f}",
        f"{stats_xgb['p95']:.1f}",
        f"{stats_xgb['p99']:.1f}",
        f"{stats_xgb['max']:.1f}",
        f"{stats_xgb['mean']:.1f} ± {stats_xgb['stdev']:.1f}",
    )
    
    # Ensemble row (if available)
    if pat_model:
        table.add_row(
            "Ensemble (XGB+PAT)",
            f"{stats_ensemble['min']:.1f}",
            f"{stats_ensemble['p50']:.1f}",
            f"{stats_ensemble['p90']:.1f}",
            f"{stats_ensemble['p95']:.1f}",
            f"{stats_ensemble['p99']:.1f}",
            f"{stats_ensemble['max']:.1f}",
            f"{stats_ensemble['mean']:.1f} ± {stats_ensemble['stdev']:.1f}",
        )
    
    console.print(table)
    
    # Performance summary
    console.print("\n[bold]Performance Summary:[/bold]")
    if stats_xgb['p99'] < 100:
        console.print("[green]✓ XGBoost P99 < 100ms target[/green]")
    else:
        console.print("[red]✗ XGBoost P99 > 100ms target[/red]")
    
    if pat_model:
        if stats_ensemble['p99'] < 200:
            console.print("[green]✓ Ensemble P99 < 200ms target[/green]")
        else:
            console.print("[red]✗ Ensemble P99 > 200ms target[/red]")
        
        overhead = stats_ensemble['p50'] - stats_xgb['p50']
        console.print(f"\nPAT overhead (P50): {overhead:.1f}ms")
    
    # Model info
    if pat_model and result:
        console.print(f"\nModels used: {', '.join(result.models_used)}")
        console.print("Confidence scores:")
        for model, conf in result.confidence_scores.items():
            console.print(f"  {model}: {conf:.2f}")


if __name__ == "__main__":
    main()