#!/usr/bin/env python3
"""Fit, freeze, evaluate, and present Simon's notebook-data showcase."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# macOS wheels can share scikit-learn's bundled OpenMP runtime. The dynamic
# loader reads this at process launch, so re-exec before importing XGBoost.
if sys.platform == 'darwin' and not os.environ.get('FECLUB_OPENMP_READY'):
    spec = importlib.util.find_spec('sklearn')
    if spec and spec.origin:
        runtime = Path(spec.origin).parent / '.dylibs'
        if (runtime / 'libomp.dylib').exists():
            environment = os.environ.copy()
            environment['DYLD_LIBRARY_PATH'] = str(runtime) + (':' + environment['DYLD_LIBRARY_PATH'] if environment.get('DYLD_LIBRARY_PATH') else '')
            environment['FECLUB_OPENMP_READY'] = '1'
            os.execve(sys.executable, [sys.executable, *sys.argv], environment)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import joblib
import numpy as np
import pandas as pd

from src.backtests.showcase import evaluate_predictions
from src.data.preparation import FEATURES
from src.data.showcase import load_showcase_data, sha256_file
from src.models.showcase import COMPONENTS, FEATURE_NAMES, train_showcase
from src.reporting.showcase import render_showcase


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')


def feature_importance(bundle):
    importance = np.zeros(len(FEATURE_NAMES), dtype=float)
    for name, weight in bundle.weights.items():
        model = bundle.models[name]
        values = np.abs(model.named_steps['elasticnet'].coef_) if name == 'elastic_net' else np.asarray(model.feature_importances_, dtype=float)
        if values.sum() > 0:
            importance += weight * values / values.sum()
    frame = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': importance})
    frame['feature'] = frame['feature'].str.replace('__csrank', '', regex=False)
    return frame.groupby('feature', as_index=False)['importance'].sum().sort_values('importance', ascending=False)


def print_result(summary, path):
    metrics = summary['evaluation']
    print(f"\nReport: {path}", flush=True)
    print(f"Selected on validation: {summary['training']['selected_weights']}", flush=True)
    print(f"Test rank IC: {metrics['rank_ic_mean']:.4f}; gross return: {metrics['gross_cumulative_return']:.2%}; net return: {metrics['net_cumulative_return']:.2%}; net Sharpe: {metrics['annualized_sharpe']:.2f}", flush=True)
    print(f"Resolved sessions: {metrics['aggregate_resolved_dates']}/{metrics['trading_dates']}; costs: {metrics['cost_bps']:g} bps per side.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true', help='Fetch a new notebook-universe snapshot and rebuild the experiment.')
    parser.add_argument('--retrain', action='store_true', help='Explicitly repeat training; repeated test viewing is recorded.')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.iterations, args.threads) < 1:
        parser.error('--iterations and --threads must be positive')
    report = ROOT / 'reports/showcase'
    model_dir = ROOT / 'models/showcase'
    processed = ROOT / 'data/processed'
    required = ['summary.json', 'leaderboard.csv', 'daily.csv', 'top_predictions.csv', 'feature_importance.csv', 'index.html']
    if not args.retrain and not args.refresh_data and all((report / name).exists() for name in required):
        previous = json.loads((report / 'summary.json').read_text())
        print('Reusing the completed frozen experiment. Use --retrain for an explicit new fit.', flush=True)
        print_result(previous, report / 'index.html')
        return
    report.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.refresh_data and all((processed / name).exists() for name in ['model_data.parquet', 'showcase_panel.parquet', 'showcase_data_manifest.json']):
        print('Loading the prepared notebook snapshot.', flush=True)
        data = pd.read_parquet(processed / 'model_data.parquet')
        panel = pd.read_parquet(processed / 'showcase_panel.parquet')
        manifest = json.loads((processed / 'showcase_data_manifest.json').read_text())
        for relative, expected in zip(manifest['raw_snapshot_paths'], [manifest['prices_sha256'], manifest['constituents_sha256']]):
            if sha256_file(ROOT / relative) != expected:
                raise ValueError('Raw snapshot checksum changed. Rebuild explicitly with --refresh-data.')
    else:
        panel, data, manifest = load_showcase_data(ROOT, refresh=args.refresh_data)
    splits = {name: data.loc[data['Split'].eq(name)].copy() for name in ['train', 'validation', 'test']}
    split_stats = {name: {'start': str(frame['Date'].min().date()), 'end': str(frame['Date'].max().date()), 'rows': int(len(frame)), 'labeled_rows': int(frame['target'].notna().sum()), 'dates': int(frame['Date'].nunique())} for name, frame in splits.items()}
    print(f"Snapshot: {len(data):,} signal rows, {manifest['tickers']} securities. Train {len(splits['train']):,}; validation {len(splits['validation']):,}; test {len(splits['test']):,}.", flush=True)
    fitted_path = model_dir / 'bundle.joblib'
    previous_evaluations = []
    if (report / 'summary.json').exists():
        old = json.loads((report / 'summary.json').read_text())
        previous_evaluations = old.get('previous_evaluations', []) + [{'experiment_id': old['experiment_id'], 'generated_at': old['generated_at'], 'dataset_sha256': old['dataset']['sha256']}]
    if fitted_path.exists() and not args.retrain and not args.refresh_data:
        bundle = joblib.load(fitted_path)
        training = bundle.training_metadata
        if training.get('dataset_sha256') != manifest['sha256']:
            raise ValueError('Frozen model and data snapshot differ. Use --retrain explicitly.')
        print('Resuming the saved, validation-selected model without refitting.', flush=True)
    else:
        print(f'Fitting four fixed models with a {args.iterations}-iteration maximum; only 2023 selects the blend.', flush=True)
        bundle, training = train_showcase(splits['train'], splits['validation'], seed=args.seed, threads=args.threads, max_iterations=args.iterations)
        training.update({'dataset_sha256': manifest['sha256'], 'max_iterations': args.iterations, 'frozen_at': datetime.now(ZoneInfo('America/Chicago')).isoformat()})
        bundle.training_metadata = training
        bundle.save(model_dir)
        write_json(report / 'training.json', training)
        print(f"Frozen blend: {bundle.weights}; iterations: {training['best_iterations']}. Evaluating test now.", flush=True)
    # Keep the complete observable population, including rows with no target.
    predictions = bundle.predict_all(splits['test'])
    predictions.to_parquet(report / 'predictions.parquet', index=False)
    leaderboard, evaluations = [], {}
    selected_daily = None
    for name in ['ensemble', *COMPONENTS, 'reversal', 'momentum']:
        print(f'Evaluating {name} on 2024–2025.', flush=True)
        evaluation, daily, trades = evaluate_predictions(panel, predictions[['Date', 'Ticker', name]], prediction_column=name, top_k=10, cost_bps=10.0, start_date='2024-01-01')
        evaluations[name] = evaluation
        rmse = np.nan
        if name == 'ensemble':
            labels = splits['test'][['Date', 'Ticker', 'target']].merge(predictions[['Date', 'Ticker', 'ensemble_predicted_return']], on=['Date', 'Ticker'], validate='one_to_one').dropna(subset=['target'])
            rmse = float(np.sqrt(np.mean((labels['target'] - labels['ensemble_predicted_return']) ** 2)))
            selected_daily = daily.copy()
            selected_daily['Date'] = selected_daily['HoldingDate']
            trades.to_parquet(report / 'trades.parquet', index=False)
        leaderboard.append({'model': name, 'rank_ic': evaluation['rank_ic_mean'], 'rank_ic_days': evaluation['rank_ic_scored_dates'], 'rmse_if_return_prediction': rmse, 'cumulative_gross_return': evaluation['gross_cumulative_return'], 'cumulative_net_return': evaluation['net_cumulative_return'], 'sharpe_net': evaluation['annualized_sharpe'], 'max_drawdown_net': evaluation['max_drawdown'], 'resolved_days': evaluation['aggregate_resolved_dates']})
        del trades
    board = pd.DataFrame(leaderboard)
    selected_daily.to_csv(report / 'daily.csv', index=False)
    board.to_csv(report / 'leaderboard.csv', index=False)
    latest = predictions.loc[predictions['Date'].eq(predictions['Date'].max())].copy()
    latest['disagreement'] = latest[list(COMPONENTS)].std(axis=1)
    latest = latest.sort_values(['ensemble', 'Ticker'], ascending=[False, True], kind='stable').head(10)
    latest['rank'] = np.arange(1, len(latest) + 1)
    latest = latest.rename(columns={'ensemble': 'score', 'ensemble_predicted_return': 'prediction'})[['Date', 'Ticker', 'rank', 'score', 'prediction', 'disagreement']]
    latest.to_csv(report / 'top_predictions.csv', index=False)
    importance = feature_importance(bundle)
    importance.to_csv(report / 'feature_importance.csv', index=False)
    packages = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'yfinance', 'matplotlib']}
    source_files = ['src/data/preparation.py', 'src/data/showcase.py', 'src/models/showcase.py', 'src/backtests/showcase.py', 'scripts/run_showcase.py']
    source_hashes = {name: sha256_file(ROOT / name) for name in source_files}
    experiment_id = hashlib.sha256(json.dumps({'dataset': manifest['sha256'], 'seed': training['seed'], 'iterations': training['max_iterations'], 'source': source_hashes}, sort_keys=True).encode()).hexdigest()[:12]
    summary = {
        'experiment_id': experiment_id, 'generated_at': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'dataset': manifest, 'splits': split_stats, 'training': training,
        'evaluation': evaluations['ensemble'], 'comparison_evaluations': evaluations,
        'runtime': {'python': sys.version, 'packages': packages, 'source_sha256': source_hashes},
        'previous_evaluations': previous_evaluations,
        'protocol': {
            'model_selection': 'Four fixed model configurations; 2023 RMSE/NDCG chooses stopping iterations, then daily rank IC selects among 12 fixed blends. The selected model is saved before test prediction.',
            'test_policy': 'First recorded fit. No test-based retuning.' if not previous_evaluations else 'A test was viewed in a previous run; this is a repeated experiment, not a new untouched holdout.',
            'predictors': 'Original nine notebook features and their same-date target-free percentile transforms only.',
            'execution': 'Rank after close, top 10 stable ticker ties, equal capital slots, next-session open to close, cash overnight.',
            'cost': '10 bps each buy/sell; buy cost funded within each slot, no leverage; cash earns zero.',
            'signal_units': 'ensemble is a ranking score, not a return. prediction in the latest ranking is a separate 2023-calibrated decimal next-day return estimate.',
            'importance': 'Validation-blend-weighted normalized native importance (absolute standardized coefficients for linear model), combined across raw/rank versions; descriptive, not causal.',
        },
        'limitations': [
            'The Wikipedia universe is current at download time; historical results have survivorship and selection bias.',
            'The Yahoo Finance snapshot is not a point-in-time institutional dataset; adjusted history can be revised.',
            'Only the nine original price/volume features and simple observable cross-sectional transforms are used; this is a technical demo, not a novel research contribution.',
            'Forecast labels include overnight returns; executable portfolio returns cover only next-session open to close.',
            'Top-10 daily turnover is costly. Commission assumptions omit market impact, volume participation limits, and spread variation.',
            'Unresolved entries/exits are explicit. Aggregate results use common resolved sessions and are conditional if any are omitted.',
            '2023 selects stopping/blending/calibration. Validation performance is optimistic; paper research needs an independent confirmation sample.',
            'Latest rankings are from the end of the historical snapshot and have no observed next session; they are not live stock recommendations.',
            'Backtest profit is measured, not promised. Ultimate goal remains a publishable paper after literature mapping, a defensible question, stronger data, and robust validation.',
        ],
    }
    write_json(report / 'summary.json', summary)
    path = render_showcase(report, clean_json(summary), board, selected_daily, latest, importance)
    print_result(summary, path)


if __name__ == '__main__':
    main()
