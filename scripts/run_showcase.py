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
import shutil
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

from src.backtests.showcase import score_predictions
from src.backtests.rank_hold import evaluate_rank_hold
from src.data.rank_hold import load_rank_hold_prices
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
    print(f"Initial capital: ${metrics['initial_capital']:,.0f}; ending net equity: ${metrics['final_net_equity']:,.0f}.", flush=True)
    print(f"Resolved sessions: {metrics['aggregate_resolved_dates']}/{metrics['trading_dates']}; costs: {metrics['cost_bps']:g} bps per side.", flush=True)


def verify_frozen_files(summary):
    """Check the saved report against its raw, prepared, model and forecast inputs."""
    manifest = summary['dataset']
    checks = dict(zip(manifest['raw_snapshot_paths'], [manifest['prices_sha256'], manifest['constituents_sha256']]))
    checks.update(summary.get('prepared_sha256', {}))
    if summary.get('model_sha256'):
        checks['models/showcase/bundle.joblib'] = summary['model_sha256']
    if summary.get('predictions_sha256'):
        checks['reports/showcase/predictions.parquet'] = summary['predictions_sha256']
    for relative, expected in checks.items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f'Frozen artifact changed: {relative}. Restore it or explicitly rebuild.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true', help='Fetch a new notebook-universe snapshot and rebuild the experiment.')
    parser.add_argument('--retrain', action='store_true', help='Explicitly repeat training; repeated test viewing is recorded.')
    parser.add_argument('--rebacktest', action='store_true', help='Recompute execution using the frozen model and predictions.')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.iterations, args.threads) < 1:
        parser.error('--iterations and --threads must be positive')
    report = ROOT / 'reports/showcase'
    model_dir = ROOT / 'models/showcase'
    processed = ROOT / 'data/processed'
    config_path = ROOT / 'config/rank_hold.json'
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    source_files = ['src/data/preparation.py', 'src/data/showcase.py', 'src/data/rank_hold.py',
                    'src/models/showcase.py', 'src/backtests/showcase.py', 'src/backtests/rank_hold.py',
                    'scripts/run_showcase.py', 'src/reporting/showcase.py']
    source_hashes = {name: sha256_file(ROOT / name) for name in source_files}
    required = ['summary.json', 'leaderboard.csv', 'daily.csv', 'top_predictions.csv', 'feature_importance.csv', 'index.html']
    old = json.loads((report / 'summary.json').read_text()) if (report / 'summary.json').exists() else None
    if not any([args.retrain, args.refresh_data, args.rebacktest]) and all((report / name).exists() for name in required):
        if old.get('strategy_config_sha256') == config_hash and old.get('runtime', {}).get('source_sha256') == source_hashes:
            verify_frozen_files(old)
            print('Reusing the completed frozen experiment. Use --rebacktest for new execution or --retrain for an explicit new fit.', flush=True)
            print_result(old, report / 'index.html')
            return
    report.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if old and not args.refresh_data:
        previous_sources = old.get('runtime', {}).get('source_sha256', {})
        for name in ['src/data/preparation.py', 'src/data/showcase.py']:
            if name in previous_sources and previous_sources[name] != source_hashes[name]:
                raise ValueError('Feature preparation code changed. Rebuild explicitly with --refresh-data.')
        if not args.retrain:
            verify_frozen_files(old)
            if previous_sources.get('src/models/showcase.py', source_hashes['src/models/showcase.py']) != source_hashes['src/models/showcase.py']:
                raise ValueError('Model code changed. Refit explicitly with --retrain.')
    if old and old.get('evaluation', {}).get('strategy_name') != config['strategy_name']:
        archive = report / 'archive/intraday-v1'
        archive.mkdir(parents=True, exist_ok=True)
        for name in [*required, 'RESULTS.md', 'training.json']:
            if (report / name).exists() and not (archive / name).exists():
                shutil.copy2(report / name, archive / name)
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
    prepared_hashes = {f'data/processed/{name}': sha256_file(processed / name) for name in ['model_data.parquet', 'showcase_panel.parquet', 'showcase_data_manifest.json']}
    if old and not args.refresh_data:
        for name, expected in old.get('prepared_sha256', {}).items():
            if prepared_hashes.get(name) != expected:
                raise ValueError('Prepared data changed. Rebuild explicitly with --refresh-data.')
    splits = {name: data.loc[data['Split'].eq(name)].copy() for name in ['train', 'validation', 'test']}
    split_stats = {name: {'start': str(frame['Date'].min().date()), 'end': str(frame['Date'].max().date()), 'rows': int(len(frame)), 'labeled_rows': int(frame['target'].notna().sum()), 'dates': int(frame['Date'].nunique())} for name, frame in splits.items()}
    print(f"Snapshot: {len(data):,} signal rows, {manifest['tickers']} securities. Train {len(splits['train']):,}; validation {len(splits['validation']):,}; test {len(splits['test']):,}.", flush=True)
    fitted_path = model_dir / 'bundle.joblib'
    previous_evaluations = []
    if old:
        previous_evaluations = old.get('previous_evaluations', []) + [{
            'experiment_id': old['experiment_id'], 'generated_at': old['generated_at'],
            'dataset_sha256': old['dataset']['sha256'],
            'strategy': old.get('evaluation', {}).get('strategy_name', 'top10_intraday'),
            'net_cumulative_return': old['evaluation']['net_cumulative_return'],
        }]
    reused_model = fitted_path.exists() and not args.retrain and not args.refresh_data
    if reused_model:
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
        print(f"Frozen blend: {bundle.weights}; iterations: {training['best_iterations']}. Evaluating test now.", flush=True)
    write_json(report / 'training.json', training)
    model_hash = sha256_file(fitted_path)
    prediction_path = report / 'predictions.parquet'
    if reused_model and prediction_path.exists() and old and old['dataset']['sha256'] == manifest['sha256']:
        if old.get('model_sha256', model_hash) != model_hash:
            raise ValueError('Saved model changed since cached prediction. Repeat explicitly with --retrain.')
        if old.get('predictions_sha256') and sha256_file(prediction_path) != old['predictions_sha256']:
            raise ValueError('Cached forecasts changed; restore the frozen table or explicitly refit.')
        print('Reusing the complete frozen forecast table; trading rules do not refit the model.', flush=True)
        predictions = pd.read_parquet(prediction_path)
    else:
        predictions = bundle.predict_all(splits['test'])
        predictions.to_parquet(prediction_path, index=False)
    quote_panel = load_rank_hold_prices(panel, ROOT)
    parameters = {name: config[name] for name in ['initial_capital', 'entry_k', 'exit_k', 'long_fraction', 'cost_bps', 'allow_additions', 'exit_unranked', 'start_date']}
    parameters.update(annual_borrow_bps=config['borrow_fee_bps_annual'], price_units=config['execution_price'])
    leaderboard, evaluations = [], {}
    selected_daily = None
    for name in ['ensemble', *COMPONENTS, 'reversal', 'momentum']:
        print(f'Evaluating {name}: persistent top20 / bottom20 on 2024–2025.', flush=True)
        ic_summary, ic_daily = score_predictions(panel, predictions[['Date', 'Ticker', name]], prediction_column=name, start_date=config['start_date'])
        try:
            evaluation, daily, trades, positions = evaluate_rank_hold(quote_panel, predictions[['Date', 'Ticker', name]], prediction_column=name, **parameters)
        except ValueError as error:
            # Missing held quotes invalidate a comparison, never the selected
            # headline. Do not drop its security or invent a flat return.
            if name == 'ensemble' or not str(error).startswith('Unresolved held mark:'):
                raise
            print(f'UNRESOLVED {name}: {error}', flush=True)
            evaluation = {
                'strategy_name': config['strategy_name'], 'status': 'unresolved',
                'failure': str(error), 'quote_resolution_complete': False,
                'fully_resolved': False, 'performance_valid': False,
                'initial_capital': config['initial_capital'],
                'final_gross_equity': np.nan, 'final_net_equity': np.nan,
                'gross_cumulative_return': np.nan, 'net_cumulative_return': np.nan,
                'annualized_sharpe': np.nan, 'max_drawdown': np.nan,
                'aggregate_resolved_dates': 0,
            }
            daily, trades, positions = None, None, None
        evaluation.update(ic_summary)
        evaluations[name] = evaluation
        rmse = np.nan
        if name == 'ensemble':
            labels = splits['test'][['Date', 'Ticker', 'target']].merge(predictions[['Date', 'Ticker', 'ensemble_predicted_return']], on=['Date', 'Ticker'], validate='one_to_one').dropna(subset=['target'])
            rmse = float(np.sqrt(np.mean((labels['target'] - labels['ensemble_predicted_return']) ** 2)))
            selected_daily = daily.merge(ic_daily[['Date', 'rank_ic']], on='Date', how='left', validate='one_to_one')
            quotes = quote_panel[['Date', 'Ticker', 'RawExecutionAverage', 'AdjustmentFactor']].rename(columns={'Date': 'ExecutionDate'})
            trades = trades.merge(quotes, on=['ExecutionDate', 'Ticker'], how='left', validate='many_to_one')
            trades['raw_share_equivalent_at_fill'] = trades['quantity'] * trades['AdjustmentFactor']
            trades.to_parquet(report / 'trades.parquet', index=False)
            positions.to_parquet(report / 'positions.parquet', index=False)
        leaderboard.append({'model': name, 'rank_ic': evaluation['rank_ic_mean'], 'rank_ic_days': evaluation['rank_ic_scored_dates'], 'rmse_if_return_prediction': rmse, 'initial_capital': evaluation['initial_capital'], 'final_gross_equity': evaluation['final_gross_equity'], 'final_net_equity': evaluation['final_net_equity'], 'cumulative_gross_return': evaluation['gross_cumulative_return'], 'cumulative_net_return': evaluation['net_cumulative_return'], 'sharpe_net': evaluation['annualized_sharpe'], 'max_drawdown_net': evaluation['max_drawdown'], 'resolved_days': evaluation['aggregate_resolved_dates'], 'status': evaluation.get('status', 'resolved'), 'failure': evaluation.get('failure', '')})
        del trades, positions
    board = pd.DataFrame(leaderboard)
    selected_daily.to_csv(report / 'daily.csv', index=False)
    board.to_csv(report / 'leaderboard.csv', index=False)
    latest = predictions.loc[predictions['Date'].eq(predictions['Date'].max())].copy()
    latest['disagreement'] = latest[list(COMPONENTS)].std(axis=1)
    latest = latest.sort_values(['ensemble', 'Ticker'], ascending=[False, True], kind='stable').head(config['entry_k'])
    latest['rank'] = np.arange(1, len(latest) + 1)
    latest = latest.rename(columns={'ensemble': 'score', 'ensemble_predicted_return': 'prediction'})[['Date', 'Ticker', 'rank', 'score', 'prediction', 'disagreement']]
    latest.to_csv(report / 'top_predictions.csv', index=False)
    importance = feature_importance(bundle)
    importance.to_csv(report / 'feature_importance.csv', index=False)
    packages = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'yfinance', 'matplotlib']}
    experiment_id = hashlib.sha256(json.dumps({'dataset': manifest['sha256'], 'model': model_hash, 'config': config_hash, 'source': source_hashes}, sort_keys=True).encode()).hexdigest()[:12]
    summary = {
        'experiment_id': experiment_id, 'generated_at': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'dataset': manifest, 'prepared_sha256': prepared_hashes, 'splits': split_stats, 'training': training,
        'strategy_config': config, 'strategy_config_sha256': config_hash,
        'model_sha256': model_hash, 'predictions_sha256': sha256_file(prediction_path),
        'evaluation': evaluations['ensemble'], 'comparison_evaluations': evaluations,
        'runtime': {'python': sys.version, 'packages': packages, 'source_sha256': source_hashes},
        'previous_evaluations': previous_evaluations,
        'protocol': {
            'model_selection': 'Four fixed model configurations; 2023 RMSE/NDCG chooses stopping iterations, then daily rank IC selects among 12 fixed blends. The selected model is saved before test prediction.',
            'test_policy': 'User-defined strategy revision on a previously viewed 2024–2025 sample: exploratory. Saved model and forecast table remain frozen when only execution changes.',
            'predictors': 'Original nine notebook features and their same-date target-free percentile transforms only.',
            'execution': 'Completed close ranks select next-session top20 long and bottom20 short additions; exits when prior-close own-side rank exceeds100 or is unranked. OHLC4 fills; adjusted-close marks. End positions remain open.',
            'capital': config['capital_policy'],
            'cost': '10 bps each entry/exit, funded within budgets; cash earns zero. Independent gross/no-cost and net/cost books. Borrow cost defaults to zero; shortability assumed.',
            'price_basis': 'Raw OHLC4 times Adj Close/Close and adjusted-close marks simulate total-return units. Fixed units embed corporate actions; no extra dividends or split credits. Units are not historical share counts.',
            'signal_units': 'ensemble is a ranking score, not a return. prediction in the latest ranking is a separate 2023-calibrated decimal next-day return estimate.',
            'importance': 'Validation-blend-weighted normalized native importance (absolute standardized coefficients for linear model), combined across raw/rank versions; descriptive, not causal.',
        },
        'limitations': [
            'The Wikipedia universe is current at download time; historical results have survivorship and future-membership selection bias.',
            'The Yahoo Finance snapshot is not point-in-time institutional data; adjusted history can be revised.',
            'OHLC4 is a hypothetical full-day execution proxy, not VWAP or a guaranteed attainable fill. Orders use only prior-session information.',
            'Adjusted total-return units approximate corporate actions; actual dividend cash timing, short dividend obligations and verified split-share inventories are not modeled.',
            'Short-sale proceeds and equal entry collateral are reserved; real maintenance margin, stock-loan availability, recalls and borrow fees are not modeled.',
            'The fixed next-day forecast target and a multi-day ranking exit strategy have different holding horizons.',
            '10 bps costs omit market impact, volume participation limits and spread variation. Ending equity includes open positions without terminal liquidation fees.',
            'Only the nine original features and simple observable transforms are used; this is a technical demo, not a novel research contribution.',
            '2023 selects stopping/blending/calibration. Validation is optimistic; repeated test development is exploratory and needs independent paper confirmation.',
            'Latest rankings are historical and have no observed next session; they are not live recommendations.',
            'Comparisons with missing held marks have no portfolio performance estimate; forecast IC remains separately measurable.',
        ],
    }
    write_json(report / 'summary.json', summary)
    path = render_showcase(report, clean_json(summary), board, selected_daily, latest, importance)
    print_result(summary, path)


if __name__ == '__main__':
    main()
