"""Build the expanded research inputs from the existing notebook snapshot."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.alpha_features import ALPHA_FEATURES, ALPHA_FEATURE_METADATA, add_alpha_features
from src.data.preparation import FEATURES, add_date_splits, make_model_dataset
from src.data.showcase import load_showcase_data, sha256_file

ALPHA_INPUTS = [*FEATURES, *ALPHA_FEATURES]
ALPHA_PATHS = ['alpha_model_data.parquet', 'alpha_panel.parquet', 'alpha_data_manifest.json']


def _verify_raw(root, manifest):
    for relative, expected in zip(manifest['raw_snapshot_paths'], [manifest['prices_sha256'], manifest['constituents_sha256']]):
        if sha256_file(root / relative) != expected:
            raise ValueError('The saved raw snapshot changed. Restore it or explicitly refresh data.')


def load_alpha_data(root: Path, *, rebuild=False, refresh=False):
    """Keep all calendar quotes, with eligibility based only on complete inputs."""
    processed = root / 'data/processed'
    source_hash = sha256_file(root / 'src/data/alpha_features.py')
    if not rebuild and not refresh and all((processed / name).exists() for name in ALPHA_PATHS):
        manifest = json.loads((processed / ALPHA_PATHS[2]).read_text())
        if manifest.get('feature_definition_sha256') != source_hash:
            raise ValueError('Alpha feature code changed. Use --rebuild-features --retrain.')
        _verify_raw(root, manifest)
        print('Loading the prepared research feature snapshot.', flush=True)
        return pd.read_parquet(processed / ALPHA_PATHS[1]), pd.read_parquet(processed / ALPHA_PATHS[0]), manifest
    base_panel = processed / 'showcase_panel.parquet'
    base_manifest = processed / 'showcase_data_manifest.json'
    if refresh or not base_panel.exists() or not base_manifest.exists():
        panel, _, original = load_showcase_data(root, refresh=refresh)
    else:
        panel = pd.read_parquet(base_panel)
        original = json.loads(base_manifest.read_text())
        _verify_raw(root, original)
        baseline_report = root / 'reports/showcase/summary.json'
        if baseline_report.exists():
            frozen = json.loads(baseline_report.read_text())
            for relative in ['data/processed/showcase_panel.parquet', 'data/processed/showcase_data_manifest.json']:
                expected = frozen.get('prepared_sha256', {}).get(relative)
                if expected and sha256_file(root / relative) != expected:
                    raise ValueError('Notebook preparation changed; restore the frozen tables or explicitly refresh data.')
    print('Building the six research feature families from saved prices; no new download.', flush=True)
    panel = add_alpha_features(panel)
    panel['NotebookSignalEligible'] = panel['SignalEligible']
    panel['SignalEligible'] = panel['AlphaSignalEligible']
    data = add_date_splits(make_model_dataset(panel, labeled_only=False))
    data = data[['Date', 'Ticker', 'Split', 'LabelEndDate', *ALPHA_INPUTS, 'target', 'target_rank']]
    if not {'train', 'validation', 'test'}.issubset(set(data['Split'])):
        raise ValueError('Expanded features must cover each chronological period.')
    manifest = dict(original)
    manifest.update({
        'feature_set': 'alpha', 'raw_snapshot_sha256': original['sha256'],
        'sha256': hashlib.sha256(f"{original['sha256']}:{source_hash}".encode()).hexdigest(),
        'feature_definition_sha256': source_hash, 'features': ALPHA_INPUTS,
        'added_features': ALPHA_FEATURES, 'feature_metadata': ALPHA_FEATURE_METADATA,
        'feature_prepared_at': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'rows': int(len(data)), 'tickers': int(data['Ticker'].nunique()),
        'feature_start': str(data['Date'].min().date()),
        'feature_warmup_sessions': 252,
        'notebook_signal_rows': int(panel['NotebookSignalEligible'].sum()),
        'alpha_signal_rows': int(panel['SignalEligible'].sum()),
        'rows_excluded_for_expanded_features': int((panel['NotebookSignalEligible'] & ~panel['SignalEligible']).sum()),
        'feature_policy': 'Complete trailing windows, no price-gap filling; only information observed by the signal close. Raw snapshot is unchanged.',
    })
    processed.mkdir(parents=True, exist_ok=True)
    data.to_parquet(processed / ALPHA_PATHS[0], index=False)
    panel.to_parquet(processed / ALPHA_PATHS[1], index=False)
    (processed / ALPHA_PATHS[2]).write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(f"Prepared {len(data):,} research signal rows with {len(ALPHA_INPUTS)} raw inputs.", flush=True)
    return panel, data, manifest
