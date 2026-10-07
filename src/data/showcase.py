"""Download and cache exactly the stock universe/date range used by Data.ipynb."""
from __future__ import annotations

from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.data.preparation import (
    FEATURES, QUOTE_COLUMNS, add_date_splits, make_model_dataset, prepare_price_panel,
)

START = '2016-01-01'
END = '2026-01-01'
SOURCE = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def execution_panel(cleaned: pd.DataFrame, raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Preserve independently available execution quotes after feature cleaning.

    The shared cleaner masks a full OHLC row if any quote is invalid. A valid
    opening quote with an absent closing quote still represents an unresolved
    executed trade; it must not silently become a cash allocation.
    """
    execution = raw_panel[['Date', 'Ticker', 'Open', 'Close']].copy()
    execution['Date'] = pd.to_datetime(execution['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    execution['Ticker'] = execution['Ticker'].astype('string').str.strip().str.upper().str.replace('.', '-', regex=False)
    for name in ('Open', 'Close'):
        values = pd.to_numeric(execution[name], errors='coerce')
        execution[f'Execution{name}'] = values.where(np.isfinite(values) & values.gt(0))
    execution = execution[['Date', 'Ticker', 'ExecutionOpen', 'ExecutionClose']].drop_duplicates()
    if execution.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Conflicting execution quotes for a date/ticker.')
    return cleaned.merge(execution, on=['Date', 'Ticker'], how='left', validate='one_to_one')


def load_showcase_data(root: Path, *, refresh: bool = False):
    """Return full quote panel, shared nine-feature dataset, and provenance."""
    raw_dir = root / 'data/raw'
    processed = root / 'data/processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(raw_dir / '.yfinance-cache'))
    constituents = raw_dir / f'sp500_{START}_{END}.csv'
    prices = raw_dir / f'prices_{START}_{END}.csv'
    manifest_path = processed / 'showcase_data_manifest.json'
    cache = not refresh and constituents.exists() and prices.exists()
    if cache:
        print('Loading the saved notebook data snapshot.', flush=True)
        stocks = pd.read_csv(constituents)
        raw = pd.read_csv(prices, header=[0, 1], index_col=0, parse_dates=[0], float_precision='round_trip')
    else:
        print('Downloading the notebook constituent list and 2016–2025 prices.', flush=True)
        response = requests.get(SOURCE, headers={'User-Agent': 'Mozilla/5.0'}, timeout=45)
        response.raise_for_status()
        columns = ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']
        tables = pd.read_html(StringIO(response.text))
        table = next((table for table in tables if set(columns).issubset(table.columns)), None)
        if table is None:
            raise ValueError('Wikipedia did not return the notebook constituent table.')
        stocks = table[columns].copy()
        stocks['Ticker'] = stocks['Symbol'].str.strip().str.replace('.', '-', regex=False)
        stocks = stocks.sort_values('Ticker').reset_index(drop=True)
        tickers = stocks['Ticker'].tolist()
        print(f'Downloading {len(tickers)} securities; unavailable histories remain recorded.', flush=True)
        raw = yf.download(tickers=tickers, start=START, end=END, interval='1d', auto_adjust=False,
                          group_by='ticker', multi_level_index=True, keepna=True, ignore_tz=True,
                          threads=8, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError('No Yahoo Finance prices were downloaded; no substitute dataset was generated.')
        raw.columns.names = ['Ticker', 'Field']
        expected = pd.MultiIndex.from_product([tickers, QUOTE_COLUMNS], names=['Ticker', 'Field'])
        raw = raw.reindex(columns=expected).sort_index()
        missing = [ticker for ticker in tickers if raw[(ticker, 'Adj Close')].notna().sum() == 0]
        if missing:
            print(f'Retrying {len(missing)} unavailable downloads once.', flush=True)
            retry = yf.download(tickers=missing, start=START, end=END, interval='1d', auto_adjust=False,
                                group_by='ticker', multi_level_index=True, keepna=True, ignore_tz=True,
                                threads=2, progress=False)
            if retry is not None and not retry.empty:
                retry.columns.names = ['Ticker', 'Field']
                raw = raw.combine_first(retry).reindex(columns=expected).sort_index()
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = 'Date'
        raw.to_csv(prices)
        stocks.to_csv(constituents, index=False)
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise ValueError('Saved price columns do not match the notebook format.')
    raw.columns.names = ['Ticker', 'Field']
    if raw.index.has_duplicates or raw.empty:
        raise ValueError('Saved dates are empty or duplicated.')
    if (raw.index < pd.Timestamp(START)).any() or (raw.index >= pd.Timestamp(END)).any():
        raise ValueError('Saved price dates are outside the notebook date range.')
    raw_panel = raw.stack(level='Ticker', future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    raw_panel.columns.name = None
    metadata = stocks[['Ticker', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    panel, audit = prepare_price_panel(raw_panel, metadata)
    panel = execution_panel(panel, raw_panel)
    model_data = add_date_splits(make_model_dataset(panel, labeled_only=False))
    model_data = model_data[['Date', 'Ticker', 'Split', 'LabelEndDate', *FEATURES, 'target', 'target_rank']]
    if not {'train', 'validation', 'test'}.issubset(set(model_data['Split'])):
        raise ValueError('Notebook data did not cover every required chronological split.')
    counts = panel.groupby('Ticker')['QuoteValid'].sum()
    raw_digest = sha256_file(prices)
    universe_digest = sha256_file(constituents)
    combined = hashlib.sha256(f'{raw_digest}:{universe_digest}'.encode()).hexdigest()
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() and cache else {}
    manifest = {
        'snapshot_created_at': previous.get('snapshot_created_at', datetime.now().astimezone().isoformat()),
        'source': SOURCE, 'price_source': 'Yahoo Finance through yfinance',
        'requested_start': START, 'requested_end_exclusive': END,
        'start': str(panel['Date'].min().date()), 'end': str(panel['Date'].max().date()),
        'rows': int(len(model_data)), 'quote_rows': int(len(panel)),
        'tickers': int(model_data['Ticker'].nunique()), 'requested_tickers': int(len(stocks)),
        'unavailable_tickers': counts.index[counts.eq(0)].tolist(),
        'features': FEATURES, 'sha256': combined, 'prices_sha256': raw_digest,
        'constituents_sha256': universe_digest,
        'raw_snapshot_paths': [str(prices.relative_to(root)), str(constituents.relative_to(root))],
        'cleaning': audit, 'universe_policy': 'Wikipedia constituents at download time; not historical membership',
    }
    model_data.to_parquet(processed / 'model_data.parquet', index=False)
    model_data.to_csv(processed / 'model_data.csv', index=False)
    panel.to_parquet(processed / 'showcase_panel.parquet', index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Prepared {len(model_data):,} usable signal rows for {manifest["tickers"]} securities.', flush=True)
    return panel, model_data, manifest


if __name__ == '__main__':
    load_showcase_data(Path(__file__).resolve().parents[2])
