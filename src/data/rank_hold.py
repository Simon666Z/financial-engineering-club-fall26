"""Execution/marking inputs for the persistent notebook-data simulation.

OHLC4 is a hypothetical daily-average fill, not measured VWAP. Price units are
normalized like yfinance auto_adjust: (Adj Close / Close) * OHLC. This creates
synthetic total-return units, not verified historical shares/dividend cashflows.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def add_rank_hold_prices(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Attach independent execution and marking quotes without using targets.

    raw must use the saved notebook's two-level Ticker/Field price columns.
    All prices remain on the full supplied calendar. Missing fills/marks remain
    missing; later execution code determines outcomes after selecting orders.
    """
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise ValueError('Expected saved notebook Ticker/Field price columns.')
    fields = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    raw = raw.copy()
    raw.columns.names = ['Ticker', 'Field']
    raw.index = pd.to_datetime(raw.index, utc=True).tz_localize(None).normalize()
    raw.index.name = 'Date'
    if raw.index.has_duplicates:
        raise ValueError('Raw notebook prices have duplicate calendar dates.')
    if not set(fields).issubset(set(raw.columns.get_level_values('Field'))):
        raise ValueError('Raw prices lack required OHLC or adjusted-close fields.')
    values = raw.loc[:, raw.columns.get_level_values('Field').isin(fields)].stack(level='Ticker', future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    values.columns.name = None
    values['Ticker'] = values['Ticker'].astype('string').str.strip().str.upper().str.replace('.', '-', regex=False)
    if values.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Raw prices contain conflicting ticker/date keys.')
    prices = values[fields].apply(pd.to_numeric, errors='coerce')
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    valid_bar = prices[['Open', 'High', 'Low', 'Close']].notna().all(axis=1)
    valid_bar &= prices['High'].ge(prices[['Open', 'Low', 'Close']].max(axis=1))
    valid_bar &= prices['Low'].le(prices[['Open', 'High', 'Close']].min(axis=1))
    factor = prices['Adj Close'] / prices['Close']
    factor = factor.where(np.isfinite(factor) & factor.gt(0))
    values['RawExecutionAverage'] = prices[['Open', 'High', 'Low', 'Close']].mean(axis=1, skipna=False).where(valid_bar)
    values['AdjustmentFactor'] = factor
    values['ExecutionAverage'] = values['RawExecutionAverage'] * factor
    values['MarkClose'] = prices['Adj Close']
    keep = ['Date', 'Ticker', 'RawExecutionAverage', 'AdjustmentFactor', 'ExecutionAverage', 'MarkClose']
    result = panel.drop(columns=[name for name in keep[2:] if name in panel]).merge(values[keep], on=['Date', 'Ticker'], how='left', validate='one_to_one')
    return result


def load_rank_hold_prices(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / 'data/raw/prices_2016-01-01_2026-01-01.csv'
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=[0], float_precision='round_trip')
    return add_rank_hold_prices(panel, raw)
