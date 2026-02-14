"""
Orchestration: build_summary_sheet, format_excel, run_pipeline.
Depends on: rules, data, engine.
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from .rules import COLS
from .data import load_parquet_files, merge_all, run_sanity_checks
from .engine import (
    build_master_sheet, build_valuation_sheet, build_quality_sheet,
    build_cashflow_sheet, build_leverage_sheet, build_growth_sheet,
    build_shareholding_sheet, build_neglected_firm_sheet, build_dividends_sheet,
    build_red_flags_sheet, build_analysis_sheet, add_peer_ranking,
    build_decision_audit_sheet,
)


def build_summary_sheet(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary sheet."""
    total = len(analysis_df)
    dc = analysis_df['Decision_Bucket'].value_counts()
    rows = [['STOCK SCREENING SUMMARY', '', ''], [f'Total: {total}', f'Generated: {datetime.now():%Y-%m-%d %H:%M}', ''],
            ['Decision', 'Count', 'Pct']]
    for b in ['GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE', 'SCREEN_PASSED_FLAGS', 'SCREEN_MARGINAL', 'SCREEN_FAILED', 'CONTRARIAN_BET', 'VALUE_TRAP']:
        c = dc.get(b, 0)
        rows.append([b, c, f'{c/total*100:.1f}%'])
    return pd.DataFrame(rows, columns=['A', 'B', 'C'])


def format_excel(writer: pd.ExcelWriter, sheets: Dict[str, pd.DataFrame]) -> None:
    """Apply formatting."""
    wb = writer.book
    hdr = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
    bucket_fmts = {
        'GATES_CLEARED': wb.add_format({'bg_color': '#92D050'}), 'SCREEN_PASSED_EXPENSIVE': wb.add_format({'bg_color': '#FFC000'}),
        'SCREEN_PASSED_FLAGS': wb.add_format({'bg_color': '#FFEB9C'}), 'SCREEN_FAILED': wb.add_format({'bg_color': '#FF6B6B'}),
        'CONTRARIAN_BET': wb.add_format({'bg_color': '#7030A0', 'font_color': 'white'}), 'VALUE_TRAP': wb.add_format({'bg_color': '#C00000', 'font_color': 'white'}),
    }
    for name, df in sheets.items():
        ws = writer.sheets[name]
        for i, c in enumerate(df.columns): ws.write(0, i, c, hdr)
        max_width = 80 if name == 'Decision_Audit' else 30
        for i, c in enumerate(df.columns):
            mx = max(len(str(c)), df[c].fillna('').astype(str).str.len().max() if len(df) > 0 else 0)
            ws.set_column(i, i, min(mx + 2, max_width))
        ws.freeze_panes(1, 1)
        if name == 'Analysis' and 'Decision_Bucket' in df.columns:
            bc = df.columns.get_loc('Decision_Bucket')
            for ri in range(len(df)):
                v = df.iloc[ri, bc]
                if v in bucket_fmts: ws.write(ri + 1, bc, v, bucket_fmts[v])


def run_pipeline(base_dir: str, output_path: str) -> None:
    """Full pipeline: load → merge → build sheets → write Excel."""
    logging.info("Starting stock analysis...")

    dfs = load_parquet_files(base_dir)
    if len(dfs) < 6:
        raise RuntimeError(f"Missing parquet files. Found {len(dfs)}, expected 6.")

    merged = merge_all(dfs)
    run_sanity_checks(merged)

    master_df = build_master_sheet(merged)
    valuation_df = build_valuation_sheet(merged)
    quality_df = build_quality_sheet(merged)
    cashflow_df = build_cashflow_sheet(merged)
    leverage_df = build_leverage_sheet(merged)
    growth_df = build_growth_sheet(merged)
    shareholding_df = build_shareholding_sheet(merged)
    neglected_df = build_neglected_firm_sheet(merged, quality_df, leverage_df, shareholding_df)
    dividends_df = build_dividends_sheet(merged)
    red_flags_df = build_red_flags_sheet(merged, quality_df, cashflow_df, leverage_df, valuation_df, growth_df)
    analysis_df = build_analysis_sheet(merged, quality_df, valuation_df, leverage_df, growth_df, cashflow_df, red_flags_df, shareholding_df)
    analysis_df = add_peer_ranking(analysis_df, quality_df, valuation_df, leverage_df, cashflow_df)
    decision_audit_df = build_decision_audit_sheet(merged, analysis_df, quality_df, valuation_df, leverage_df, growth_df, cashflow_df, red_flags_df, shareholding_df)
    summary_df = build_summary_sheet(analysis_df)

    sort_idx = analysis_df['Composite_Score'].astype(float).argsort()[::-1]
    analysis_df = analysis_df.iloc[sort_idx].reset_index(drop=True)
    sheet_dfs = {n: df.iloc[sort_idx].reset_index(drop=True) for n, df in
                 {'Master': master_df, 'Valuation': valuation_df, 'Quality': quality_df, 'Cash_Flow': cashflow_df,
                  'Leverage': leverage_df, 'Growth': growth_df, 'Shareholding': shareholding_df,
                  'Neglected_Firm': neglected_df, 'Dividends': dividends_df, 'Red_Flags': red_flags_df,
                  'Decision_Audit': decision_audit_df}.items()}

    all_sheets = {'Analysis': analysis_df, **sheet_dfs, 'Summary': summary_df}
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        format_excel(writer, all_sheets)

    logging.info(f"Done! Output: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
