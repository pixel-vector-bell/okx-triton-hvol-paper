import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import pandas as pd
from research_engine import load_pair_df, load_btc_regime, add_btc_regime, CAPITAL
from strategies import _prep, run_strategy
from backtester import walk_forward_splits

print('=== AUDIT 7: CONTAMINATION TIMELINE ===')
print()
print('Reconstructing exact sequence of events:')
print()
print('STEP 1: Research run on 5m data (9,129 tests)')
print('  - All 17 pairs including ENA in discovery. 5m=5 days, too few trades.')
print()
print('STEP 2: Re-run on 1H data')
print('  - All 20 pairs tested including ENA')
print('  - ENA-USDT HVOL emerged as #1 in ranked_results_1h.csv')
print()
print('STEP 3: Validation used ENA (same as discovery)')
print('  - walk_forward_validate() included ENA')
print('  - run_comparison() included ENA')
print()
print('=== CONTAMINATION VERDICT ===')
print()
print('CONTAMINATED: ENA was NOT a holdout coin. It was in the discovery set.')
print('All validation metrics including OOS are partially contaminated by ENA.')
print()

df = pd.read_csv('results/ranked_results_1h.csv')
top_hvol_ena = df[(df['strategy']=='HVOL') & (df['pair']=='ENA-USDT') & (df['n_trades']>=10)].sort_values('score',ascending=False)
top_non_ena = df[(df['strategy']=='HVOL') & (df['pair']!='ENA-USDT') & (df['n_trades']>=10)].sort_values('score',ascending=False)
print(f'ENA in top HVOL results: {len(top_hvol_ena)} configs. Best: ${top_hvol_ena.iloc[0]["total_pnl"]:.2f}')
print(f'Best non-ENA HVOL: {top_non_ena.iloc[0]["pair"]} ${top_non_ena.iloc[0]["total_pnl"]:.2f} (SAND-USDT +$316)')

FROZEN_PARAMS = {
    'min_adx': 30, 'vol_thresh': 1.5, 'max_ext': 0.02, 'adx_exit': 20,
    'allowed_hours': [], 'risk_pct': 0.01, 'max_hold_bars': 120,
    'use_trail': False, 'tp_atr_mult': 2.5,
}
btc_1h = load_btc_regime('1H')
WF_EX_ENA = ['SOL-USDT', 'AVAX-USDT', 'LINK-USDT', 'NEAR-USDT',
              'HYPE-USDT', 'ONDO-USDT', 'FIL-USDT', 'WLD-USDT', 'APT-USDT']
oos_pnls = []
oos_profitable = 0
for pair in WF_EX_ENA:
    df_p = load_pair_df(pair, '1H')
    if df_p is None or len(df_p) < 300:
        continue
    df_p = _prep(df_p, {})
    df_p = add_btc_regime(df_p, btc_1h)
    splits = walk_forward_splits(df_p, n_splits=4, train_frac=0.7)
    for train_df, test_df in splits:
        r = run_strategy(test_df, 'HVOL', FROZEN_PARAMS, pair=pair, capital=CAPITAL)
        if r.n_trades >= 1:
            oos_pnls.append(r.total_pnl)
            if r.total_pnl > 0:
                oos_profitable += 1

print()
print('OOS Walk-Forward excluding ENA:')
print(f'  Folds: {len(oos_pnls)}')
print(f'  Profitable: {oos_profitable} / {len(oos_pnls)} = {oos_profitable/len(oos_pnls):.0%}' if oos_pnls else '  No folds')
print(f'  Avg OOS PnL: ${sum(oos_pnls)/len(oos_pnls):.2f}' if oos_pnls else '')
print(f'  Aggregate: ${sum(oos_pnls):.2f}' if oos_pnls else '')
