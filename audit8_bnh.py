import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from research_engine import load_pair_df, load_btc_regime, add_btc_regime, CAPITAL
from strategies import _prep, run_strategy

FROZEN_PARAMS = {
    'min_adx': 30, 'vol_thresh': 1.5, 'max_ext': 0.02, 'adx_exit': 20,
    'allowed_hours': [], 'risk_pct': 0.01, 'max_hold_bars': 120,
    'use_trail': False, 'tp_atr_mult': 2.5,
}

# Buy-and-hold benchmark: same $3000, same pairs, same period
# For a fair comparison vs TRITON, we use equivalent capital exposure
# TRITON risks 1% per trade, max 20% position at once
# B&H: equal-weight across the same pairs
ALL_PAIRS = [
    'ENA-USDT', 'SOL-USDT', 'AVAX-USDT', 'LINK-USDT', 'NEAR-USDT',
    'SUI-USDT', 'UNI-USDT', 'HYPE-USDT', 'ONDO-USDT', 'FIL-USDT',
    'WLD-USDT', 'DOGE-USDT', 'PEPE-USDT', 'ADA-USDT', 'XLM-USDT',
    'BTC-USDT', 'ETH-USDT', 'APT-USDT', 'LIT-USDT',
    'DOT-USDT', 'LTC-USDT', 'BNB-USDT', 'XRP-USDT', 'ZEC-USDT',
    'SAND-USDT', 'STRK-USDT', 'PUMP-USDT', 'OKB-USDT', 'TRUMP-USDT',
]

print('=== AUDIT 8: BUY-AND-HOLD COMPARISON ===')
print()

btc_1h = load_btc_regime('1H')
bnh_results = []
triton_results = []

for pair in ALL_PAIRS:
    df = load_pair_df(pair, '1H')
    if df is None or len(df) < 200:
        continue
    df = _prep(df, {})
    df = add_btc_regime(df, btc_1h)

    # Buy-and-hold: buy at open of first bar, sell at close of last bar
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    bnh_return = (end_price / start_price - 1)
    # Equivalent exposure: same $3000 capital, equal weight
    # We compute the gross return and assume same $3000 invested
    bnh_pnl = CAPITAL * bnh_return
    # Subtract round-trip fees (one buy + one sell)
    fee_rt = CAPITAL * 0.0008 * 2 + CAPITAL * 0.0005 * 2
    bnh_pnl_net = bnh_pnl - fee_rt

    # TRITON
    r = run_strategy(df, 'HVOL', FROZEN_PARAMS, pair=pair, capital=CAPITAL)

    bnh_results.append({'pair': pair, 'bnh_pnl': round(bnh_pnl_net, 2),
                        'bnh_return': round(bnh_return * 100, 1)})
    triton_results.append({'pair': pair, 'triton_pnl': round(r.total_pnl, 2),
                           'n': r.n_trades})

    print(f'{pair:15s}: BnH ${bnh_pnl_net:+7.2f} ({bnh_return:+.0%})  vs  TRITON ${r.total_pnl:+7.2f} (n={r.n_trades:3d})')

print()
bnh_total = sum(r['bnh_pnl'] for r in bnh_results)
triton_total = sum(r['triton_pnl'] for r in triton_results)
bnh_wins = sum(1 for r in bnh_results if r['bnh_pnl'] > 0)
triton_wins = sum(1 for r in triton_results if r['triton_pnl'] > 0)
n_pairs = len(bnh_results)

avg_bnh = bnh_total / n_pairs
avg_triton = triton_total / n_pairs

print('=== AGGREGATE COMPARISON ===')
print(f'  Pairs: {n_pairs}')
print(f'  BUY AND HOLD: Total PnL ${bnh_total:+.2f} | Avg per pair ${avg_bnh:+.2f} | Profitable pairs: {bnh_wins}/{n_pairs}')
print(f'  TRITON-HVOL:  Total PnL ${triton_total:+.2f} | Avg per pair ${avg_triton:+.2f} | Profitable pairs: {triton_wins}/{n_pairs}')
print()
print(f'  TRITON edge vs BnH: ${triton_total - bnh_total:+.2f}')

# Equal-weight portfolio B&H
print()
print('=== EQUAL-WEIGHT PORTFOLIO B&H (single bet of $3000 on 29 coins) ===')
avg_return = np.mean([r['bnh_return'] for r in bnh_results])
print(f'  Equal-weight avg return: {avg_return:+.1f}%')
print(f'  $3000 portfolio B&H return: ${CAPITAL * avg_return / 100:+.2f}')

# Also: per-trade comparison
print()
print('=== RISK-ADJUSTED CONTEXT ===')
print('  TRITON max exposure per trade: ~20% of capital = ~$600')
print('  BnH full exposure: $3000 (5x more capital at risk per coin)')
print('  To be fair, scale BnH to same 20% exposure per coin:')
scaled_bnh = sum(r['bnh_pnl'] * 0.20 for r in bnh_results)
print(f'  Scaled B&H (20% position): ${scaled_bnh:+.2f}')
print(f'  TRITON at same exposure: ${triton_total:+.2f}')
