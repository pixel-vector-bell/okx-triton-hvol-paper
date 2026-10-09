"""
Research Engine — Poseidon Trading Challenge
Generates, backtests, and ranks 200+ strategy configurations.
Reads OKX historical data, tests all strategy families, records all results.
"""
import sys, os, json, time, gzip, itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import load_data, candles_to_dicts, PAIRS, fetch_full_history, save_data
from indicators import as_df, ema
from strategies import _prep, run_strategy
from backtester import walk_forward_splits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CAPITAL = 3000.0

# ─── Parameter Grids (generates 200+ configs) ──────────────────────────────

def get_param_grid():
    """Returns list of (strategy_name, params_dict) for all 200+ configs."""
    configs = []

    # ── A: TRITON-VSMO (Volume Surge + Momentum Oscillator) ──
    for rsi_lo in [35, 40, 45]:
        for vol_thresh in [1.5, 2.0, 2.5, 3.0]:
            for stop_mult in [1.0, 1.5, 2.0]:
                for hours in [
                    [],  # all hours
                    [14, 15, 16, 17, 18, 19],  # US/EU overlap
                    [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21],  # main session
                ]:
                    configs.append(("VSMO", {
                        "rsi_lo": rsi_lo, "rsi_hi": rsi_lo + 35,
                        "vol_thresh": vol_thresh,
                        "stop_atr_mult": stop_mult,
                        "allowed_hours": hours,
                        "tp_atr_mult": 3.0,
                        "risk_pct": 0.01,
                        "max_hold_bars": 144,  # 12h on 5m
                        "use_trail": True, "trail_ema": 21,
                    }))

    # ── B: TRITON-STAB (Squeeze and Break) ──
    for min_sq in [3, 5, 8]:
        for vol_thresh in [1.5, 2.0, 2.5]:
            for stop_mult in [1.5, 2.0, 2.5]:
                for adx_exit in [15, 20]:
                    configs.append(("STAB", {
                        "min_squeeze_bars": min_sq,
                        "vol_thresh": vol_thresh,
                        "stop_atr_mult": stop_mult,
                        "adx_exit": adx_exit,
                        "tp_atr_mult": 3.0,
                        "allowed_hours": [8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21],
                        "risk_pct": 0.01,
                        "max_hold_bars": 240,  # 20h on 5m
                        "use_trail": True, "trail_ema": 21,
                    }))

    # ── C: TRITON-MTALIGN (Multi-TF Alignment) ──
    for pb_thresh in [0.003, 0.005, 0.008]:
        for vol_thresh in [1.1, 1.3, 1.5, 2.0]:
            for hours in [
                [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21],
                [14, 15, 16, 17, 18, 19],
                [],
            ]:
                configs.append(("MTALIGN", {
                    "pb_thresh": pb_thresh,
                    "vol_thresh": vol_thresh,
                    "allowed_hours": hours,
                    "tp_atr_mult": 2.5,
                    "risk_pct": 0.01,
                    "max_hold_bars": 96,  # 8h on 5m
                    "use_trail": True, "trail_ema": 21,
                    "stop_extra": 0.002,
                }))

    # ── D: TRITON-RVOL (Relative Volume Leader) ──
    for min_roc in [1.0, 1.5, 2.0, 2.5]:
        for vol_thresh in [2.0, 2.5, 3.0, 3.5]:
            for max_rsi in [65, 70, 75]:
                for stop_mult in [1.5, 2.0]:
                    configs.append(("RVOL", {
                        "vol_thresh": vol_thresh,
                        "min_roc": min_roc,
                        "max_rsi": max_rsi,
                        "min_adx": 18,
                        "stop_atr_mult": stop_mult,
                        "rsi_exit": max_rsi + 5,
                        "allowed_hours": [14, 15, 16, 17, 18, 19, 20, 21],
                        "risk_pct": 0.01,
                        "max_hold_bars": 72,
                        "use_trail": True, "trail_pct": 0.02,
                        "tp_atr_mult": 3.0,
                    }))

    # ── E: TRITON-HVOL (ADX Breakout) ──
    for min_adx in [20, 25, 30]:
        for vol_thresh in [1.3, 1.5, 2.0]:
            for max_ext in [0.01, 0.02, 0.03]:
                for adx_exit in [15, 20, 25]:
                    configs.append(("HVOL", {
                        "min_adx": min_adx,
                        "vol_thresh": vol_thresh,
                        "max_ext": max_ext,
                        "adx_exit": adx_exit,
                        "allowed_hours": [],  # all hours
                        "risk_pct": 0.01,
                        "max_hold_bars": 120,
                        "use_trail": False,
                        "tp_atr_mult": 2.5,
                    }))

    # ── F: TRITON-EMASTACK (EMA Stack Pullback) ──
    for max_pb in [0.005, 0.010, 0.015]:
        for vol_thresh in [1.2, 1.5, 2.0]:
            for rsi_lo in [38, 45, 50]:
                for min_adx in [15, 20, 25]:
                    for hours in [
                        [8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21],
                        [],
                    ]:
                        configs.append(("EMASTACK", {
                            "max_pb": max_pb,
                            "vol_thresh": vol_thresh,
                            "rsi_lo": rsi_lo,
                            "rsi_hi": rsi_lo + 28,
                            "rsi_exit": rsi_lo + 35,
                            "min_adx": min_adx,
                            "allowed_hours": hours,
                            "risk_pct": 0.01,
                            "max_hold_bars": 144,
                            "use_trail": True, "trail_ema": 21,
                            "tp_atr_mult": 2.5,
                            "stop_extra": 0.002,
                        }))

    return configs


# ─── Data Loading ──────────────────────────────────────────────────────────

def load_pair_df(pair: str, bar: str = "5m") -> pd.DataFrame:
    """Load and prepare a pair's candle data."""
    raw = load_data(pair, bar)
    if not raw:
        return None
    dicts = candles_to_dicts(raw)
    df = as_df(dicts)
    return df


def load_btc_regime(bar: str = "1H") -> pd.DataFrame:
    """Load BTC 1H data for regime filtering."""
    raw = load_data("BTC-USDT", bar)
    if not raw:
        return None
    dicts = candles_to_dicts(raw)
    df = as_df(dicts)
    df = _prep(df, {})
    return df


# ─── BTC Regime Alignment ─────────────────────────────────────────────────

def add_btc_regime(df: pd.DataFrame, btc_1h: pd.DataFrame) -> pd.DataFrame:
    """Add btc_regime column (+1/-1) to df by aligning timestamps."""
    if btc_1h is None:
        df["btc_regime"] = 1
        return df
    btc_regime = (btc_1h["ema21"] >= btc_1h["ema50"] * 0.997).astype(int).replace({0: -1})
    # Resample to 5m (forward-fill)
    btc_5m = btc_regime.reindex(df.index, method="ffill")
    df["btc_regime"] = btc_5m.fillna(1)
    return df


# ─── Core Research Loop ───────────────────────────────────────────────────

def run_research(max_configs: int = None, pairs_to_test: list = None,
                 bar: str = "5m") -> pd.DataFrame:
    """
    Full research loop. Generates all configs, backtests on all pairs,
    saves results. Returns ranked DataFrame.
    """
    configs = get_param_grid()
    if max_configs:
        configs = configs[:max_configs]

    if pairs_to_test is None:
        # Use liquid altcoins (skip BTC/ETH for backtest diversity, include them for RVOL)
        pairs_to_test = [
            "SOL-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "NEAR-USDT",
            "SUI-USDT", "UNI-USDT", "PEPE-USDT", "HYPE-USDT", "ONDO-USDT",
            "ADA-USDT", "DOGE-USDT", "FIL-USDT", "WLD-USDT", "XLM-USDT",
            "BTC-USDT", "ETH-USDT",
        ]

    print(f"Research engine: {len(configs)} configs × up to {len(pairs_to_test)} pairs")
    print(f"Loading BTC regime data...")
    btc_1h = load_btc_regime("1H")
    print(f"BTC 1H loaded: {len(btc_1h) if btc_1h is not None else 'FAILED'} bars")

    # Load and prep all pair DFs upfront
    pair_dfs = {}
    print("Loading pair data...")
    for pair in pairs_to_test:
        df = load_pair_df(pair, bar)
        if df is None or len(df) < 200:
            print(f"  [skip] {pair}: insufficient data")
            continue
        df = _prep(df, {})
        df = add_btc_regime(df, btc_1h)
        pair_dfs[pair] = df
        print(f"  ✓ {pair}: {len(df)} bars")

    if not pair_dfs:
        print("ERROR: No pair data available!")
        return pd.DataFrame()

    # Load HTF data for MTALIGN
    htf_dfs = {}
    if any(s == "MTALIGN" for s, _ in configs):
        print("Loading 1H data for MTALIGN...")
        for pair in pair_dfs:
            df_1h = load_pair_df(pair, "1H")
            if df_1h is not None and len(df_1h) > 50:
                df_1h = _prep(df_1h, {})
                htf_dfs[pair] = df_1h

    # Run all configs
    all_results = []
    total = len(configs) * len(pair_dfs)
    done = 0
    start_time = time.time()

    checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.json")
    completed_set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpointed = json.load(f)
            completed_set = set(checkpointed.get("completed", []))
            all_results = checkpointed.get("results", [])
            print(f"  Resuming from checkpoint: {len(completed_set)} already done")

    for strategy, params in configs:
        for pair in pair_dfs:
            config_key = f"{strategy}_{pair}_{json.dumps(params, sort_keys=True)}"
            if config_key in completed_set:
                done += 1
                continue

            try:
                df = pair_dfs[pair]
                htf_df = htf_dfs.get(pair) if strategy == "MTALIGN" else None
                result = run_strategy(df, strategy, params, pair=pair, capital=CAPITAL, htf_df=htf_df)
                summary = result.summary()
                all_results.append(summary)
                completed_set.add(config_key)
            except Exception as e:
                all_results.append({
                    "pair": pair, "strategy": strategy, "params": params,
                    "error": str(e), "total_pnl": -9999, "n_trades": 0
                })
                completed_set.add(config_key)

            done += 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {strategy}/{pair} | ETA: {remaining/60:.1f}m")
                # Save checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump({"completed": list(completed_set), "results": all_results}, f)

    # Save full results
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Build ranked DataFrame
    valid = [r for r in all_results if "error" not in r and r.get("n_trades", 0) >= 5]
    if not valid:
        print("No valid results!")
        return pd.DataFrame()

    df_results = pd.DataFrame(valid)

    # Composite score: weighted combination of net PnL, profit factor, sharpe, drawdown control
    df_results["score"] = (
        df_results["total_pnl"].clip(-1000, 5000) / 100 * 0.4 +
        df_results["profit_factor"].clip(0, 10) * 5 * 0.3 +
        df_results["sharpe"].clip(-5, 20) * 2 * 0.2 +
        (1 - df_results["max_drawdown"].clip(0, 1)) * 10 * 0.1
    )

    df_results = df_results.sort_values("score", ascending=False)

    # Save ranked results
    ranked_path = os.path.join(RESULTS_DIR, "ranked_results.csv")
    df_results.to_csv(ranked_path, index=False)
    print(f"\n✓ Saved {len(df_results)} results to {ranked_path}")

    return df_results


# ─── Walk-Forward Validation ───────────────────────────────────────────────

def walk_forward_validate(strategy: str, params: dict, pairs: list,
                           n_splits: int = 5, bar: str = "5m") -> dict:
    """
    Walk-forward validation of a specific strategy/params combo.
    Returns summary with OOS (out-of-sample) performance.
    """
    from indicators import as_df
    oos_results = []

    for pair in pairs:
        df = load_pair_df(pair, bar)
        if df is None or len(df) < 300:
            continue
        df = _prep(df, {})

        splits = walk_forward_splits(df, n_splits=n_splits, train_frac=0.7)
        for train_df, test_df in splits:
            # We'd normally optimise on train — here we use fixed params on OOS test
            result = run_strategy(test_df, strategy, params, pair=pair, capital=CAPITAL)
            if result.n_trades >= 3:
                oos_results.append(result.summary())

    if not oos_results:
        return {"oos_results": 0, "avg_pnl": 0, "avg_pf": 0}

    avg_pnl = np.mean([r["total_pnl"] for r in oos_results])
    avg_pf = np.mean([r["profit_factor"] for r in oos_results])
    win_count = sum(1 for r in oos_results if r["total_pnl"] > 0)

    return {
        "oos_results": len(oos_results),
        "avg_oos_pnl": round(avg_pnl, 2),
        "avg_oos_pf": round(avg_pf, 4),
        "oos_win_rate": round(win_count / len(oos_results), 3),
        "details": oos_results,
    }


# ─── Critic Agent ──────────────────────────────────────────────────────────

def critic_review(result_summary: dict, oos_validation: dict) -> dict:
    """
    Adversarial critic. Checks for:
    1. Overfitting (IS vs OOS divergence)
    2. Low trade count (not statistically significant)
    3. High drawdown
    4. Profit from a few lucky trades
    5. Unrealistic assumptions
    """
    flags = []
    score = 100  # start with 100, deduct for issues

    n = result_summary.get("n_trades", 0)
    pf = result_summary.get("profit_factor", 0)
    pnl = result_summary.get("total_pnl", 0)
    dd = result_summary.get("max_drawdown", 0)
    wr = result_summary.get("win_rate", 0)

    # Trade count
    if n < 10:
        flags.append(f"⚠️ Only {n} trades — statistically weak (need 20+)")
        score -= 30
    elif n < 20:
        flags.append(f"⚠️ Only {n} trades — borderline significance")
        score -= 10

    # Profit factor
    if pf > 5.0:
        flags.append(f"⚠️ PF={pf:.2f} suspiciously high — check for overfitting")
        score -= 20
    elif pf > 3.0 and n < 30:
        flags.append(f"⚠️ High PF with few trades — may be luck")
        score -= 10

    # Drawdown
    if dd > 0.25:
        flags.append(f"🔴 Drawdown {dd:.1%} — exceeds 25% threshold")
        score -= 25
    elif dd > 0.15:
        flags.append(f"⚠️ Drawdown {dd:.1%} — elevated risk")
        score -= 10

    # Win rate vs expectancy
    if wr > 0.8 and pf > 4:
        flags.append("⚠️ Very high win rate — may be stop-loss avoidance")
        score -= 15
    if wr < 0.3 and pf < 1.5:
        flags.append("🔴 Low win rate AND low profit factor — not viable")
        score -= 20

    # OOS validation
    oos_pnl = oos_validation.get("avg_oos_pnl", 0)
    oos_pf = oos_validation.get("avg_oos_pf", 0)
    oos_wr = oos_validation.get("oos_win_rate", 0)
    n_oos = oos_validation.get("oos_results", 0)

    if n_oos < 3:
        flags.append("⚠️ Insufficient OOS folds for validation")
        score -= 15
    elif oos_pnl < 0 and pnl > 0:
        flags.append(f"🔴 IS profitable but OOS negative — clear overfitting signal")
        score -= 40
    elif oos_pf < 1.0 and pf > 1.5:
        flags.append(f"🔴 PF drops from {pf:.2f} (IS) to {oos_pf:.2f} (OOS) — overfitting")
        score -= 30
    elif oos_pf < 1.3 and pf > 1.5:
        flags.append(f"⚠️ PF shrinks {pf:.2f} → {oos_pf:.2f} IS→OOS")
        score -= 10

    # Summary
    verdict = "REJECT" if score < 40 else ("CAUTION" if score < 65 else "ACCEPT")

    return {
        "score": max(0, score),
        "verdict": verdict,
        "flags": flags,
        "is_pnl": pnl,
        "is_pf": pf,
        "oos_pnl": oos_pnl,
        "oos_pf": oos_pf,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Poseidon Research Engine")
    parser.add_argument("--fetch", action="store_true", help="Fetch OKX data first")
    parser.add_argument("--configs", type=int, default=None, help="Limit configs for testing")
    parser.add_argument("--bar", default="5m", help="Candle timeframe")
    parser.add_argument("--pairs", nargs="+", default=None, help="Specific pairs to test")
    args = parser.parse_args()

    if args.fetch:
        print("Fetching data first...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "fetch_data.py")])

    configs = get_param_grid()
    print(f"Total configs generated: {len(configs)}")

    df_results = run_research(
        max_configs=args.configs,
        pairs_to_test=args.pairs,
        bar=args.bar,
    )

    if not df_results.empty:
        print("\n=== TOP 20 CONFIGURATIONS ===")
        cols = ["strategy", "pair", "n_trades", "win_rate", "total_pnl",
                "profit_factor", "expectancy", "max_drawdown", "sharpe", "score"]
        available_cols = [c for c in cols if c in df_results.columns]
        print(df_results[available_cols].head(20).to_string(index=False))
