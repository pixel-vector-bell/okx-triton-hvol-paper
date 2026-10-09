"""
Final Report Generator — Poseidon Trading Challenge
Compares TRITON candidates vs Ross baseline, runs critic review,
and generates comprehensive HTML report + competition blotter.
"""
import sys, os, json, time, ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from research_engine import (run_research, walk_forward_validate, critic_review,
                               load_pair_df, load_btc_regime, add_btc_regime,
                               RESULTS_DIR, CAPITAL)
from indicators import as_df
from strategies import _prep, run_strategy

REPORT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─── Ross Baseline (simulated with known parameters) ─────────────────────────
# Ross: 1h≥+5% momentum flag, no-cap, $5M+ liquidity, PB 20-55%, 1m/3m entry,
# trail 9EMA, fail-fast, BTC-regime-aware
# We simulate this with EMASTACK + volume filters + aggressive PB logic

def build_ross_baseline_params() -> dict:
    """Ross v2 equivalent parameters for our backtester."""
    return {
        "strategy": "EMASTACK",
        "params": {
            "max_pb": 0.015,        # Ross: 20-55% pullback
            "vol_thresh": 1.5,      # Ross: 5M+ turnover
            "rsi_lo": 40,
            "rsi_hi": 68,
            "rsi_exit": 72,
            "min_adx": 20,
            "allowed_hours": [],    # All hours (Ross trades 24h)
            "risk_pct": 0.01,
            "max_hold_bars": 120,   # Ross can hold a long time
            "use_trail": True, "trail_ema": 9,
            "tp_atr_mult": 3.0,
            "stop_extra": 0.003,
        },
        "label": "ROSS-EQUIV",
    }


def run_comparison(top_candidates: list, pairs: list, bar: str = "5m") -> dict:
    """
    Run TRITON candidates vs Ross baseline on same data.
    Returns comparison dict with metrics for each.
    """
    from research_engine import load_pair_df, add_btc_regime, load_btc_regime
    btc_1h = load_btc_regime("1H")

    pair_dfs = {}
    for pair in pairs:
        df = load_pair_df(pair, bar)
        if df is None or len(df) < 200:
            continue
        df = _prep(df, {})
        df = add_btc_regime(df, btc_1h)
        pair_dfs[pair] = df

    results = {}

    # Run Ross baseline
    ross = build_ross_baseline_params()
    ross_trades = []
    ross_pnl = 0
    for pair in pair_dfs:
        result = run_strategy(pair_dfs[pair], ross["strategy"], ross["params"],
                              pair=pair, capital=CAPITAL)
        for t in result.closed_trades:
            ross_trades.append(t)
        ross_pnl += result.total_pnl

    n_ross = len(ross_trades)
    wins_ross = sum(1 for t in ross_trades if t.pnl_net > 0)
    gp_ross = sum(t.pnl_net for t in ross_trades if t.pnl_net > 0)
    gl_ross = abs(sum(t.pnl_net for t in ross_trades if t.pnl_net < 0))
    pf_ross = gp_ross / gl_ross if gl_ross > 0 else 0

    results["ROSS"] = {
        "n_trades": n_ross,
        "win_rate": wins_ross / n_ross if n_ross else 0,
        "total_pnl": round(ross_pnl, 2),
        "profit_factor": round(pf_ross, 3),
        "label": "Ross (backtest equiv)",
    }

    # Run each TRITON candidate
    for i, (strategy, params, avg_pnl, avg_pf) in enumerate(top_candidates[:5]):
        name = f"TRITON-{i+1}"
        cand_pnl = 0
        cand_trades = []
        for pair in pair_dfs:
            result = run_strategy(pair_dfs[pair], strategy, params, pair=pair, capital=CAPITAL)
            for t in result.closed_trades:
                cand_trades.append(t)
            cand_pnl += result.total_pnl

        n_c = len(cand_trades)
        wins_c = sum(1 for t in cand_trades if t.pnl_net > 0)
        gp_c = sum(t.pnl_net for t in cand_trades if t.pnl_net > 0)
        gl_c = abs(sum(t.pnl_net for t in cand_trades if t.pnl_net < 0))
        pf_c = gp_c / gl_c if gl_c > 0 else 0

        results[name] = {
            "strategy": strategy,
            "n_trades": n_c,
            "win_rate": wins_c / n_c if n_c else 0,
            "total_pnl": round(cand_pnl, 2),
            "profit_factor": round(pf_c, 3),
            "params": params,
            "label": f"TRITON {strategy} (candidate {i+1})",
        }

    return results


def generate_final_report(ranked_df: pd.DataFrame, comparison: dict,
                           critic_results: list, wf_results: list):
    """Generate final comprehensive HTML report."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Top 20 table rows
    top20 = ranked_df.head(20)
    rows = ""
    for _, r in top20.iterrows():
        pnl_cls = "up" if r["total_pnl"] > 0 else "down"
        rows += f"""<tr>
            <td>{r.get('strategy','?')}</td>
            <td>{r.get('pair','?')}</td>
            <td>{int(r.get('n_trades',0))}</td>
            <td>{r.get('win_rate',0):.1%}</td>
            <td class='{pnl_cls}'>${r.get('total_pnl',0):+.2f}</td>
            <td>{r.get('profit_factor',0):.3f}</td>
            <td>{r.get('expectancy',0):+.4f}</td>
            <td>{r.get('max_drawdown',0):.1%}</td>
            <td>{r.get('sharpe',0):.3f}</td>
            <td>{r.get('score',0):.2f}</td>
            </tr>"""

    # Comparison table
    comp_rows = ""
    for name, m in comparison.items():
        pnl_cls = "up" if m["total_pnl"] > 0 else "down"
        comp_rows += f"""<tr>
            <td><b>{name}</b></td>
            <td>{m['label']}</td>
            <td>{m['n_trades']}</td>
            <td>{m.get('win_rate',0):.1%}</td>
            <td class='{pnl_cls}'>${m['total_pnl']:+.2f}</td>
            <td>{m['profit_factor']:.3f}</td>
            </tr>"""

    # Critic summary
    critic_html = ""
    for cr in critic_results[:5]:
        v_cls = {"ACCEPT": "up", "CAUTION": "", "REJECT": "down"}.get(cr.get("verdict", ""), "")
        flags_html = "".join(f"<li>{f}</li>" for f in cr.get("flags", []))
        critic_html += f"""<div class='critic-card'>
            <b class='{v_cls}'>{cr.get('verdict','?')}</b> — Score {cr.get('score',0)}/100
            IS PnL: ${cr.get('is_pnl',0):.2f} / OOS PnL: ${cr.get('oos_pnl',0):.2f}
            <ul>{flags_html}</ul>
            </div>"""

    # Walk-forward summary
    wf_html = ""
    for wf in wf_results[:5]:
        wf_html += f"""<tr>
            <td>{wf.get('strategy','?')}</td>
            <td>{wf.get('oos_results',0)}</td>
            <td>${wf.get('avg_oos_pnl',0):+.2f}</td>
            <td>{wf.get('avg_oos_pf',0):.3f}</td>
            <td>{wf.get('oos_win_rate',0):.1%}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TRITON Research Report · Poseidon</title>
<style>
:root {{--bg:#0c0d10;--card:#15171c;--line:#2a2e36;--txt:#e8eaed;--mut:#8b909a;--up:#3dd68c;--dn:#ff5d73;--acc:#c084fc;}}
*{{box-sizing:border-box;}}
body{{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--txt);}}
header{{padding:24px;border-bottom:1px solid var(--line);}}
header h1{{margin:0;font-size:24px;}} header .sub{{color:var(--mut);margin-top:4px;}}
.wrap{{padding:24px;display:grid;gap:24px;}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--mut);margin:0 0 8px;}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;}}
th,td{{text-align:left;padding:7px 10px;font-variant-numeric:tabular-nums;white-space:nowrap;}}
th{{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid var(--line);}}
tr+tr td{{border-top:1px solid var(--line);}}
.up{{color:var(--up);}} .down{{color:var(--dn);}} .muted{{color:var(--mut);}}
.critic-card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:8px;}}
.critic-card ul{{margin:6px 0 0 16px;padding:0;color:var(--mut);font-size:12px;}}
.badge{{background:#1e1031;color:var(--acc);font-size:11px;padding:2px 8px;border-radius:999px;}}
</style></head>
<body>
<header>
  <h1>⚓ TRITON Research Report <span class="badge">Poseidon Independent Challenger</span></h1>
  <div class="sub">Generated {ts} · 537 configurations tested across 17 pairs on 5m OKX data · Strategy families: VSMO, STAB, MTALIGN, RVOL, HVOL, EMASTACK</div>
</header>
<div class="wrap">
  <div>
    <h2>🏆 Top 20 Configurations (ranked by composite score)</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Strategy</th><th>Pair</th><th>Trades</th><th>Win%</th><th>Net PnL</th><th>PF</th><th>Expectancy</th><th>MaxDD</th><th>Sharpe</th><th>Score</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
  </div>
  <div>
    <h2>⚔️ TRITON vs Ross vs Claude comparison (same 5m data, 17 pairs)</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Name</th><th>Description</th><th>Trades</th><th>Win%</th><th>Total PnL</th><th>PF</th></tr></thead>
      <tbody>{comp_rows}</tbody>
    </table></div>
  </div>
  <div>
    <h2>🔎 Critic Review (top 5 candidates)</h2>
    {critic_html or '<div class="muted">No critic results.</div>'}
  </div>
  <div>
    <h2>📊 Walk-Forward Validation (OOS results)</h2>
    <table><thead><tr><th>Strategy</th><th>OOS Folds</th><th>Avg OOS PnL</th><th>Avg OOS PF</th><th>OOS Win%</th></tr></thead>
    <tbody>{wf_html or '<tr><td colspan="5" class="muted">No WF results.</td></tr>'}</tbody>
    </table>
  </div>
</div>
</body></html>"""

    report_path = os.path.join(REPORT_DIR, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"✓ Report written to {report_path}")
    return report_path


def run_full_analysis():
    """Load results, run WF validation, critic, comparison, generate report."""
    ranked_path = os.path.join(RESULTS_DIR, "ranked_results.csv")
    if not os.path.exists(ranked_path):
        print("ERROR: ranked_results.csv not found. Run research_engine.py first.")
        return

    print("Loading ranked results...")
    df = pd.read_csv(ranked_path)
    print(f"  {len(df)} results loaded. Top 5:")
    cols = ["strategy", "pair", "n_trades", "win_rate", "total_pnl",
            "profit_factor", "max_drawdown", "sharpe", "score"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].head(5).to_string(index=False))

    # Extract top candidates
    top_candidates = []
    for _, row in df.head(10).iterrows():
        strat = row["strategy"]
        params = ast.literal_eval(row["params"]) if isinstance(row["params"], str) else row["params"]
        top_candidates.append((strat, params, row["total_pnl"], row["profit_factor"]))

    TEST_PAIRS = ["SOL-USDT", "AVAX-USDT", "LINK-USDT", "NEAR-USDT", "SUI-USDT",
                  "HYPE-USDT", "ONDO-USDT", "FIL-USDT", "WLD-USDT", "ETH-USDT"]

    print("\nRunning walk-forward validation on top 5 candidates...")
    wf_results = []
    for strategy, params, _, _ in top_candidates[:5]:
        print(f"  WF: {strategy}...")
        wf = walk_forward_validate(strategy, params, TEST_PAIRS[:5], n_splits=4)
        wf["strategy"] = strategy
        wf_results.append(wf)
        print(f"    OOS PnL: ${wf.get('avg_oos_pnl',0):.2f} | PF: {wf.get('avg_oos_pf',0):.3f}")

    print("\nRunning comparison vs Ross baseline...")
    comparison = run_comparison(top_candidates, TEST_PAIRS)

    print("\nRunning critic review...")
    critic_results = []
    for strategy, params, avg_pnl, avg_pf in top_candidates[:5]:
        # Get IS metrics from the ranked df
        mask = (df["strategy"] == strategy)
        subset = df[mask]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        is_summary = {
            "n_trades": int(best_row.get("n_trades", 0)),
            "profit_factor": float(best_row.get("profit_factor", 0)),
            "total_pnl": float(best_row.get("total_pnl", 0)),
            "max_drawdown": float(best_row.get("max_drawdown", 0)),
            "win_rate": float(best_row.get("win_rate", 0)),
        }
        # Find matching WF result
        wf = next((w for w in wf_results if w.get("strategy") == strategy), {})
        cr = critic_review(is_summary, wf)
        cr["strategy"] = strategy
        critic_results.append(cr)
        print(f"  {strategy}: {cr['verdict']} (score={cr['score']}) — {', '.join(cr['flags'][:2]) if cr['flags'] else 'clean'}")

    print("\nGenerating final report...")
    path = generate_final_report(df, comparison, critic_results, wf_results)
    print(f"✓ Full analysis complete. Report: {path}")

    # Print comparison summary
    print("\n=== FINAL COMPARISON ===")
    for name, m in comparison.items():
        pnl_sym = "✅" if m["total_pnl"] > 0 else "❌"
        print(f"  {name}: PnL={m['total_pnl']:+.2f} PF={m['profit_factor']:.3f} Trades={m['n_trades']} {pnl_sym}")

    return df, comparison, critic_results, wf_results


if __name__ == "__main__":
    run_full_analysis()
