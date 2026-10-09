"""
TRITON Paper Trader — Live paper trading using best strategy from research.
Polls OKX every 60 seconds, writes state to JSON, generates HTML blotter.
"""
import sys, os, json, time, signal, gzip
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import fetch_candles_deep, candles_to_dicts
from indicators import as_df, ema, atr, rsi, macd, volume_ratio, adx, rate_of_change
from strategies import _prep
import pandas as pd
import numpy as np

STATE_FILE = os.path.join(os.path.dirname(__file__), "paper_state.json")
BLOTTER_FILE = os.path.join(os.path.dirname(__file__), "index.html")

CAPITAL = 3000.0
MAX_SELLS = 100
STRATEGY_NAME = "TRITON-HVOL"
STRATEGY_VERSION = "triton-hvol-v1"
LABEL = "EXPERIMENTAL · UNPROVEN · triton-hvol-v1"

# ─── Best params from research (HVOL — highest validated performer) ──────────
BEST_PARAMS = {
    "min_adx": 30,
    "vol_thresh": 1.5,
    "max_ext": 0.02,
    "adx_exit": 20,
    "allowed_hours": [],     # 24h — ENA-style global market
    "risk_pct": 0.01,
    "max_hold_bars": 120,    # max 120h (5 days) on 1H bars
    "use_trail": False,
    "tp_atr_mult": 2.5,
}

PAIRS_SCAN = [
    "SOL-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "NEAR-USDT",
    "SUI-USDT", "UNI-USDT", "HYPE-USDT", "ONDO-USDT", "FIL-USDT",
    "WLD-USDT", "DOGE-USDT", "PEPE-USDT", "ADA-USDT", "XLM-USDT",
    "APT-USDT", "ENA-USDT", "STRK-USDT", "LIT-USDT",
]

FEE_RATE = 0.0008
SLIP = 0.0005


# ─── State Management ────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "equity": CAPITAL,
        "open_position": None,
        "closed_trades": [],
        "tape": [],
        "total_sells": 0,
    }


def save_state(state: dict):
    """Atomic write via temp file + os.replace() — safe against mid-write crashes."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ─── Signal Generation ───────────────────────────────────────────────────────

def compute_signal(pair: str, params: dict, state: dict) -> dict:
    """Fetch 1H candles and compute entry/exit signal for a pair.
    Uses only confirmed (closed) candles — matches backtester behaviour exactly.
    fetch_candles_deep filters confirm=='1' so no live/partial bar leaks through.
    """
    try:
        from fetch_data import fetch_candles_deep, candles_to_dicts as deep_to_dicts
        raw = fetch_candles_deep(pair, bar="1H", days=5, pause=0.05)   # ~120 confirmed 1H bars
        if not raw or len(raw) < 60:
            return {"signal": 0, "pair": pair, "error": "insufficient_data"}
        dicts = deep_to_dicts(raw)
        df = as_df(dicts)
        df = _prep(df, params)
        df["btc_regime"] = 1  # simplified for live
        # iloc[-1] is the last confirmed closed candle (no lookahead risk)
        row = df.iloc[-1].to_dict()
        prev = df.iloc[-2].to_dict()
        from strategies import signal_hvol
        sig, stop = signal_hvol(row, prev, params)
        return {
            "signal": sig,
            "pair": pair,
            "price": row["close"],
            "stop": stop,
            "adx14": row.get("adx14", 0),
            "vr20": row.get("vr20", 0),
            "rsi14": row.get("rsi14", 0),
            "don10_hi": row.get("don10_hi", 0),
        }
    except Exception as e:
        return {"signal": 0, "pair": pair, "error": str(e)}


# ─── Trade Execution (Paper) ─────────────────────────────────────────────────

def paper_buy(state: dict, pair: str, price: float, stop: float) -> dict:
    """Execute a paper buy."""
    fill = price * (1 + SLIP)
    risk_amt = state["equity"] * 0.01
    stop_dist = abs(fill - stop) / fill
    if stop_dist <= 0:
        stop_dist = 0.01
    qty = risk_amt / (stop_dist * fill)
    max_qty = (state["equity"] * 0.20) / fill
    qty = min(qty, max_qty)
    fee = fill * qty * FEE_RATE
    state["equity"] -= (fill * qty + fee)
    state["open_position"] = {
        "pair": pair, "entry": fill, "qty": qty,
        "stop": stop, "fee_in": fee,
        "mfe": 0.0, "mae": 0.0,
        "entry_ts": int(time.time() * 1000),
        "bars": 0,
    }
    trade_entry = f"BUY {pair} @ {fill:.6g}"
    state["tape"].insert(0, {"side": "BUY", "pair": pair, "price": fill, "note": "hvol_entry",
                              "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")})
    state["tape"] = state["tape"][:50]
    return state


def paper_sell(state: dict, price: float, reason: str) -> dict:
    """Execute a paper sell."""
    pos = state["open_position"]
    if pos is None:
        return state
    fill = price * (1 - SLIP)
    fee = fill * pos["qty"] * FEE_RATE
    proceeds = fill * pos["qty"] - fee
    pnl_raw = (fill - pos["entry"]) * pos["qty"]
    pnl_net = pnl_raw - pos["fee_in"] - fee
    pct = (fill - pos["entry"]) / pos["entry"] * 100
    state["equity"] += proceeds
    state["closed_trades"].append({
        "pair": pos["pair"],
        "entry": pos["entry"],
        "exit": fill,
        "qty": pos["qty"],
        "pnl": round(pnl_net, 4),
        "pct": round(pct, 4),
        "reason": reason,
        "mfe": pos["mfe"],
        "mae": pos["mae"],
        "ts_exit": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
    })
    state["tape"].insert(0, {"side": "SELL", "pair": pos["pair"], "price": fill,
                              "note": reason, "pnl": round(pnl_net, 4), "pct": round(pct, 4),
                              "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")})
    state["tape"] = state["tape"][:50]
    state["total_sells"] = len(state["closed_trades"])
    state["open_position"] = None
    return state


# ─── Git Push ────────────────────────────────────────────────────────────────

REPO_DIR = os.path.dirname(__file__)
_last_push_ts = 0.0

def _push_blotter():
    """Commit and push index.html to pixel-vector-bell/okx-triton-hvol-paper."""
    global _last_push_ts
    now = time.time()
    # Throttle: don't push more than once per 60s
    if now - _last_push_ts < 58:
        return
    try:
        import subprocess
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subprocess.run(
            ["git", "-C", REPO_DIR, "add", "index.html", "paper_state.json"],
            capture_output=True, timeout=15
        )
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "commit", "-m", f"blotter {ts}"],
            capture_output=True, text=True, timeout=15
        )
        if "nothing to commit" in result.stdout + result.stderr:
            return
        subprocess.run(
            ["git", "-C", REPO_DIR, "push", "origin", "main"],
            capture_output=True, timeout=30
        )
        with open(os.path.join(REPO_DIR, ".last_push"), "w") as f:
            f.write(str(now))
        _last_push_ts = now
    except Exception as e:
        print(f"  [push] error: {e}")


# ─── Blotter HTML Generator ──────────────────────────────────────────────────

def gen_blotter(state: dict, watchlist: list):
    trades = state["closed_trades"]
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / n * 100 if n > 0 else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
    total_pnl = sum(t["pnl"] for t in trades)
    equity = state["equity"]
    ret = (equity / CAPITAL - 1) * 100
    ret_cls = "up" if ret >= 0 else "down"
    pos = state.get("open_position")
    ts_now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Open position
    pos_html = ""
    if pos:
        from fetch_data import fetch_candles_deep, candles_to_dicts
        try:
            mk_raw = fetch_candles_deep(pos["pair"], bar="1m", days=1)
            mk = float(mk_raw[-1][4]) if mk_raw else pos["entry"]
        except:
            mk = pos["entry"]
        upnl = (mk - pos["entry"]) * pos["qty"]
        upnl_pct = (mk / pos["entry"] - 1) * 100
        upnl_cls = "up" if upnl >= 0 else "down"
        pos_html = f"""<tr><td class='sym'>{pos['pair']}</td>
            <td>{pos['entry']:.6g}</td><td>{mk:.6g}</td>
            <td><span class='{upnl_cls}'>{upnl_pct:+.2f}%</span></td>
            <td>rvol_entry</td>
            <td><span class='up'>+{pos['mfe']:.2f}%</span></td>
            <td><span class='down'>-{pos['mae']:.2f}%</span></td>
            <td>{pos['stop']:.6g}</td></tr>"""
    else:
        pos_html = "<tr><td colspan='8' class='muted'>Flat — scanning for TRITON-RVOL entry.</td></tr>"

    # Watchlist
    wl_html = ""
    if watchlist:
        for w in watchlist[:20]:
            vr_cls = "up" if w.get("vr20", 0) >= BEST_PARAMS["vol_thresh"] else "muted"
            sig_cls = "up" if w.get("signal", 0) == 1 else ("down" if w.get("signal", 0) == -1 else "muted")
            sig_label = "ENTRY" if w.get("signal", 0) == 1 else "—"
            wl_html += f"""<tr><td class='sym'>{w['pair']}</td>
                <td class='{vr_cls}'>{w.get('vr20', 0):.2f}×</td>
                <td>{w.get('roc5', 0):+.2f}%</td>
                <td>{w.get('rsi14', 0):.0f}</td>
                <td>{w.get('adx14', 0):.0f}</td>
                <td class='{sig_cls}'>{sig_label}</td></tr>"""
    else:
        wl_html = "<tr><td colspan='6' class='muted'>Scanning…</td></tr>"

    # Tape
    tape_html = ""
    for t in state["tape"][:30]:
        side_cls = "up" if t["side"] == "BUY" else "down"
        pnl_str = ""
        if "pnl" in t:
            pnl_cls = "up" if t["pnl"] >= 0 else "down"
            pnl_str = f" <span class='{pnl_cls}'>{t['pct']:+.2f}%</span> ${t['pnl']:.2f}"
        tape_html += f"<div class='tape'><span class='{side_cls}'>{t['side']}</span> {t['pair']} @ {t['price']:.6g} <span class='muted'>{t['note']}</span>{pnl_str}</div>"
    if not tape_html:
        tape_html = "<div class='muted'>No fills yet.</div>"

    sells_count = state["total_sells"]
    frozen_label = f"frozen to {MAX_SELLS} sells" if sells_count >= MAX_SELLS else f"{sells_count}/{MAX_SELLS} sells"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="60"/>
<title>OKX TRITON · paper</title>
<style>
  :root {{ --bg:#0c0d10; --card:#15171c; --line:#2a2e36; --txt:#e8eaed; --mut:#8b909a; --up:#3dd68c; --dn:#ff5d73; --accent:#c084fc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.45 ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--txt); }}
  header {{ padding:20px 24px 8px; border-bottom:1px solid var(--line); }}
  header h1 {{ margin:0; font-size:20px; letter-spacing:.04em; }}
  header .sub {{ color:var(--mut); font-size:13px; margin-top:4px; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; padding:16px 24px; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .stat .k {{ color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
  .stat .v {{ font-size:22px; font-variant-numeric:tabular-nums; margin-top:4px; }}
  .up {{ color:var(--up); }} .down {{ color:var(--dn); }}
  .wrap {{ padding:0 24px 32px; display:grid; gap:18px; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.1em; color:var(--mut); margin:8px 0; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:8px 12px; font-variant-numeric:tabular-nums; }}
  th {{ font-size:11px; color:var(--mut); text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid var(--line); }}
  tr+tr td {{ border-top:1px solid var(--line); }}
  .sym {{ font-weight:650; }} .muted {{ color:var(--mut); }}
  .tape {{ padding:6px 0; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }}
  .paper {{ display:inline-block; background:#1e1031; color:var(--accent); font-size:11px; padding:2px 8px; border-radius:999px; margin-left:8px; }}
</style>
</head>
<body>
<header>
  <h1>OKX TRITON-RVOL <span class="paper">PAPER · {LABEL} · {frozen_label}</span></h1>
  <div class="sub">Updated {ts_now} · Poseidon independent challenger · Relative Volume Leader · 5m candles · EMA trend + ADX + vol surge · allowed hours {BEST_PARAMS['allowed_hours']} UTC · risk 1%/trade · fee 0.08%+slip 0.05%</div>
</header>
<div class="grid">
  <div class="stat"><div class="k">Equity</div><div class="v">${equity:,.2f}</div></div>
  <div class="stat"><div class="k">Return</div><div class="v"><span class="{ret_cls}">{ret:+.2f}%</span></div></div>
  <div class="stat"><div class="k">Open</div><div class="v">{"1" if pos else "0"}/1</div></div>
  <div class="stat"><div class="k">Sells / wins / PF</div><div class="v">{n} / {len(wins)} / {pf:.2f}</div></div>
</div>
<div class="wrap">
  <div>
    <h2>Open position</h2>
    <table><thead><tr><th>Pair</th><th>Entry</th><th>Mark</th><th>uPnL</th><th>Style</th><th>MFE</th><th>MAE</th><th>Stop</th></tr></thead>
    <tbody>{pos_html}</tbody></table>
  </div>
  <div>
    <h2>TRITON-RVOL watchlist · {len(watchlist)} scanned</h2>
    <table><thead><tr><th>Pair</th><th>Vol Ratio</th><th>ROC 5m</th><th>RSI</th><th>ADX</th><th>Signal</th></tr></thead>
    <tbody>{wl_html}</tbody></table>
  </div>
  <div>
    <h2>Tape (latest 30)</h2>
    {tape_html}
  </div>
</div>
</body></html>"""
    with open(BLOTTER_FILE, "w") as f:
        f.write(html)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print(f"TRITON Paper Trader starting — {STRATEGY_NAME} v1")
    print(f"Capital: ${CAPITAL}, Max sells: {MAX_SELLS}")
    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False
        print("\nStopping gracefully...")

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    state = load_state()
    params = BEST_PARAMS

    while running:
        loop_start = time.time()
        try:
            now_utc = datetime.now(tz=timezone.utc)
            print(f"\n[{now_utc.strftime('%H:%M:%S')}] Scanning {len(PAIRS_SCAN)} pairs...")

            # Check if we're at max sells
            if state["total_sells"] >= MAX_SELLS:
                print(f"Max sells ({MAX_SELLS}) reached — frozen.")
                gen_blotter(state, [])
                time.sleep(60)
                continue

            # Scan all pairs for signals
            watchlist = []
            for pair in PAIRS_SCAN:
                sig_data = compute_signal(pair, params, state)
                sig_data["pair"] = pair
                watchlist.append(sig_data)
                time.sleep(0.1)

            # Sort by vol ratio × ROC
            watchlist.sort(key=lambda x: x.get("vr20", 0) * abs(x.get("roc5", 0)), reverse=True)

            # Handle open position
            if state["open_position"]:
                pos = state["open_position"]
                # Compute bars_open from entry_ts — survives restarts correctly
                now_ms = int(time.time() * 1000)
                bar_ms = 3600 * 1000  # 1H in ms
                bars_open = max(1, (now_ms - pos.get("entry_ts", now_ms)) // bar_ms)
                pos["bars"] = bars_open
                # Check exit on the current pair
                sig_data = next((w for w in watchlist if w["pair"] == pos["pair"]), None)
                if sig_data:
                    price = sig_data["price"]
                    # Update MFE/MAE
                    # Stop check
                    if price <= pos["stop"]:
                        state = paper_sell(state, price, "stop")
                        print(f"  STOP LOSS: {pos['pair']} @ {price:.4f}")
                    # Exit signal
                    elif sig_data["signal"] == -1:
                        state = paper_sell(state, price, "signal")
                        print(f"  EXIT SIGNAL: {pos['pair']} @ {price:.4f}")
                    # Timeout
                    elif pos["bars"] >= params.get("max_hold_bars", 72):
                        state = paper_sell(state, price, "timeout")
                        print(f"  TIMEOUT: {pos['pair']} @ {price:.4f}")

            # Look for new entry (if flat)
            if state["open_position"] is None:
                for sig_data in watchlist:
                    if sig_data.get("signal") == 1:
                        pair = sig_data["pair"]
                        price = sig_data["price"]
                        stop = sig_data.get("stop") or (price * 0.98)
                        print(f"  ENTRY: {pair} @ {price:.4f} | VOL={sig_data.get('vr20',0):.2f}× ROC={sig_data.get('roc5',0):+.2f}%")
                        state = paper_buy(state, pair, price, stop)
                        break

            save_state(state)
            gen_blotter(state, watchlist)
            _push_blotter()

            n = len(state["closed_trades"])
            equity = state["equity"]
            ret = (equity / CAPITAL - 1) * 100
            print(f"  Equity: ${equity:,.2f} ({ret:+.2f}%) | Trades: {n} | Open: {'YES' if state['open_position'] else 'NO'}")

        except Exception as e:
            print(f"  ERROR in main loop: {e}")

        # Sleep for remainder of 60s
        elapsed = time.time() - loop_start
        sleep_time = max(5, 60 - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
