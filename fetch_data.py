"""
OKX Historical Data Fetcher — Poseidon Research Engine
Uses /market/history-candles with backward after= pagination and User-Agent header.
Based on Zeus's ~/crypto_momentum/engine/okx_data.py approach — no auth required.
"""
import json, time, os, gzip, datetime as dt, urllib.request

BASE = "https://www.okx.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STABLES = {"USDC","USDT","USDG","DAI","TUSD","FDUSD","USDP","PYUSD","EURT","USDE","XAUT","PAXG"}

# Liquid pairs to fetch (same list as before, known to have history)
PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
    "DOGE-USDT", "ADA-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT",
    "LTC-USDT", "UNI-USDT", "NEAR-USDT", "FIL-USDT", "ONDO-USDT",
    "PEPE-USDT", "WLD-USDT", "SUI-USDT", "TIA-USDT", "HYPE-USDT",
    "ZEC-USDT", "XLM-USDT", "TRUMP-USDT", "LIT-USDT", "SAND-USDT",
    "STRK-USDT", "PUMP-USDT", "ENA-USDT", "OKB-USDT", "APT-USDT",
]

BARS = ["1H", "5m", "15m", "4H", "1D"]


def _get(url, tries=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def fetch_candles_deep(inst_id: str, bar: str, days: int = 92, pause: float = 0.15) -> list:
    """
    Paginate backwards via after= until we cover `days` of history.
    Returns list of confirmed-only candle dicts, sorted oldest→newest.
    Matches Zeus's okx_data.py approach exactly.
    """
    want_from = int((dt.datetime.utcnow() - dt.timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    after = ""
    pages = 0

    while True:
        url = f"{BASE}/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit=100"
        if after:
            url += f"&after={after}"
        js = _get(url)
        data = js.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        oldest = int(data[-1][0])
        after = str(oldest)
        pages += 1
        if oldest <= want_from:
            break
        if len(all_rows) > 70000:  # safety cap
            break
        time.sleep(pause)

    if not all_rows:
        return []

    # Filter: confirmed candles only (no lookahead on live bar) — matches Zeus
    confirmed = [r for r in all_rows if r[8] == "1"]
    # Remove duplicates, sort oldest→newest
    seen = set()
    unique = []
    for r in confirmed:
        ts = r[0]
        if ts not in seen:
            seen.add(ts)
            unique.append(r)
    unique.sort(key=lambda x: int(x[0]))
    # Trim to requested window
    unique = [r for r in unique if int(r[0]) >= want_from]
    return unique


def save_data(inst_id: str, bar: str, data: list):
    safe_name = inst_id.replace("-", "_")
    path = os.path.join(DATA_DIR, f"{safe_name}_{bar}.json.gz")
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return path


def load_data(inst_id: str, bar: str) -> list:
    safe_name = inst_id.replace("-", "_")
    path = os.path.join(DATA_DIR, f"{safe_name}_{bar}.json.gz")
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt") as f:
        return json.load(f)


def candles_to_dicts(data: list) -> list:
    result = []
    for c in data:
        result.append({
            "ts": int(c[0]),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "vol": float(c[5]),
            "vol_usdt": float(c[6]),
            "confirmed": True,  # always True — we filter on fetch
        })
    return result


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 92
    force = "--force" in sys.argv

    print(f"Fetching OKX data: {len(PAIRS)} pairs × {len(BARS)} bars | {days} days history")
    print(f"Method: /market/history-candles + after= pagination + User-Agent (Zeus's approach)")
    print()
    errors = []
    count = 0

    for inst_id in PAIRS:
        for bar in BARS:
            safe = inst_id.replace("-", "_")
            path = os.path.join(DATA_DIR, f"{safe}_{bar}.json.gz")
            if os.path.exists(path) and not force:
                # Check if it's the old shallow data (< 500 bars on 1H = < 20 days)
                try:
                    existing = load_data(inst_id, bar)
                    if bar == "1H" and len(existing) < 1500:
                        print(f"  [refresh] {inst_id}/{bar}: only {len(existing)} bars (shallow), re-fetching")
                    elif bar == "5m" and len(existing) < 20000:
                        print(f"  [refresh] {inst_id}/{bar}: only {len(existing)} bars (shallow), re-fetching")
                    else:
                        print(f"  [skip] {inst_id}/{bar}: {len(existing)} bars cached")
                        count += 1
                        continue
                except Exception:
                    pass

            try:
                data = fetch_candles_deep(inst_id, bar, days=days)
                if len(data) < 10:
                    print(f"  ✗ {inst_id}/{bar}: only {len(data)} bars returned")
                    errors.append((inst_id, bar, f"only {len(data)} bars"))
                    count += 1
                    continue
                save_data(inst_id, bar, data)
                oldest_ts = int(data[0][0])
                newest_ts = int(data[-1][0])
                oldest_str = dt.datetime.utcfromtimestamp(oldest_ts / 1000).strftime("%Y-%m-%d")
                newest_str = dt.datetime.utcfromtimestamp(newest_ts / 1000).strftime("%Y-%m-%d")
                span = (newest_ts - oldest_ts) / 1000 / 86400
                print(f"  ✓ {inst_id}/{bar}: {len(data)} bars | {oldest_str} → {newest_str} ({span:.0f}d)")
                count += 1
            except Exception as e:
                print(f"  ✗ {inst_id}/{bar}: {e}")
                errors.append((inst_id, bar, str(e)))
                count += 1

    print(f"\nDone: {count} datasets. Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
