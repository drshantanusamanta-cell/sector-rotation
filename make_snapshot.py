"""Build the warm-start snapshot (data/seed/) on your own computer.

    python make_snapshot.py                  # fetch / top up everything, then write data/seed/
    python make_snapshot.py --no-ownership   # skip the FII/DII shareholding pages (much faster)

Then upload the data/seed folder to the GitHub repo (replace the old files).
Run it weekly. It fetches 10 years of PE history and 11 years of index prices,
so any lookback chosen in the app is covered. The first run with ownership reads one
Screener page per stock (~550 pages, about 20 minutes at a polite pace); later runs only
re-read stocks older than three weeks.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import data_sources as ds
import store
from sectors import BENCHMARKS, INDICES

PE_YEARS = 10
INDEX_YEARS = 11
STOCK_YEARS = 2


def main() -> int:
    t0 = time.time()
    defs = list({d.key: d for d in [*INDICES, *BENCHMARKS.values()]}.values())

    print("Constituents…")
    symbols = set()
    for d in defs:
        try:
            df, src = ds.fetch_constituents(d)
            symbols |= set(df["Symbol"])
            print(f"  {d.label:22s} {len(df):4d}  {src}")
        except Exception as e:  # noqa: BLE001
            print(f"  {d.label:22s} FAILED  {e}")

    print("PE / PB / DY history…")
    for d in defs:
        try:
            df, src = ds.fetch_pe_history(d, PE_YEARS)
            print(f"  {d.label:22s} {len(df):5d} days  {src}")
        except Exception as e:  # noqa: BLE001
            print(f"  {d.label:22s} FAILED  {e}")

    print("Index prices…")
    px, src = ds.fetch_index_prices_yf(defs, INDEX_YEARS)
    print(f"  Yahoo: {len(px.columns)} indices  {src}")
    for d in defs:
        if d.key not in px.columns:
            try:
                s, lab = ds.fetch_index_price_ni(d, INDEX_YEARS)
                print(f"  {d.label:22s} {len(s):5d} days  {lab}")
            except Exception as e:  # noqa: BLE001
                print(f"  {d.label:22s} FAILED  {e}")

    print(f"Prices for {len(symbols)} stocks (a few minutes)…")
    stocks, src = ds.fetch_stock_prices(sorted(symbols), STOCK_YEARS)
    print(f"  {stocks.shape[1]} stocks  {src}")

    print("FPI sector flows (NSDL)…")
    try:
        fpi, src = ds.fetch_fpi_flows(24)
        print(f"  {fpi['date'].nunique()} fortnights  {src}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED  {e}")

    if "--no-ownership" not in sys.argv:
        todo = ds.stale_ownership(sorted(symbols))
        print(f"FII/DII ownership from Screener.in: {len(todo)} stocks to read…")
        res = ds.refresh_ownership(sorted(symbols), budget_s=3600,
                                   progress=lambda i, n, s: print(f"  {i}/{n} {s}", end="\r", flush=True))
        print(f"\n  updated {res['done']}, left {res['left']}" + (f"  (stopped: {res['blocked']})" if res["blocked"] else ""))

    out = Path(__file__).parent / "data" / "seed"
    files = store.export_seed(out)
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nWrote {len(files)} files ({size:.1f} MB) to {out} in {time.time() - t0:.0f} s.")
    print("Upload the data/seed folder to GitHub to update the app's warm start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
