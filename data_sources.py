"""Live data fetchers for the Sector Rotation Screener.

Sources and the order they are tried:
    constituents   niftyindices.com CSV  -> nseindia.com API -> data/constituents/<file>.csv
    PE / PB / DY   niftyindices.com history -> data/pe_history.csv (or an uploaded CSV)
    index prices   yfinance -> niftyindices.com history -> equal-weight proxy (built in engine)
    stock prices   yfinance (batched)
    quarterly      screener.in company page (drill-down only, on demand)

Every request is paced and retried by net.py, and every result is kept in the disk
cache (store.py) so stale data is topped up instead of downloaded again. If a live
fetch fails, the last good copy is used and labelled as cached.

Parsing is kept separate from network code so it can be tested offline.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import difflib
import io
import json
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sectors import IndexDef

import net
import store

UA = net.UA
TIMEOUT = net.TIMEOUT
NI_BASE = "https://www.niftyindices.com"
NSE_BASE = "https://www.nseindia.com"
DATA_DIR = Path(__file__).parent / "data"

# how long cached data is used before going back to the source
TTL_CONS = 7 * 86400          # index constituents change twice a year
TTL_HIST = 12 * 3600          # PE/PB/DY and niftyindices prices: top up twice a day
TTL_PX = 6 * 3600             # Yahoo prices: top up every 6 h
FULL_PX_REFRESH = 7 * 86400   # full Yahoo re-download weekly (re-applies dividend adjustments)
TTL_SCREENER = 24 * 3600

NI_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": NI_BASE,
    "Referer": f"{NI_BASE}/reports/historical-data",
}


class DataError(Exception):
    """Raised with a user-readable message when data cannot be obtained."""


_PROBE: dict[str, tuple[float, bool]] = {}


def site_up(base: str, ttl: int = 600) -> bool:
    """Quick reachability probe so a blocked site fails once, not once per index."""
    now = time.time()
    hit = _PROBE.get(base)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        ok = net.get(base, timeout=6).status_code < 500
    except Exception:  # noqa: BLE001  (unreachable, or Blocked / cooling down)
        ok = False
    _PROBE[base] = (now, ok)
    return ok


_NSE_WARM = {"ts": 0.0}


def _nse_warm() -> None:
    """NSE's API needs the cookies its home page sets; refresh them every 10 minutes."""
    if time.time() - _NSE_WARM["ts"] > 600:
        net.get(NSE_BASE, headers={"Accept": "text/html"})
        _NSE_WARM["ts"] = time.time()


def _cached_label(label: str, age: float) -> str:
    return f"{label} · cached {store.age_text(age)} ago (live fetch failed)"


def _merge(old, new):
    """Union of two date-indexed frames/series; values from `new` win, NaNs in `new` never erase old data."""
    if old is None or len(old) == 0:
        return new
    if new is None or len(new) == 0:
        return old
    if isinstance(old, pd.Series):
        out = pd.concat([old[~old.index.isin(new.dropna().index)], new.dropna()])
        return out[~out.index.duplicated(keep="last")].sort_index()
    out = old.reindex(index=old.index.union(new.index), columns=old.columns.union(new.columns))
    out.update(new)
    return out.sort_index()


# ============================================================================
# Parsers (offline-testable)
# ============================================================================
def _norm_symbol(sym: str) -> str:
    return str(sym).strip().upper()


def parse_constituents_csv(text: str) -> pd.DataFrame:
    """niftyindices constituent CSV -> DataFrame[Symbol, Company, Industry]."""
    df = pd.read_csv(io.StringIO(text))
    cols = {c.lower().strip(): c for c in df.columns}
    sym = cols.get("symbol")
    if sym is None:
        raise DataError("Constituent file has no 'Symbol' column.")
    out = pd.DataFrame({
        "Symbol": df[sym].map(_norm_symbol),
        "Company": df[cols["company name"]] if "company name" in cols else df[sym],
        "Industry": df[cols["industry"]] if "industry" in cols else "",
    })
    out = out[out["Symbol"].str.len() > 0].drop_duplicates("Symbol").reset_index(drop=True)
    if out.empty:
        raise DataError("Constituent file is empty.")
    return out


def parse_nse_index_json(payload: dict, index_name: str) -> pd.DataFrame:
    """nseindia.com /api/equity-stockIndices payload -> constituents."""
    rows = []
    for r in payload.get("data", []):
        sym = _norm_symbol(r.get("symbol", ""))
        if not sym or sym == _norm_symbol(index_name):
            continue  # first row is the index itself
        meta = r.get("meta") or {}
        rows.append({"Symbol": sym, "Company": meta.get("companyName", sym),
                     "Industry": meta.get("industry", "")})
    if not rows:
        raise DataError("NSE returned no constituents.")
    return pd.DataFrame(rows).drop_duplicates("Symbol").reset_index(drop=True)


def parse_ni_response(payload) -> list[dict]:
    """niftyindices /BackPage responses are a JSON list, or older-style {"d": "<json string>"}."""
    if isinstance(payload, dict) and "d" in payload:
        payload = payload["d"]
    if isinstance(payload, str):
        payload = json.loads(payload) if payload.strip() else []
    if not isinstance(payload, list):
        return []
    return [r for r in payload if isinstance(r, dict)]


def _find_key(keys, *cands, contains: str | None = None):
    low = {k.lower().replace(" ", ""): k for k in keys}
    for c in cands:
        if c in low:
            return low[c]
    if contains:
        for k_low, k in low.items():
            if contains in k_low:
                return k
    return None


def _to_float(v) -> float:
    if v is None:
        return np.nan
    t = str(v).replace(",", "").replace("%", "").strip()
    if t in ("", "-", "--", "—", "NA", "N/A"):
        return np.nan
    try:
        return float(t)
    except ValueError:
        return np.nan


def parse_pe_records(records: list[dict]) -> pd.DataFrame:
    """PE/PB/DY records -> DataFrame(index=Date, columns=[pe, pb, dy])."""
    if not records:
        return pd.DataFrame(columns=["pe", "pb", "dy"])
    keys = records[0].keys()
    kd = _find_key(keys, "date", "historicaldate", contains="date")
    kpe = _find_key(keys, "pe", "p/e")
    kpb = _find_key(keys, "pb", "p/b")
    kdy = _find_key(keys, "divyield", "dy", contains="div")
    if kd is None or kpe is None:
        raise DataError("Unexpected PE history format.")
    df = pd.DataFrame({
        "Date": pd.to_datetime([r.get(kd) for r in records], dayfirst=True, errors="coerce", format="mixed"),
        "pe": [_to_float(r.get(kpe)) for r in records],
        "pb": [_to_float(r.get(kpb)) for r in records] if kpb else np.nan,
        "dy": [_to_float(r.get(kdy)) for r in records] if kdy else np.nan,
    })
    df = df.dropna(subset=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    df.loc[(df["pe"] <= 0) | (df["pe"] > 500), "pe"] = np.nan
    return df


def parse_price_records(records: list[dict]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    keys = records[0].keys()
    kd = _find_key(keys, "historicaldate", "date", contains="date")
    kc = _find_key(keys, "close", contains="close")
    if kd is None or kc is None:
        raise DataError("Unexpected price history format.")
    s = pd.Series([_to_float(r.get(kc)) for r in records],
                  index=pd.to_datetime([r.get(kd) for r in records], dayfirst=True, errors="coerce", format="mixed"))
    s = s[~s.index.isna()]
    return s[~s.index.duplicated()].sort_index().dropna()


def parse_pe_upload(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """User/stored CSV with columns Date, Index, PE[, PB, DivYield] -> {NSE index name: frame}."""
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    need = ["date", "index", "pe"]
    if not all(n in cols for n in need):
        raise DataError("PE CSV needs columns: Date, Index, PE (PB and DivYield optional).")
    out = {}
    for name, g in df.groupby(df[cols["index"]].astype(str).str.upper().str.strip()):
        recs = g.rename(columns={cols["date"]: "date", cols["pe"]: "pe"})
        if "pb" in cols:
            recs = recs.rename(columns={cols["pb"]: "pb"})
        dyk = next((cols[k] for k in cols if k.startswith("div") or k == "dy"), None)
        if dyk:
            recs = recs.rename(columns={dyk: "divYield"})
        out[name] = parse_pe_records(recs.to_dict("records"))
    return out


# ---------------------------------------------------------------- Screener ---
def _clean_label(text: str) -> str:
    t = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
    return re.sub(r"\s*\+$", "", t).strip()


def parse_shareholding(soup: BeautifulSoup) -> pd.DataFrame | None:
    """Screener's quarterly shareholding table -> DataFrame(index=quarter, columns=[FII, DII, Promoters])."""
    sec = soup.find("section", id="shareholding")
    table = sec.find("table") if sec else None
    if table is None or table.find("thead") is None:
        return None
    hdr = [c.get_text(strip=True) for c in table.find("thead").find_all("th")][1:]
    rows = {}
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if tds:
            rows[_clean_label(tds[0].get_text(" ", strip=True))] = [_to_float(td.get_text(strip=True)) for td in tds[1:]]
    cols = {"FII": rows.get("FIIs"), "DII": rows.get("DIIs"), "Promoters": rows.get("Promoters")}
    cols = {k: v for k, v in cols.items() if v and len(v) == len(hdr)}
    if not hdr or "FII" not in cols:
        return None
    return pd.DataFrame(cols, index=hdr)


def parse_screener_company(html: str) -> dict:
    """Latest-quarter YoY growth for sales and net profit, plus headline ratios."""
    soup = BeautifulSoup(html, "html.parser")
    out = {"name": None, "mcap_cr": np.nan, "pe": np.nan, "roce": np.nan,
           "q_label": None, "sales_yoy": np.nan, "pat_yoy": np.nan, "pat_qoq": np.nan}
    h1 = soup.find("h1")
    if h1:
        out["name"] = h1.get_text(strip=True)
    top = soup.find(id="top-ratios")
    if top:
        for li in top.find_all("li"):
            txt = li.get_text(" ", strip=True)
            num = re.findall(r"[-\d,]+\.?\d*", txt.split("₹")[-1] if "₹" in txt else txt)
            val = _to_float(num[0]) if num else np.nan
            if txt.startswith("Market Cap"):
                out["mcap_cr"] = val
            elif txt.startswith("Stock P/E"):
                out["pe"] = val
            elif txt.startswith("ROCE"):
                out["roce"] = val
    out["own"] = parse_shareholding(soup)
    sec = soup.find("section", id="quarters")
    table = sec.find("table") if sec else None
    if table is None or table.find("thead") is None:
        return out
    hdr = [c.get_text(strip=True) for c in table.find("thead").find_all("th")]
    rows = {}
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        rows[_clean_label(tds[0].get_text(" ", strip=True))] = [_to_float(td.get_text(strip=True)) for td in tds[1:]]
    periods = hdr[1:]
    sales = rows.get("Sales") or rows.get("Revenue")
    pat = rows.get("Net Profit")
    if periods:
        out["q_label"] = periods[-1]

    def yoy(vals):
        if not vals or len(vals) < 5:
            return np.nan
        a, b = vals[-1], vals[-5]
        if np.isnan(a) or np.isnan(b) or b <= 0:
            return np.nan
        return (a / b - 1) * 100

    out["sales_yoy"] = yoy(sales)
    out["pat_yoy"] = yoy(pat)
    if pat and len(pat) >= 2 and pat[-2] and pat[-2] > 0:
        out["pat_qoq"] = (pat[-1] / pat[-2] - 1) * 100
    return out


# ============================================================================
# Network fetchers (all paced through net.py and backed by the disk cache)
# ============================================================================
def fetch_constituents(idx: IndexDef) -> tuple[pd.DataFrame, str]:
    """Return (constituents, source label)."""
    key = f"cons_{idx.key}"
    cached, age = store.load(key)
    if cached is not None and age < TTL_CONS:
        return cached
    errors = []
    try:
        if not site_up(NI_BASE):
            raise DataError("site unreachable")
        r = net.get(f"{NI_BASE}/IndexConstituent/{idx.csv}")
        r.raise_for_status()
        res = (parse_constituents_csv(r.text), "niftyindices")
        store.save(key, res)
        return res
    except Exception as e:  # noqa: BLE001
        errors.append(f"niftyindices: {e}")
    try:
        if not site_up(NSE_BASE):
            raise DataError("site unreachable")
        _nse_warm()
        r = net.get(f"{NSE_BASE}/api/equity-stockIndices", params={"index": idx.nse_name},
                    headers={"Accept": "*/*", "Referer": f"{NSE_BASE}/market-data/live-equity-market"})
        r.raise_for_status()
        res = (parse_nse_index_json(r.json(), idx.nse_name), "nseindia")
        store.save(key, res)
        return res
    except Exception as e:  # noqa: BLE001
        errors.append(f"nseindia: {e}")
    if cached is not None:
        return cached[0], _cached_label(cached[1], age)
    local = DATA_DIR / "constituents" / idx.csv
    if local.exists():
        return parse_constituents_csv(local.read_text()), "saved file"
    raise DataError(f"Constituents for {idx.label} unavailable ({'; '.join(errors)}).")


def _date_chunks(start: date, end: date, days: int = 1000):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _ni_post(endpoint: str, name: str, d0: date, d1: date):
    """POST to niftyindices /BackPage/<endpoint> (layout as of Oct 2026)."""
    cinfo = json.dumps({"name": name, "startDate": d0.strftime("%d-%b-%Y"),
                        "endDate": d1.strftime("%d-%b-%Y"), "indexName": name})
    r = net.post(f"{NI_BASE}/BackPage/{endpoint}", data=json.dumps({"cinfo": cinfo}), headers=NI_HEADERS)
    r.raise_for_status()
    try:
        return parse_ni_response(json.loads(r.text))
    except ValueError as e:
        raise DataError("niftyindices returned a page instead of data (endpoint may have changed)") from e


def _ni_history(key: str, endpoint: str, name: str, years: int, parse, min_rows: int, label: str):
    """Cached niftyindices history: reuse if fresh, otherwise download only the missing days."""
    end = date.today()
    start = end - timedelta(days=int(365.25 * years) + 30)
    cached, age = store.load(key)
    covers = cached is not None and len(cached) > 0 and cached.index.min().date() <= start + timedelta(days=15)
    if covers and age < TTL_HIST:
        return cached, "niftyindices"
    fetch_from = (cached.index.max().date() - timedelta(days=10)) if covers else start
    try:
        if not site_up(NI_BASE):
            raise DataError("niftyindices.com unreachable from this server")
        recs: list[dict] = []
        for d0, d1 in _date_chunks(fetch_from, end):
            recs.extend(_ni_post(endpoint, name, d0, d1))
        merged = _merge(cached if covers else None, parse(recs))
        col = merged["pe"] if isinstance(merged, pd.DataFrame) else merged
        if merged is None or col.notna().sum() < min_rows:
            raise DataError(f"No {label} returned.")
        store.save(key, merged)
        return merged, "niftyindices"
    except Exception as e:  # noqa: BLE001
        if cached is not None and len(cached):
            return cached, _cached_label("niftyindices", age)
        raise DataError(str(e)) from e


def fetch_pe_history(idx: IndexDef, years: int) -> tuple[pd.DataFrame, str]:
    df, src = _ni_history(f"pe_{idx.key}", "getpepbHistoricaldataDBtoString", idx.pe_key, years,
                          parse_pe_records, 20, "PE history")
    return df, src


def load_pe_store() -> dict[str, pd.DataFrame]:
    p = DATA_DIR / "pe_history.csv"
    if not p.exists():
        return {}
    try:
        return parse_pe_upload(pd.read_csv(p))
    except Exception:  # noqa: BLE001
        return {}


def fetch_index_price_ni(idx: IndexDef, years: int) -> tuple[pd.Series, str]:
    return _ni_history(f"nipx_{idx.key}", "getHistoricaldatatabletoString", idx.nse_name, years,
                       parse_price_records, 60, "price history")


# ---------------------------------------------------------------- Yahoo -----
YF_CHUNK = 50


def _yf_batch(tickers: list[str], **when) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, interval="1d", auto_adjust=True, progress=False, threads=4,
                      group_by="column", **when)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.loc[:, ~close.columns.duplicated()]


def _yf_download(tickers: list[str], **when) -> pd.DataFrame:
    """Chunked, paced Yahoo download. A chunk that comes back mostly empty is treated as
    throttling: wait and retry it once. Tickers still missing get one last retry at the end."""
    frames = []
    for i in range(0, len(tickers), YF_CHUNK):
        part = tickers[i:i + YF_CHUNK]
        got = pd.DataFrame()
        for attempt in range(2):
            try:
                got = _yf_batch(part, **when)
            except Exception:  # noqa: BLE001  (YFRateLimitError and network errors)
                got = pd.DataFrame()
            filled = got.notna().any().sum() if not got.empty else 0
            if filled >= 0.5 * len(part):
                break
            time.sleep(15 * (attempt + 1))
        if not got.empty:
            frames.append(got)
        if i + YF_CHUNK < len(tickers):
            time.sleep(1.0 + random.uniform(0, 1.0))
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not out.empty:
        out = out.loc[:, ~out.columns.duplicated()]
    missing = [t for t in tickers if t not in out.columns or out[t].isna().all()]
    if missing and len(missing) < len(tickers):
        time.sleep(3)
        try:
            out = _merge(out, _yf_batch(missing, **when))
        except Exception:  # noqa: BLE001
            pass
    return out.sort_index() if not out.empty else out


def _yf_cached(key: str, tickers: list[str], years: int) -> tuple[pd.DataFrame, str]:
    """Disk-cached Yahoo closes. Fresh cache is reused; stale cache is topped up with only the
    missing days; a full re-download happens weekly or for tickers never seen before."""
    cached, age = store.load(key)
    px = cached["px"] if cached else pd.DataFrame()
    full_at = cached["full_at"] if cached else 0.0
    new_t = [t for t in tickers if t not in px.columns]
    have = [t for t in tickers if t in px.columns]
    need_from = pd.Timestamp.today() - pd.DateOffset(years=years) + pd.Timedelta(days=30)
    too_short = px.empty or px.index.min() > need_from
    if cached and age < TTL_PX and not new_t and not too_short:
        return px[have], "yfinance"
    try:
        if not cached or too_short or time.time() - full_at > FULL_PX_REFRESH:
            merged = _merge(None, _yf_download(sorted(set(tickers) | set(px.columns)), period=f"{years}y"))
            full_at = time.time()
        else:
            merged = px
            if age >= TTL_PX and have:
                since = (px.index.max() - pd.Timedelta(days=7)).date()
                merged = _merge(merged, _yf_download(have, start=since.isoformat()))
            if new_t:
                merged = _merge(merged, _yf_download(new_t, period=f"{years}y"))
        if merged is None or merged.empty:
            raise DataError("Yahoo Finance returned no data.")
        store.save(key, {"px": merged, "full_at": full_at})
        label = "yfinance"
    except Exception as e:  # noqa: BLE001
        if px.empty:
            raise DataError(f"Yahoo Finance unavailable ({e}).") from e
        merged, label = px, _cached_label("yfinance", age)
    cols = [t for t in tickers if t in merged.columns]
    return merged[cols], label


def fetch_index_prices_yf(indices: list[IndexDef], years: int = 10) -> tuple[pd.DataFrame, str]:
    """Close prices keyed by IndexDef.key; columns with < 250 rows are dropped."""
    tick = {i.yf: i.key for i in indices if i.yf}
    close, label = _yf_cached("yf_index", list(tick), years)
    if close.empty:
        return close, label
    close = close.rename(columns=tick)
    keep = [c for c in close.columns if close[c].notna().sum() >= 250]
    return close[keep], label


def fetch_stock_prices(symbols: list[str], years: int = 2) -> tuple[pd.DataFrame, str]:
    """Close prices for NSE symbols (columns = NSE symbols without suffix)."""
    tick = {f"{s}.NS": s for s in symbols}
    close, label = _yf_cached("yf_stocks", list(tick), years)
    return (close.rename(columns=tick) if not close.empty else close), label


# ---------------------------------------------------------------- Screener --
def fetch_screener_company(symbol: str) -> dict:
    key = f"scr_{symbol}"
    cached, age = store.load(key)
    own_fresh = store.load(f"own_{symbol}")[1] < TTL_OWN
    if cached is not None and age < TTL_SCREENER and own_fresh:
        return cached
    last = None
    for path in (f"/company/{symbol}/consolidated/", f"/company/{symbol}/"):
        try:
            r = net.get(f"https://www.screener.in{path}", timeout=15)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            d = parse_screener_company(r.text)
            own = d.pop("own", None)
            if own is not None and len(own):
                own = own.assign(mcap_cr=d["mcap_cr"])
                store.save(f"own_{symbol}", own)
            if not np.isnan(d["pat_yoy"]) or not np.isnan(d["sales_yoy"]):
                store.save(key, d)
                return d
            last = d
        except net.Blocked as e:
            return cached if cached is not None else {"error": str(e), "blocked": True}
        except Exception as e:  # noqa: BLE001
            last = last or {"error": str(e)}
    if last and "error" not in last:
        store.save(key, last)
    return last or cached or {"error": "not found"}


def screener_search(query: str) -> list[tuple[str, str]]:
    """[(SYMBOL, Company name)] from Screener's search API."""
    key = f"scrsearch_{query.lower().strip()}"
    cached, age = store.load(key)
    if cached is not None and age < 7 * 86400:
        return cached
    try:
        r = net.get("https://www.screener.in/api/company/search/",
                    params={"q": query, "v": 3, "fts": 1}, timeout=10)
        r.raise_for_status()
        out = []
        for row in r.json():
            m = re.search(r"/company/([^/]+)/", row.get("url", ""))
            if m:
                out.append((m.group(1).upper(), row.get("name", "")))
        store.save(key, out)
        return out
    except Exception:  # noqa: BLE001
        return cached or []


# ---------------------------------------------------------- NSDL FPI flows --
NSDL_BASE = "https://www.fpi.nsdl.co.in/web"
TTL_FPI = 24 * 3600            # reports come out twice a month
TTL_OWN = 21 * 86400           # shareholding changes quarterly


def norm_sector(name: str) -> str:
    """'Oil, Gas & Consumable Fuels' and 'Oil Gas & Consumable Fuels' -> same key."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def parse_fpi_selection(html: str) -> list[tuple[pd.Timestamp, str]]:
    """Report list from FPI_Fortnightly_Selection.aspx -> [(period end, absolute URL)], newest first."""
    out = []
    for val, label in re.findall(r'<option[^>]*value="([^"]+)"[^>]*>([^<]+)<', html):
        if "StaticReports" not in val:
            continue
        d = pd.to_datetime(label.strip().replace("JUNE", "JUN").replace("JULY", "JUL").replace("SEPT", "SEP"),
                           errors="coerce", format="mixed")
        if pd.isna(d):
            continue
        url = val.replace("~/", f"{NSDL_BASE}/") if val.startswith("~/") else val
        out.append((d.normalize(), url))
    return sorted(out, key=lambda x: x[0], reverse=True)


def parse_fpi_report(html: str) -> tuple[pd.Timestamp, pd.DataFrame]:
    """One fortnightly sector report -> (period end, DataFrame[sector, net_cr, auc_cr]) for equity, in INR crore.
    Takes the latest 'Net Investment …' and 'AUC as on …' blocks in the report."""
    t = pd.read_html(io.StringIO(html))[0]
    hdr_row = next((r for r in range(min(8, len(t))) if any(str(x).strip() == "Sectors" for x in t.iloc[r])), None)
    if hdr_row is None or hdr_row < 3:
        raise DataError("Unexpected NSDL report layout.")
    top, unit, grp, sub = (t.iloc[hdr_row - 3], t.iloc[hdr_row - 2], t.iloc[hdr_row - 1], t.iloc[hdr_row])
    sec_col = next(c for c in range(t.shape[1]) if str(sub.iloc[c]).strip() == "Sectors")

    def pick(prefix):
        cols = [c for c in range(t.shape[1])
                if str(top.iloc[c]).startswith(prefix) and "INR" in str(unit.iloc[c]).upper()
                and str(grp.iloc[c]).strip() == "Equity" and str(sub.iloc[c]).strip() == "Equity"]
        return (cols[-1], str(top.iloc[cols[-1]])) if cols else (None, None)

    net_c, _ = pick("Net Investment")
    auc_c, auc_label = pick("AUC as on")
    if net_c is None or auc_c is None:
        raise DataError("Equity columns not found in NSDL report.")
    end = pd.to_datetime(auc_label.replace("AUC as on", "").strip(), errors="coerce", format="mixed")
    body = t.iloc[hdr_row + 1:]
    rows = []
    for _, r in body.iterrows():
        name = str(r.iloc[sec_col]).strip()
        if not name or name.lower() in ("nan", "grand total", "total"):
            continue
        rows.append({"sector": name, "net_cr": _to_float(r.iloc[net_c]), "auc_cr": _to_float(r.iloc[auc_c])})
    if not rows:
        raise DataError("No sector rows in NSDL report.")
    return pd.Timestamp(end).normalize(), pd.DataFrame(rows)


def fetch_fpi_flows(n_reports: int = 12) -> tuple[pd.DataFrame, str]:
    """Long table [date, sector, key, net_cr, auc_cr] for the latest n fortnights (cached, topped up)."""
    key = "fpi_flows"
    cached, age = store.load(key)
    have = set(pd.to_datetime(cached["date"]).dt.normalize()) if cached is not None and len(cached) else set()
    if cached is not None and age < TTL_FPI and len(have) >= n_reports:
        return cached, "NSDL"
    try:
        r = net.get(f"{NSDL_BASE}/Reports/FPI_Fortnightly_Selection.aspx")
        r.raise_for_status()
        wanted = parse_fpi_selection(r.text)[:n_reports]
        frames = [cached] if cached is not None and len(cached) else []
        for d, url in wanted:
            if d in have:
                continue
            rr = net.get(url)
            rr.raise_for_status()
            end, df = parse_fpi_report(rr.text)
            frames.append(df.assign(date=end if not pd.isna(end) else d))
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out["key"] = out["sector"].map(norm_sector)
        out = out.drop_duplicates(["date", "key"], keep="last").sort_values(["date", "sector"])
        keep = sorted(out["date"].unique())[-max(n_reports, 24):]
        out = out[out["date"].isin(keep)].reset_index(drop=True)
        store.save(key, out)
        return out, "NSDL"
    except Exception as e:  # noqa: BLE001
        if cached is not None and len(cached):
            return cached, _cached_label("NSDL", age)
        raise DataError(f"NSDL FPI data unavailable ({e}).") from e


# ------------------------------------------------- FII / DII ownership ----
def cached_ownership(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Ownership tables already in the cache or seed (never downloads)."""
    out = {}
    for s in symbols:
        df, _ = store.load(f"own_{s}")
        if df is not None and len(df):
            out[s] = df
    return out


def stale_ownership(symbols: list[str]) -> list[str]:
    return [s for s in symbols if store.load(f"own_{s}")[1] > TTL_OWN]


def refresh_ownership(symbols: list[str], budget_s: float = 170, progress=None) -> dict:
    """Fetch Screener pages for stocks whose ownership data is missing or stale, within a time budget.
    Stops at the first refusal from Screener."""
    todo = stale_ownership(symbols)
    done, t0 = 0, time.time()
    for i, s in enumerate(todo):
        if time.time() - t0 > budget_s:
            break
        d = fetch_screener_company(s)
        if d.get("blocked"):
            return {"done": done, "left": len(todo) - i, "blocked": d.get("error")}
        done += 1
        if progress:
            progress(i + 1, len(todo), s)
    return {"done": done, "left": max(0, len(todo) - done), "blocked": None}


# ============================================================================
# Symbol picker (same behaviour as the Forward Matrix / Fair Value apps)
# ============================================================================
def resolve_symbol(query: str, universe: pd.DataFrame, online: bool = True) -> tuple[str | None, list[tuple[str, str]]]:
    """Exact symbol -> (symbol, []); otherwise (None, probable [(symbol, name)])."""
    q = query.strip().upper()
    if not q:
        return None, []
    syms = set(universe["Symbol"])
    if q in syms:
        return q, []
    names = dict(zip(universe["Symbol"], universe["Company"].astype(str)))
    cands: list[tuple[str, str]] = []
    ql = q.lower()
    for s, n in names.items():
        if ql in n.lower() or s.startswith(q):
            cands.append((s, n))
    for s in difflib.get_close_matches(q, list(syms), n=5, cutoff=0.6):
        cands.append((s, names[s]))
    by_name = difflib.get_close_matches(ql, [n.lower() for n in names.values()], n=5, cutoff=0.6)
    for s, n in names.items():
        if n.lower() in by_name:
            cands.append((s, n))
    if online and len(cands) < 3:
        cands += screener_search(query)
    seen, uniq = set(), []
    for s, n in cands:
        if s not in seen:
            seen.add(s)
            uniq.append((s, n))
    return None, uniq[:10]
