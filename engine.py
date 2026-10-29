"""Scoring engine for the Sector Rotation Screener (pure functions, no I/O).

Four legs, each scored 0-100 (higher = more attractive to rotate into):
    Valuation   PE Z-score vs the index's own history (cheap = high score)
    Momentum    RRG position vs the benchmark (RS-Ratio and RS-Momentum ranks)
    Breadth     % of constituents above 50/200-day averages, highs vs lows
    Earnings    implied index EPS (= price / PE) YoY growth and acceleration
    Flows       FPI net equity investment by sector (NSDL, fortnightly) as % of holdings,
                plus the quarterly change in FII + DII ownership of the constituents (Screener)

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONSOLIDATED_SWITCH = pd.Timestamp("2021-04-01")  # NSE moved index PE to consolidated earnings
LEGS = ["Valuation", "Momentum", "Breadth", "Earnings", "Flows"]
DEFAULT_WEIGHTS = {"Valuation": 25, "Momentum": 30, "Breadth": 15, "Earnings": 15, "Flows": 15}


@dataclass
class Settings:
    pe_years: int = 5
    consolidated_only: bool = True
    rrg_len: int = 10           # weeks in RS-Ratio average
    rrg_mom: int = 4            # weeks for RS-Momentum change
    tail: int = 6               # weeks of RRG tail
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    z_cheap: float = -0.75
    z_rich: float = 0.75
    fpi_periods: int = 6        # fortnights summed for the flow leg (6 = about 3 months)


# ============================================================================
# Valuation
# ============================================================================
def z_label(z: float) -> str:
    if np.isnan(z):
        return "n/a"
    if z <= -1.5:
        return "Deeply undervalued"
    if z <= -0.75:
        return "Undervalued"
    if z < 0.75:
        return "Fair"
    if z < 1.5:
        return "Overvalued"
    return "Deeply overvalued"


def valuation(pe: pd.DataFrame | None, st: Settings) -> dict:
    out = {"pe": np.nan, "pe_mean": np.nan, "pe_z": np.nan, "pe_z_1m": np.nan,
           "pe_pctile": np.nan, "pb": np.nan, "pb_z": np.nan, "dy": np.nan,
           "val_label": "n/a", "val_score": np.nan, "pe_obs": 0}
    if pe is None or pe.empty or pe["pe"].notna().sum() < 60:
        return out
    end = pe.index.max()
    start = end - pd.DateOffset(years=st.pe_years)
    if st.consolidated_only:
        start = max(start, CONSOLIDATED_SWITCH)
    hist = pe.loc[pe.index >= start]
    s = hist["pe"].dropna()
    if len(s) < 60:
        return out
    mu, sd = s.mean(), s.std(ddof=0)
    cur = s.iloc[-1]
    prev = s.iloc[-22] if len(s) > 22 else np.nan
    z = (cur - mu) / sd if sd > 0 else np.nan
    out.update(pe=cur, pe_mean=mu, pe_z=z, pe_z_1m=(prev - mu) / sd if sd > 0 else np.nan,
               pe_pctile=(s < cur).mean() * 100, pe_obs=len(s), val_label=z_label(z),
               val_score=float(np.clip(50 - 25 * z, 0, 100)) if not np.isnan(z) else np.nan)
    if "pb" in hist and hist["pb"].notna().sum() > 60:
        b = hist["pb"].dropna()
        out["pb"] = b.iloc[-1]
        out["pb_z"] = (b.iloc[-1] - b.mean()) / b.std(ddof=0) if b.std(ddof=0) > 0 else np.nan
    if "dy" in hist and hist["dy"].notna().any():
        out["dy"] = hist["dy"].dropna().iloc[-1]
    return out


# ============================================================================
# Earnings (implied EPS = index close / PE)
# ============================================================================
def implied_eps(close: pd.Series, pe: pd.DataFrame) -> pd.Series:
    p = pe["pe"].reindex(close.index, method="ffill", tolerance=pd.Timedelta(days=5))
    eps = (close / p).replace([np.inf, -np.inf], np.nan).dropna()
    # rolling median removes one-day jumps from constituent changes
    return eps.rolling(10, min_periods=3).median().dropna()


def earnings(close: pd.Series | None, pe: pd.DataFrame | None) -> dict:
    out = {"eps_yoy": np.nan, "eps_yoy_3m_ago": np.nan, "eps_accel": np.nan}
    if close is None or pe is None or close.empty or pe.empty:
        return out
    eps = implied_eps(close, pe)
    if len(eps) < 320:
        return out
    yoy = (eps / eps.shift(252) - 1) * 100
    yoy = yoy.dropna()
    if len(yoy) < 70:
        return out
    out["eps_yoy"] = yoy.iloc[-1]
    out["eps_yoy_3m_ago"] = yoy.iloc[-64]
    out["eps_accel"] = yoy.iloc[-1] - yoy.iloc[-64]
    return out


# ============================================================================
# Momentum / RRG
# ============================================================================
def weekly(s: pd.Series) -> pd.Series:
    return s.dropna().resample("W-FRI").last().dropna()


def rrg_series(close: pd.Series, bench: pd.Series, n: int = 10, m: int = 4) -> pd.DataFrame:
    """Weekly RS-Ratio and RS-Momentum, both centred on 100."""
    a, b = weekly(close).align(weekly(bench), join="inner")
    rs = 100 * a / b
    ratio = 100 * rs.ewm(span=3, adjust=False).mean() / rs.rolling(n).mean()
    smooth = ratio.ewm(span=3, adjust=False).mean()
    mom = 100 * smooth / smooth.shift(m)
    return pd.DataFrame({"rs_ratio": ratio, "rs_mom": mom}).dropna()


def quadrant(ratio: float, mom: float) -> str:
    if np.isnan(ratio) or np.isnan(mom):
        return "n/a"
    if ratio >= 100:
        return "Leading" if mom >= 100 else "Weakening"
    return "Improving" if mom >= 100 else "Lagging"


def price_stats(close: pd.Series, bench: pd.Series) -> dict:
    c = close.dropna()
    out = {"ret_1m": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "rel_3m": np.nan, "from_52wh": np.nan}
    if len(c) < 130:
        return out

    def r(s, d):
        return (s.iloc[-1] / s.iloc[-d - 1] - 1) * 100 if len(s) > d else np.nan

    out["ret_1m"], out["ret_3m"], out["ret_6m"] = r(c, 21), r(c, 63), r(c, 126)
    bb = bench.dropna()
    out["rel_3m"] = out["ret_3m"] - r(bb, 63)
    out["from_52wh"] = (c.iloc[-1] / c.iloc[-252:].max() - 1) * 100
    return out


# ============================================================================
# Breadth
# ============================================================================
def breadth(prices: pd.DataFrame, symbols: list[str]) -> dict:
    out = {"pct_50": np.nan, "pct_200": np.nan, "pct_50_1m_ago": np.nan,
           "hi_lo": np.nan, "coverage": 0.0, "n_stocks": len(symbols)}
    cols = [s for s in symbols if s in prices.columns]
    if not symbols or not cols:
        return out
    px = prices[cols].ffill(limit=3)
    px = px.loc[:, px.notna().sum() >= 210]
    out["coverage"] = px.shape[1] / len(symbols)
    if px.shape[1] < 3:
        return out
    last = px.iloc[-1]
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    out["pct_50"] = (last > ma50.iloc[-1]).mean() * 100
    out["pct_200"] = (last > ma200.iloc[-1]).mean() * 100
    out["pct_50_1m_ago"] = (px.iloc[-22] > ma50.iloc[-22]).mean() * 100
    w = px.iloc[-252:]
    near_hi = (last >= 0.98 * w.max()).mean() * 100
    near_lo = (last <= 1.02 * w.min()).mean() * 100
    out["hi_lo"] = near_hi - near_lo
    return out


def breadth_score(b: dict) -> float:
    if np.isnan(b["pct_50"]):
        return np.nan
    thrust = 0 if np.isnan(b["pct_50_1m_ago"]) else np.clip(b["pct_50"] - b["pct_50_1m_ago"], -30, 30)
    s = 0.35 * b["pct_50"] + 0.35 * b["pct_200"] + 0.15 * (50 + b["hi_lo"] / 2) + 0.15 * (50 + thrust * 5 / 3)
    return float(np.clip(s, 0, 100))


def equal_weight_index(prices: pd.DataFrame, symbols: list[str]) -> pd.Series | None:
    cols = [s for s in symbols if s in prices.columns]
    if len(cols) < 3:
        return None
    rets = prices[cols].pct_change(fill_method=None).clip(-0.25, 0.25)
    avg = rets.mean(axis=1, skipna=True).fillna(0)
    return 100 * (1 + avg).cumprod()


# ============================================================================
# Flows: FPI (NSDL fortnightly) and FII / DII ownership (Screener, quarterly)
# ============================================================================
def norm_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def fpi_sector_stats(flows: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Per NSDL sector: net equity flow over the last `periods` fortnights, as % of the latest holdings (AUC),
    and acceleration = latest 2 fortnights vs the average fortnight before them (both % of AUC)."""
    if flows is None or flows.empty:
        return pd.DataFrame()
    dates = sorted(flows["date"].unique())[-periods:]
    f = flows[flows["date"].isin(dates)]
    piv = f.pivot_table(index="key", columns="date", values="net_cr", aggfunc="sum").reindex(columns=dates).fillna(0)
    auc = flows[flows["date"] == dates[-1]].set_index("key")["auc_cr"]
    names = flows.drop_duplicates("key", keep="last").set_index("key")["sector"]
    out = pd.DataFrame({"sector": names.reindex(piv.index), "net_cr": piv.sum(axis=1), "auc_cr": auc.reindex(piv.index)})
    out = out[out["auc_cr"] > 0]
    out["flow_pct"] = out["net_cr"] / out["auc_cr"] * 100
    recent = piv.iloc[:, -2:].mean(axis=1)
    before = piv.iloc[:, :-2].mean(axis=1) if piv.shape[1] > 2 else recent * 0
    out["accel_pct"] = ((recent - before) / out["auc_cr"] * 100).reindex(out.index)
    out["last_net_cr"] = piv.iloc[:, -1].reindex(out.index)
    return out


def index_fpi(cons: pd.DataFrame | None, stats: pd.DataFrame) -> dict:
    """Map an index to NSDL sectors through its constituents' NSE industry (same taxonomy as NSDL),
    weighting each sector by how many constituents fall in it."""
    out = {"fpi_flow_pct": np.nan, "fpi_accel_pct": np.nan, "fpi_net_cr": np.nan, "fpi_cover": 0.0}
    if cons is None or cons.empty or stats is None or stats.empty or "Industry" not in cons:
        return out
    keys = cons["Industry"].map(norm_key)
    w = keys[keys.isin(stats.index)].value_counts()
    if w.empty:
        return out
    s = stats.loc[w.index]
    out["fpi_flow_pct"] = float(np.average(s["flow_pct"], weights=w))
    out["fpi_accel_pct"] = float(np.average(s["accel_pct"].fillna(0), weights=w))
    out["fpi_net_cr"] = float(s["net_cr"].sum()) if len(w) == 1 else float(np.average(s["net_cr"], weights=w))
    out["fpi_cover"] = w.sum() / len(cons)
    out["fpi_sectors"] = ", ".join(s["sector"].astype(str).head(3))
    return out


def index_ownership(cons: pd.DataFrame | None, own: dict[str, pd.DataFrame]) -> dict:
    """Market-cap-weighted FII and DII holding % of the constituents, latest quarter and the change
    over 1 and 4 quarters (percentage points)."""
    out = {k: np.nan for k in ["fii", "dii", "fii_d1q", "dii_d1q", "fii_d4q", "dii_d4q"]}
    out["own_cover"] = 0.0
    if cons is None or cons.empty or not own:
        return out
    rows = []
    for sym in cons["Symbol"]:
        df = own.get(sym)
        if df is None or len(df) < 2 or "FII" not in df:
            continue
        fii, dii = df["FII"].astype(float), df["DII"].astype(float) if "DII" in df else pd.Series(np.nan, df.index)
        cap = float(df["mcap_cr"].iloc[-1]) if "mcap_cr" in df and pd.notna(df["mcap_cr"].iloc[-1]) else 1.0

        def d(x, k):
            return x.iloc[-1] - x.iloc[-1 - k] if len(x) > k else np.nan

        rows.append({"w": cap, "fii": fii.iloc[-1], "dii": dii.iloc[-1], "fii_d1q": d(fii, 1), "dii_d1q": d(dii, 1),
                     "fii_d4q": d(fii, 4), "dii_d4q": d(dii, 4)})
    if not rows:
        return out
    r = pd.DataFrame(rows)
    for k in ["fii", "dii", "fii_d1q", "dii_d1q", "fii_d4q", "dii_d4q"]:
        m = r[k].notna()
        if m.any():
            out[k] = float(np.average(r.loc[m, k], weights=r.loc[m, "w"]))
    out["own_cover"] = len(r) / len(cons)
    return out


def flows_score(df: pd.DataFrame) -> pd.Series:
    """0-100: FPI flow intensity (half absolute, half rank) blended 70/30 with the 1-quarter change in
    FII + DII ownership when at least half the constituents have ownership data."""
    inten = df["fpi_flow_pct"] + 0.5 * df["fpi_accel_pct"].fillna(0)
    fpi = (0.5 * (50 + 50 * np.tanh(df["fpi_flow_pct"] / 1.5)) + 0.5 * pct_rank(inten)).where(df["fpi_flow_pct"].notna())
    inst = df["fii_d1q"].fillna(0) + df["dii_d1q"].fillna(0)
    ok = (df["own_cover"] >= 0.5) & df["fii_d1q"].notna()
    own = (0.5 * (50 + 50 * np.tanh(inst / 1.0)) + 0.5 * pct_rank(inst.where(ok))).where(ok)
    both = 0.7 * fpi + 0.3 * own
    return both.where(fpi.notna() & own.notna(), fpi.where(fpi.notna(), own))


# ============================================================================
# Composite
# ============================================================================
def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100


ROTATION_MATRIX = {
    ("Improving", "cheap"): "🟢 Rotate in — early",
    ("Improving", "fair"): "🟢 Rotate in — early",
    ("Improving", "rich"): "🟡 Tactical only",
    ("Leading", "cheap"): "🟢 Overweight — confirmed",
    ("Leading", "fair"): "🟢 Overweight",
    ("Leading", "rich"): "🟡 Ride with trailing stop",
    ("Weakening", "cheap"): "🟡 Pause — watch",
    ("Weakening", "fair"): "🟠 Book partial",
    ("Weakening", "rich"): "🔴 Rotate out",
    ("Lagging", "cheap"): "🟡 Value watch — wait for turn",
    ("Lagging", "fair"): "🔴 Avoid",
    ("Lagging", "rich"): "🔴 Avoid",
}


def rotation_signal(quad: str, z: float, st: Settings) -> str:
    if quad == "n/a":
        return "⚪ Insufficient data"
    if np.isnan(z):
        return {"Leading": "🟢 Momentum leader (no PE)", "Improving": "🟢 Turning up (no PE)",
                "Weakening": "🟠 Fading (no PE)", "Lagging": "🔴 Lagging (no PE)"}[quad]
    band = "cheap" if z <= st.z_cheap else "rich" if z >= st.z_rich else "fair"
    return ROTATION_MATRIX[(quad, band)]


def tier(score: float) -> str:
    if np.isnan(score):
        return "n/a"
    if score >= 65:
        return "Overweight"
    if score >= 52:
        return "Accumulate"
    if score >= 40:
        return "Neutral"
    return "Underweight"


def reconcile_tier(t: str, signal: str) -> str:
    """Keep the score tier from contradicting the RRG x valuation signal."""
    if signal.startswith("🔴") and t in ("Overweight", "Accumulate"):
        return "Neutral"
    if signal[:1] in ("🟠", "🟡") and t == "Overweight":
        return "Accumulate"
    if signal.startswith("🟢") and t == "Underweight":
        return "Neutral"
    return t


def confidence(row: pd.Series) -> str:
    legs = [row[f"{l}_score"] for l in LEGS if not np.isnan(row[f"{l}_score"])]
    if len(legs) < 2:
        return "Low"
    spread = float(np.std(legs))
    cov = row.get("coverage", 1.0)
    if len(legs) >= 4 and spread < 18 and cov >= 0.8:
        return "High"
    if len(legs) >= 3 and spread < 25:
        return "Medium"
    return "Low"


def score_table(rows: pd.DataFrame, st: Settings) -> pd.DataFrame:
    """rows: one row per index with raw metrics -> adds leg scores, composite, signals."""
    df = rows.copy()
    df["Valuation_score"] = df["val_score"]
    # half absolute (where it sits vs the 100/100 cross-hair), half rank vs other indices,
    # so a lagging sector cannot score high just because everything else lags more
    absolute = 50 + 50 * (0.6 * np.tanh((df["rs_ratio"] - 100) / 3) + 0.4 * np.tanh((df["rs_mom"] - 100) / 3))
    ranked = 0.6 * pct_rank(df["rs_ratio"]) + 0.4 * pct_rank(df["rs_mom"])
    df["Momentum_score"] = (0.5 * absolute + 0.5 * ranked).where(df["rs_ratio"].notna())
    df["Breadth_score"] = df["breadth_score"]
    earn_raw = df["eps_yoy"].clip(-50, 80) + 0.5 * df["eps_accel"].clip(-50, 50)
    df["Earnings_score"] = pct_rank(earn_raw).where(earn_raw.notna())
    for c, v in {"fpi_flow_pct": np.nan, "fpi_accel_pct": np.nan, "fii_d1q": np.nan, "dii_d1q": np.nan,
                 "own_cover": 0.0}.items():
        if c not in df:
            df[c] = v
    df["Flows_score"] = flows_score(df)
    w = pd.Series({l: max(0.0, float(st.weights.get(l, 0))) for l in LEGS})

    def comp(r):
        vals = pd.Series({l: r[f"{l}_score"] for l in LEGS}).astype(float)
        mask = vals.notna() & (w > 0)
        if mask.sum() == 0 or w[mask].sum() == 0:
            return np.nan
        return float((vals[mask] * w[mask]).sum() / w[mask].sum())

    df["Composite"] = df.apply(comp, axis=1)
    df["Confidence"] = df.apply(confidence, axis=1)
    df["Signal"] = [rotation_signal(q, z, st) for q, z in zip(df["quadrant"], df["pe_z"])]
    df["Tier"] = [reconcile_tier(tier(c), s) for c, s in zip(df["Composite"], df["Signal"])]
    return df.sort_values("Composite", ascending=False, na_position="last")


# ============================================================================
# Stock drill-down
# ============================================================================
def stock_table(prices: pd.DataFrame, cons: pd.DataFrame, sector_close: pd.Series) -> pd.DataFrame:
    rows = []
    sc = sector_close.dropna()
    for _, c in cons.iterrows():
        sym = c["Symbol"]
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        if len(s) < 210:
            continue
        ps = price_stats(s, sc)
        ma50, ma200 = s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
        r1_rel = ps["ret_1m"] - ((sc.iloc[-1] / sc.iloc[-22] - 1) * 100 if len(sc) > 22 else np.nan)
        rows.append({"Symbol": sym, "Company": c["Company"], "Industry": c.get("Industry", ""),
                     "Price": s.iloc[-1], "1M %": ps["ret_1m"], "3M %": ps["ret_3m"],
                     "Rel 3M vs sector": ps["rel_3m"], "Rel 1M vs sector": r1_rel,
                     "> 50 DMA": bool(s.iloc[-1] > ma50), "> 200 DMA": bool(s.iloc[-1] > ma200),
                     "From 52W high %": ps["from_52wh"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["RS score"] = (0.45 * pct_rank(df["Rel 3M vs sector"]) + 0.25 * pct_rank(df["Rel 1M vs sector"])
                      + 15 * df["> 50 DMA"] + 15 * df["> 200 DMA"]).round(1)
    return df.sort_values("RS score", ascending=False).reset_index(drop=True)
