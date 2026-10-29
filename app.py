"""Sector Rotation Screener — NSE sectoral, thematic and size indices.

Four-leg rotation model: Valuation (PE Z-score), Momentum (RRG vs benchmark),
Breadth (constituents above 50/200 DMA) and Earnings (implied index EPS growth),
with a drill-down into each sector's strongest stocks.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import hmac
import io
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds
import engine as en
from sectors import BENCHMARKS, GROUPS, INDEX_BY_KEY, INDICES

st.set_page_config(page_title="Sector Rotation Screener · Dr Shantanu Samanta",
                   page_icon="🧭", layout="wide", initial_sidebar_state="auto")

DEMO = os.getenv("SRS_DEMO") == "1"
AUTHOR = "Dr Shantanu Samanta"
AUTHOR_TITLE = "Physician · Quantitative Market Analyst & Trading-Systems Developer · Kolkata, India"
QUAD_COLORS = {"Leading": "#16a34a", "Weakening": "#d97706", "Lagging": "#dc2626", "Improving": "#2563eb", "n/a": "#9ca3af"}

# ============================================================================
# Styling
# ============================================================================
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#0e7490 100%); color:#fff;
       border-radius:16px; padding:18px 22px; margin-bottom:14px;}
.hero h1 {font-size: clamp(1.35rem, 2.6vw, 2rem); margin:0 0 4px 0; color:#fff;}
.hero p {margin:0; opacity:.9; font-size: clamp(.82rem,1.4vw,.95rem);}
.hero .by {margin-top:8px; font-size:.82rem; opacity:.85}
.kpis {display:grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:6px 0 12px;}
.kpi {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:10px 14px;}
.kpi .t {font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; opacity:.7}
.kpi .v {font-size:1.05rem; font-weight:650; margin-top:2px; line-height:1.35}
.small {font-size:.8rem; opacity:.75}
@media (max-width: 640px){ .block-container{padding-left:.6rem; padding-right:.6rem} }
</style>""", unsafe_allow_html=True)


# ============================================================================
# Cached data loaders
# ============================================================================
TTL = 3 * 3600
REFRESH_COOLDOWN = 10 * 60   # seconds between manual refreshes per visitor


# The in-memory cache (st.cache_data) sits on top of the disk cache in store.py:
# the disk cache decides when anything is actually downloaded again.
# Few workers on purpose — net.py paces each site anyway, so more threads only queue.


@st.cache_data(ttl=TTL, show_spinner=False)
def load_constituents(keys: tuple[str, ...]):
    if DEMO:
        from demo_data import demo_bundle
        return demo_bundle(list(keys), "NIFTY50")[0], {}
    out, errs = {}, {}

    def one(k):
        try:
            return k, ds.fetch_constituents(INDEX_BY_KEY[k]), None
        except Exception as e:  # noqa: BLE001
            return k, None, str(e)

    with ThreadPoolExecutor(max_workers=2) as ex:
        for k, res, err in ex.map(one, keys):
            if res is not None:
                out[k] = res
            else:
                errs[k] = err
    return out, errs


@st.cache_data(ttl=TTL, show_spinner=False)
def load_index_prices(keys: tuple[str, ...], years: int):
    if DEMO:
        from demo_data import demo_bundle
        return demo_bundle(list(keys), "NIFTY50")[1], {k: "demo" for k in keys}
    defs = [INDEX_BY_KEY[k] for k in keys]
    yrs = max(years, 3) + 1
    try:
        px, label = ds.fetch_index_prices_yf(defs, yrs)
    except Exception:  # noqa: BLE001
        px, label = pd.DataFrame(), ""
    src = {k: label for k in px.columns}
    missing = [d for d in defs if d.key not in px.columns]

    def one(d):
        try:
            return d.key, *ds.fetch_index_price_ni(d, yrs)
        except Exception:  # noqa: BLE001
            return d.key, None, None

    with ThreadPoolExecutor(max_workers=2) as ex:
        for k, s, lab in ex.map(one, missing):
            if s is not None:
                px = px.join(s.rename(k), how="outer") if not px.empty else s.rename(k).to_frame()
                src[k] = lab
    return px.sort_index(), src


@st.cache_data(ttl=TTL, show_spinner=False)
def load_pe(keys: tuple[str, ...], years: int):
    if DEMO:
        from demo_data import demo_bundle
        return demo_bundle(list(keys), "NIFTY50")[2], {k: "demo" for k in keys}, {}
    saved = ds.load_pe_store()
    out, src, errs = {}, {}, {}

    def one(k):
        try:
            return k, *ds.fetch_pe_history(INDEX_BY_KEY[k], years), None
        except Exception as e:  # noqa: BLE001
            return k, None, None, str(e)

    with ThreadPoolExecutor(max_workers=2) as ex:
        for k, df, lab, err in ex.map(one, keys):
            d = INDEX_BY_KEY[k]
            if df is not None:
                out[k], src[k] = df, lab
            elif d.nse_name in saved or d.pe_key in saved:
                out[k], src[k] = saved.get(d.nse_name, saved.get(d.pe_key)), "saved file"
            else:
                errs[k] = err
    return out, src, errs


@st.cache_data(ttl=TTL, show_spinner=False)
def load_stock_prices(symbols: tuple[str, ...]):
    if DEMO:
        from demo_data import demo_bundle
        return demo_bundle([i.key for i in INDICES], "NIFTY50")[3], "demo"
    try:
        return ds.fetch_stock_prices(list(symbols), years=2)
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), f"failed: {e}"


@st.cache_data(ttl=TTL, show_spinner=False)
def load_fpi():
    if DEMO:
        from demo_data import demo_flows
        return demo_flows(), "demo", None
    try:
        df, src = ds.fetch_fpi_flows(12)
        return df, src, None
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), "—", str(e)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_ownership(symbols: tuple[str, ...]):
    if DEMO:
        from demo_data import demo_ownership
        return demo_ownership(list(symbols))
    return ds.cached_ownership(list(symbols))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_screener(symbol: str) -> dict:
    return ds.fetch_screener_company(symbol)


# ============================================================================
# Sidebar
# ============================================================================
# ---------------------------------------------------------------- owner ----
SETTINGS_FILE = Path(__file__).parent / "data" / "owner_settings.json"
BUILTIN_DEFAULTS = dict(pe_years=5, consolidated_only=True, z_cheap=-0.75, z_rich=0.75,
                        rrg_len=10, rrg_mom=4, tail=6, weights=dict(en.DEFAULT_WEIGHTS), do_breadth=True,
                        fpi_periods=6)
MAX_TRIES, LOCKOUT_S = 3, 120


def owner_password() -> str | None:
    """From Streamlit secrets (OWNER_PASSWORD) or the environment. None = owner section disabled."""
    try:
        pw = st.secrets.get("OWNER_PASSWORD")
    except Exception:  # noqa: BLE001  (no secrets file)
        pw = None
    return pw or os.getenv("OWNER_PASSWORD") or None


def load_defaults() -> dict:
    """Settings every visitor gets: data/owner_settings.json if present, else the built-ins."""
    d = json.loads(json.dumps(BUILTIN_DEFAULTS))
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        d.update({k: v for k, v in saved.items() if k in d and k != "weights"})
        d["weights"].update({k: v for k, v in saved.get("weights", {}).items() if k in d["weights"]})
    except Exception:  # noqa: BLE001  (missing or malformed file -> built-ins)
        pass
    return d


def owner_login(sb) -> bool:
    ss = st.session_state
    if ss.get("owner"):
        if sb.button("🔓 Log out of owner settings", width="stretch"):
            ss["owner"] = False
            st.rerun()
        return True
    pw = owner_password()
    with sb.expander("🔒 Owner", expanded=False):
        if not pw:
            st.caption("Owner settings are off. Set OWNER_PASSWORD in the app's Secrets to enable them.")
            return False
        locked = ss.get("owner_lock_until", 0) - time.time()
        if locked > 0:
            st.caption(f"Too many attempts — try again in {locked:.0f} s.")
            return False
        with st.form("owner_login", clear_on_submit=True, border=False):
            entered = st.text_input("Password", type="password")
            ok = st.form_submit_button("Unlock", width="stretch")
        if ok:
            if hmac.compare_digest(entered.encode(), pw.encode()):
                ss["owner"], ss["owner_fails"] = True, 0
                st.rerun()
            ss["owner_fails"] = ss.get("owner_fails", 0) + 1
            if ss["owner_fails"] >= MAX_TRIES:
                ss["owner_lock_until"], ss["owner_fails"] = time.time() + LOCKOUT_S, 0
            st.error("Wrong password.")
    return False


def owner_panel(sb, d: dict) -> tuple[dict, object]:
    """All tuning controls, the PE upload and the data refresh — owner only."""
    sb.markdown("#### 🔓 Owner settings")
    with sb.expander("Valuation", expanded=False):
        d["pe_years"] = st.select_slider("PE history lookback (years)", [1, 2, 3, 5, 7, 10], value=d["pe_years"])
        d["consolidated_only"] = st.checkbox(
            "Only use PE from Apr 2021 (consolidated basis)", value=d["consolidated_only"],
            help="NSE switched index PE from standalone to consolidated earnings in April 2021; "
                 "older PE is not comparable.")
        d["z_cheap"] = st.slider("Cheap if Z ≤", -2.0, 0.0, float(d["z_cheap"]), 0.05)
        d["z_rich"] = st.slider("Rich if Z ≥", 0.0, 2.0, float(d["z_rich"]), 0.05)
        upload = st.file_uploader("Optional PE history CSV (Date, Index, PE, PB, DivYield)", type="csv",
                                  help="Fills in any index whose PE history could not be fetched live.")
    with sb.expander("Momentum (RRG)", expanded=False):
        d["rrg_len"] = st.slider("RS-Ratio length (weeks)", 6, 26, int(d["rrg_len"]))
        d["rrg_mom"] = st.slider("RS-Momentum lookback (weeks)", 1, 8, int(d["rrg_mom"]))
        d["tail"] = st.slider("Tail length (weeks)", 3, 15, int(d["tail"]))
    with sb.expander("Flows (FPI)", expanded=False):
        d["fpi_periods"] = st.slider("FPI window (fortnights)", 2, 12, int(d["fpi_periods"]),
                                     help="Fortnights of NSDL net investment summed for the Flows leg (6 ≈ 3 months).")
    with sb.expander("Leg weights", expanded=False):
        d["weights"] = {l: st.slider(l, 0, 50, int(d["weights"][l]), 5) for l in en.LEGS}
    d["do_breadth"] = sb.toggle("Compute breadth & drill-down (downloads constituent prices)",
                                value=bool(d["do_breadth"]))
    with sb.expander("Defaults for all visitors", expanded=False):
        st.caption("Visitors see the saved defaults; your changes above apply only to your session until saved.")
        if st.button("Save these as the default", width="stretch"):
            try:
                SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
                SETTINGS_FILE.write_text(json.dumps(d, indent=1))
                st.success("Saved on the server (kept until the app restarts).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not save: {e}")
        st.download_button("⬇️ owner_settings.json", json.dumps(d, indent=1), "owner_settings.json",
                           "application/json", width="stretch",
                           help="Upload this to the repo's data/ folder to keep the defaults permanently.")
    last = st.session_state.get("last_refresh", 0.0)
    wait = REFRESH_COOLDOWN - (time.time() - last)
    if sb.button("🔄 Refresh live data", width="stretch", disabled=wait > 0,
                 help="Re-checks the sources for everyone. Only data older than its refresh interval is "
                      "downloaded again (prices 6 h, PE 12 h, constituents 7 days)."):
        st.session_state["last_refresh"] = time.time()
        st.cache_data.clear()
        st.rerun()
    if wait > 0:
        sb.caption(f"Refresh available again in {wait / 60:.0f} min — protects the data sources from repeat downloads.")
    return d, upload


def sidebar() -> tuple[en.Settings, dict]:
    sb = st.sidebar
    sb.markdown(f"### 🧭 Settings\n<span class='small'>by {AUTHOR}</span>", unsafe_allow_html=True)
    bench_key = sb.selectbox("Benchmark", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k].label)
    groups = sb.multiselect("Index groups", GROUPS, default=GROUPS)
    d, upload = load_defaults(), None
    if owner_login(sb):
        d, upload = owner_panel(sb, d)
    cfg = dict(bench_key=bench_key, groups=groups, upload=upload, do_breadth=d["do_breadth"])
    return en.Settings(pe_years=d["pe_years"], consolidated_only=d["consolidated_only"], rrg_len=d["rrg_len"],
                       rrg_mom=d["rrg_mom"], tail=d["tail"], weights=d["weights"], z_cheap=d["z_cheap"],
                       z_rich=d["z_rich"], fpi_periods=int(d["fpi_periods"])), cfg


# ============================================================================
# Pipeline
# ============================================================================
def build(stt: en.Settings, cfg: dict):
    bench_key = cfg["bench_key"]
    sel = [i for i in INDICES if i.group in cfg["groups"] and i.key != bench_key]
    keys = tuple(sorted({i.key for i in sel} | {bench_key}))
    status = {}

    with st.status("Collecting live data…", expanded=False) as box:
        box.write("Index prices (yfinance → niftyindices)…")
        prices, price_src = load_index_prices(keys, stt.pe_years)
        box.write("PE / PB / dividend-yield history (niftyindices)…")
        pe, pe_src, pe_err = load_pe(tuple(i.key for i in sel), stt.pe_years)
        if cfg["upload"] is not None:
            try:
                up = ds.parse_pe_upload(pd.read_csv(cfg["upload"]))
                for i in sel:
                    hit = up.get(i.nse_name) if i.nse_name in up else up.get(i.pe_key)
                    if i.key not in pe and hit is not None:
                        pe[i.key], pe_src[i.key] = hit, "uploaded CSV"
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not read the uploaded PE CSV: {e}")
        stocks, stocks_src = pd.DataFrame(), "—"
        box.write("Index constituents (niftyindices → NSE)…")
        cons, cons_err = load_constituents(keys)
        syms = sorted({s for k, (df, _) in cons.items() for s in df["Symbol"]})
        if cfg["do_breadth"] and syms:
            box.write(f"Prices for {len(syms)} constituent stocks (yfinance)…")
            stocks, stocks_src = load_stock_prices(tuple(syms))
        box.write("FPI sector flows (NSDL) and FII/DII ownership…")
        fpi, fpi_src, fpi_err = load_fpi()
        own = load_ownership(tuple(syms))
        box.update(label="Data ready", state="complete")
    fpi_stats = en.fpi_sector_stats(fpi, stt.fpi_periods)

    bench = prices[bench_key].dropna() if bench_key in prices else None
    if bench is None or bench.empty:
        st.error("Could not load the benchmark price history from Yahoo Finance or niftyindices. "
                 "Try **Refresh live data** in a minute.")
        st.stop()

    rows, rrg_tails, sec_close = [], {}, {}
    for i in sel:
        close = prices[i.key].dropna() if i.key in prices else None
        psrc = price_src.get(i.key)
        if (close is None or len(close) < 260) and i.key in cons and not stocks.empty:
            proxy = en.equal_weight_index(stocks, list(cons[i.key][0]["Symbol"]))
            if proxy is not None and proxy.notna().sum() > 260:
                close, psrc = proxy.dropna(), "equal-weight proxy"
        row = {"key": i.key, "Index": i.label, "Group": i.group, "price_src": psrc or "—",
               "pe_src": pe_src.get(i.key, "—"),
               "cons_src": cons[i.key][1] if i.key in cons else "—"}
        p = pe.get(i.key)
        row |= en.valuation(p, stt)
        icons = cons[i.key][0] if i.key in cons else None
        row |= en.index_fpi(icons, fpi_stats)
        row |= en.index_ownership(icons, own)
        # implied EPS only makes sense with the real index level
        row |= en.earnings(close if psrc and not psrc.startswith("equal") else None, p)
        if close is not None and len(close) > 260:
            rr = en.rrg_series(close, bench, stt.rrg_len, stt.rrg_mom)
            last = rr.iloc[-1] if not rr.empty else pd.Series({"rs_ratio": np.nan, "rs_mom": np.nan})
            row |= {"rs_ratio": last["rs_ratio"], "rs_mom": last["rs_mom"],
                    "quadrant": en.quadrant(last["rs_ratio"], last["rs_mom"])}
            row |= en.price_stats(close, bench)
            rrg_tails[i.label] = rr.tail(stt.tail)
        else:
            row |= {"rs_ratio": np.nan, "rs_mom": np.nan, "quadrant": "n/a"} | \
                   {k: np.nan for k in ["ret_1m", "ret_3m", "ret_6m", "rel_3m", "from_52wh"]}
        b = en.breadth(stocks, list(cons[i.key][0]["Symbol"])) if i.key in cons and not stocks.empty \
            else {"pct_50": np.nan, "pct_200": np.nan, "pct_50_1m_ago": np.nan, "hi_lo": np.nan,
                  "coverage": 0.0, "n_stocks": 0}
        row |= b
        row["breadth_score"] = en.breadth_score(b)
        rows.append(row)
        if close is not None:
            sec_close[i.key] = close

    table = en.score_table(pd.DataFrame(rows), stt)
    status.update(pe_err=pe_err, cons_err=cons_err, stocks_src=stocks_src, fpi_src=fpi_src, fpi_err=fpi_err)
    return dict(table=table, tails=rrg_tails, prices=prices, sec_close=sec_close, fpi=fpi, fpi_stats=fpi_stats,
                own=own, syms=syms, pe=pe, cons=cons, stocks=stocks,
                bench=bench, status=status, sel=sel)


# ============================================================================
# UI pieces
# ============================================================================
def hero():
    st.markdown(f"""
<div class="hero">
  <h1>🧭 Sector Rotation Screener</h1>
  <p>NSE sectoral, thematic and size indices scored on Valuation · Momentum · Breadth · Earnings · Flows —
     where money may be rotating next.</p>
  <div class="by">Developed by <b>{AUTHOR}</b> — {AUTHOR_TITLE}</div>
</div>""", unsafe_allow_html=True)


def fmt_list(df, n=3):
    return ", ".join(df["Index"].head(n)) if not df.empty else "—"


def kpis(t: pd.DataFrame):
    rot_in = t[t["Signal"].str.startswith("🟢")]
    rot_out = t[t["Signal"].str.startswith("🔴")].sort_values("Composite")
    cheap = t[t["pe_z"] <= -0.75].sort_values("pe_z")
    improving = t[t["quadrant"] == "Improving"].sort_values("rs_mom", ascending=False)
    st.markdown(f"""
<div class="kpis">
 <div class="kpi"><div class="t">Top composite</div><div class="v">{fmt_list(t)}</div></div>
 <div class="kpi"><div class="t">🟢 Rotate in / overweight</div><div class="v">{fmt_list(rot_in)}</div></div>
 <div class="kpi"><div class="t">🔵 Improving (RRG)</div><div class="v">{fmt_list(improving)}</div></div>
 <div class="kpi"><div class="t">💰 Undervalued (PE Z ≤ −0.75)</div><div class="v">{fmt_list(cheap)}</div></div>
 <div class="kpi"><div class="t">🔴 Rotate out / avoid</div><div class="v">{fmt_list(rot_out)}</div></div>
</div>""", unsafe_allow_html=True)


def board(t: pd.DataFrame):
    show = pd.DataFrame({
        "Index": t["Index"], "Group": t["Group"], "Signal": t["Signal"], "Composite": t["Composite"],
        "Tier": t["Tier"], "Confidence": t["Confidence"],
        "Valuation": t["Valuation_score"], "Momentum": t["Momentum_score"],
        "Breadth": t["Breadth_score"], "Earnings": t["Earnings_score"], "Flows": t["Flows_score"],
        "PE": t["pe"], "PE Z": t["pe_z"], "Valuation zone": t["val_label"], "RRG": t["quadrant"],
        "3M %": t["ret_3m"], "Rel 3M %": t["rel_3m"], "% > 200 DMA": t["pct_200"],
        "EPS YoY %": t["eps_yoy"], "FPI 3M % AUC": t["fpi_flow_pct"],
        "FII+DII Δ1Q pp": t["fii_d1q"] + t["dii_d1q"],
    })
    for c in show.columns:
        if c not in ("Index", "Group", "Signal", "Tier", "Confidence", "Valuation zone", "RRG"):
            show[c] = pd.to_numeric(show[c], errors="coerce").astype(float)
    score_cols = ["Composite", "Valuation", "Momentum", "Breadth", "Earnings", "Flows"]
    sty = (show.style
           .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=100)
           .background_gradient(subset=["PE Z"], cmap="RdYlGn_r", vmin=-2, vmax=2)
           .format({c: "{:.0f}" for c in score_cols} | {"PE": "{:.1f}", "PE Z": "{:+.2f}", "3M %": "{:+.1f}",
                    "Rel 3M %": "{:+.1f}", "% > 200 DMA": "{:.0f}", "EPS YoY %": "{:+.1f}",
                    "FPI 3M % AUC": "{:+.2f}", "FII+DII Δ1Q pp": "{:+.2f}"}, na_rep="—"))
    st.dataframe(sty, width="stretch", hide_index=True, height=min(60 + 35 * len(show), 820))
    st.download_button("⬇️ Download board (CSV)", show.to_csv(index=False).encode(),
                       f"sector_rotation_{datetime.now():%Y%m%d}.csv", "text/csv")


def rrg_chart(tails: dict, table: pd.DataFrame):
    fig = go.Figure()
    allx = pd.concat([d["rs_ratio"] for d in tails.values()]) if tails else pd.Series([100])
    ally = pd.concat([d["rs_mom"] for d in tails.values()]) if tails else pd.Series([100])
    rx = max(abs(allx - 100).max(), 1) * 1.15
    ry = max(abs(ally - 100).max(), 1) * 1.15
    for (x0, x1, y0, y1, name, col) in [(100, 100 + rx, 100, 100 + ry, "Leading", "#16a34a"),
                                        (100, 100 + rx, 100 - ry, 100, "Weakening", "#d97706"),
                                        (100 - rx, 100, 100 - ry, 100, "Lagging", "#dc2626"),
                                        (100 - rx, 100, 100, 100 + ry, "Improving", "#2563eb")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, opacity=0.07, line_width=0)
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - ry * 0.06, text=f"<b>{name}</b>", showarrow=False,
                           font=dict(color=col, size=13))
    quad = dict(zip(table["Index"], table["quadrant"]))
    for name, d in tails.items():
        col = QUAD_COLORS.get(quad.get(name, "n/a"), "#9ca3af")
        fig.add_trace(go.Scatter(x=d["rs_ratio"], y=d["rs_mom"], mode="lines+markers",
                                 line=dict(color=col, width=1.6), marker=dict(size=4, color=col),
                                 opacity=0.75, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[d["rs_ratio"].iloc[-1]], y=[d["rs_mom"].iloc[-1]], mode="markers+text",
                                 marker=dict(size=11, color=col, line=dict(color="white", width=1)),
                                 text=[name], textposition="top center", name=name,
                                 hovertemplate=f"<b>{name}</b><br>RS-Ratio %{{x:.2f}}<br>RS-Mom %{{y:.2f}}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color="gray", width=1))
    fig.add_vline(x=100, line=dict(color="gray", width=1))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis=dict(title="RS-Ratio (trend of relative strength) →", range=[100 - rx, 100 + rx]),
                      yaxis=dict(title="RS-Momentum →", range=[100 - ry, 100 + ry]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Sectors usually rotate clockwise: Improving → Leading → Weakening → Lagging. "
               "The dot is this week; the tail shows the path over the chosen number of weeks.")


def valuation_tab(d: dict, stt: en.Settings):
    t = d["table"].dropna(subset=["pe_z"]).sort_values("pe_z", ascending=False)  # cheapest drawn at top
    if t.empty:
        st.info("No PE history could be fetched live for any index. Upload a PE CSV in the sidebar "
                "(Date, Index, PE, PB, DivYield) or place it at data/pe_history.csv in the repo.")
        return
    colors = ["#16a34a" if z <= stt.z_cheap else "#dc2626" if z >= stt.z_rich else "#9ca3af" for z in t["pe_z"]]
    fig = go.Figure(go.Bar(x=t["pe_z"], y=t["Index"], orientation="h", marker_color=colors,
                           text=[f"PE {p:.1f} · Z {z:+.2f}" for p, z in zip(t["pe"], t["pe_z"])],
                           textposition="auto"))
    for v in (stt.z_cheap, stt.z_rich):
        fig.add_vline(x=v, line=dict(dash="dot", color="gray"))
    fig.update_layout(height=max(320, 28 * len(t)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="PE Z-score vs own history (negative = cheap)")
    st.plotly_chart(fig, width="stretch")
    pick = st.selectbox("PE history for", list(t["Index"])[::-1])
    key = t.loc[t["Index"] == pick, "key"].iloc[0]
    p = d["pe"][key]["pe"].dropna()
    start = p.index.max() - pd.DateOffset(years=stt.pe_years)
    if stt.consolidated_only:
        start = max(start, en.CONSOLIDATED_SWITCH)
    p = p[p.index >= start]
    mu, sd = p.mean(), p.std(ddof=0)
    f2 = go.Figure(go.Scatter(x=p.index, y=p, name="PE", line=dict(color="#1e3a8a")))
    for k, dash in [(0, "solid"), (1, "dash"), (-1, "dash"), (2, "dot"), (-2, "dot")]:
        f2.add_hline(y=mu + k * sd, line=dict(color="gray", dash=dash, width=1),
                     annotation_text="mean" if k == 0 else f"{k:+d} SD", annotation_position="right")
    f2.update_layout(height=380, margin=dict(l=10, r=40, t=10, b=10), yaxis_title="PE (TTM)")
    st.plotly_chart(f2, width="stretch")


def breadth_earn_tab(d: dict):
    t = d["table"]
    c1, c2 = st.columns(2)
    with c1:
        b = t.dropna(subset=["pct_200"]).sort_values("pct_200")
        if b.empty:
            st.info("Breadth needs constituent lists and prices — turn on breadth in the sidebar.")
        else:
            fig = go.Figure()
            fig.add_bar(y=b["Index"], x=b["pct_200"], orientation="h", name="% > 200 DMA", marker_color="#1e3a8a")
            fig.add_bar(y=b["Index"], x=b["pct_50"], orientation="h", name="% > 50 DMA", marker_color="#0e7490")
            fig.update_layout(barmode="group", height=max(320, 34 * len(b)), margin=dict(l=10, r=10, t=30, b=10),
                              title="Breadth: % of constituents above moving averages",
                              legend=dict(orientation="h", y=-0.08))
            st.plotly_chart(fig, width="stretch")
    with c2:
        e = t.dropna(subset=["eps_yoy"]).sort_values("eps_yoy")
        if e.empty:
            st.info("Earnings momentum needs both PE history and the real index price.")
        else:
            fig = go.Figure(go.Bar(y=e["Index"], x=e["eps_yoy"], orientation="h",
                                   marker_color=["#16a34a" if a > 0 else "#dc2626" for a in e["eps_accel"].fillna(0)],
                                   text=[f"{y:+.1f}%" for y in e["eps_yoy"]],
                                   textposition="auto"))
            fig.update_layout(height=max(320, 30 * len(e)), margin=dict(l=10, r=10, t=30, b=10),
                              title="Implied index EPS growth YoY (green = accelerating)")
            st.plotly_chart(fig, width="stretch")


def flows_tab(d: dict, stt: en.Settings):
    t, fpi, stats = d["table"], d["fpi"], d["fpi_stats"]
    src, err = d["status"].get("fpi_src", "—"), d["status"].get("fpi_err")
    if fpi is None or fpi.empty:
        st.info(f"NSDL FPI data could not be loaded{': ' + err if err else ''}. The Flows leg is left out "
                "and the other legs are re-weighted.")
    else:
        dates = sorted(fpi["date"].unique())
        last = pd.Timestamp(dates[-1])
        st.caption(f"Foreign portfolio investors' net equity buying by sector, from NSDL fortnightly reports "
                   f"(latest fortnight ending {last:%d %b %Y}; source: {src}). Window for scoring: last "
                   f"{stt.fpi_periods} fortnights.")
        s2 = stats.sort_values("net_cr")
        fig = go.Figure(go.Bar(y=s2["sector"], x=s2["net_cr"], orientation="h",
                               marker_color=["#16a34a" if v > 0 else "#dc2626" for v in s2["net_cr"]],
                               text=[f"{v:+,.0f} Cr · {p:+.2f}% of holdings" for v, p in zip(s2["net_cr"], s2["flow_pct"])],
                               textposition="auto"))
        fig.update_layout(height=max(380, 26 * len(s2)), margin=dict(l=10, r=10, t=40, b=10),
                          title=f"FPI net equity investment, last {stt.fpi_periods} fortnights (₹ crore)")
        st.plotly_chart(fig, width="stretch")
        piv = fpi.pivot_table(index="sector", columns="date", values="net_cr", aggfunc="sum").fillna(0)
        piv = piv.loc[piv.abs().sum(axis=1).sort_values(ascending=False).index]
        piv = pd.concat([piv.sum().to_frame("All sectors (total)").T, piv])
        lim = float(np.nanpercentile(np.abs(piv.iloc[1:].values), 95)) or 1.0
        hm = go.Figure(go.Heatmap(z=piv.values, x=[pd.Timestamp(c).strftime("%d %b") for c in piv.columns],
                                  y=list(piv.index), colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim,
                                  hovertemplate="%{y}<br>%{x}: %{z:,.0f} Cr<extra></extra>",
                                  colorbar=dict(title="₹ Cr")))
        hm.update_layout(height=max(420, 24 * len(piv)), margin=dict(l=10, r=10, t=40, b=10),
                         title="Fortnight by fortnight (green = buying, red = selling)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(hm, width="stretch")

    st.markdown("##### By index")
    view = pd.DataFrame({
        "Index": t["Index"], "Flows score": t["Flows_score"], "FPI % of holdings": t["fpi_flow_pct"],
        "FPI accel": t["fpi_accel_pct"], "NSDL sectors": t.get("fpi_sectors", pd.Series("", index=t.index)),
        "FII %": t["fii"], "FII Δ1Q pp": t["fii_d1q"], "DII %": t["dii"], "DII Δ1Q pp": t["dii_d1q"],
        "FII Δ4Q pp": t["fii_d4q"], "DII Δ4Q pp": t["dii_d4q"], "Ownership coverage": t["own_cover"],
    }).sort_values("Flows score", ascending=False, na_position="last")
    thin = view["Ownership coverage"] < 0.5          # too few stocks loaded to describe the index
    view.loc[thin, ["FII %", "FII Δ1Q pp", "DII %", "DII Δ1Q pp", "FII Δ4Q pp", "DII Δ4Q pp"]] = np.nan
    for c in view.columns:
        if c not in ("Index", "NSDL sectors"):
            view[c] = pd.to_numeric(view[c], errors="coerce").astype(float)
    st.dataframe(view.style.background_gradient(subset=["Flows score"], cmap="RdYlGn", vmin=0, vmax=100)
                 .format({"Flows score": "{:.0f}", "FPI % of holdings": "{:+.2f}", "FPI accel": "{:+.2f}",
                          "FII %": "{:.1f}", "DII %": "{:.1f}", "FII Δ1Q pp": "{:+.2f}", "DII Δ1Q pp": "{:+.2f}",
                          "FII Δ4Q pp": "{:+.2f}", "DII Δ4Q pp": "{:+.2f}", "Ownership coverage": "{:.0%}"},
                         na_rep="—"), width="stretch", hide_index=True, height=min(60 + 35 * len(view), 820))
    st.caption("NSDL reports flows by sector, not by index. Each index is mapped through its constituents' NSE "
               "industry (the same classification NSDL uses), so bank indices share the Financial Services figure "
               "and broad indices get a blend. FII/DII % are market-cap-weighted holdings of the constituents "
               "from quarterly shareholding patterns (Screener.in); pp = percentage points.")

    n_have, n_all = len(d["own"]), len(d["syms"])
    st.markdown(f"**Ownership data:** {n_have} of {n_all} constituent stocks.")
    if st.session_state.get("owner") and not DEMO:
        stale = ds.stale_ownership(d["syms"])
        st.caption(f"{len(stale)} stocks need (re)loading. Each click reads up to ~3 minutes of Screener pages "
                   "at a polite pace (about 80 stocks); data is kept for 3 weeks and included in the snapshot.")
        if stale and st.button("Update ownership data"):
            bar = st.progress(0.0, "Reading shareholding patterns…")
            res = ds.refresh_ownership(d["syms"], budget_s=170,
                                       progress=lambda i, n, s: bar.progress(min(i / n, 1.0), f"{s} ({i}/{n})"))
            bar.empty()
            if res["blocked"]:
                st.warning(f"Screener.in is limiting requests: {res['blocked']}. Updated {res['done']} stocks.")
            else:
                st.success(f"Updated {res['done']} stocks; {res['left']} left.")
            load_ownership.clear()
    elif n_have < n_all:
        st.caption("The site owner refreshes ownership data periodically.")


def picker(universe: pd.DataFrame, cons: dict):
    """Same behaviour as the Forward Matrix / Fair Value apps."""
    st.markdown("##### 🔎 Find a stock")
    q = st.text_input("Symbol or company name", placeholder="e.g. TATASTEEL or Tata Steel", key="pick_q")
    if not q:
        return None
    exact, cands = ds.resolve_symbol(q, universe, online=not DEMO)
    if exact:
        return exact
    if not cands:
        st.warning("No matching NSE scrip found.")
        return None
    choice = st.radio("Did you mean:", [f"{s} — {n}" for s, n in cands], key="pick_choice")
    return choice.split(" — ")[0] if choice else None


def drilldown(d: dict):
    t, cons, stocks = d["table"], d["cons"], d["stocks"]
    if stocks.empty or not cons:
        st.info("Turn on **Compute breadth & drill-down** in the sidebar to rank stocks within sectors.")
        return
    universe = pd.concat([df for df, _ in cons.values()]).drop_duplicates("Symbol")
    sym = picker(universe, cons)
    if sym:
        homes = [INDEX_BY_KEY[k].label for k, (df, _) in cons.items() if sym in set(df["Symbol"])]
        st.success(f"**{sym}** is in: {', '.join(homes) if homes else 'none of the selected indices'}")
        if sym in stocks.columns:
            s = stocks[sym].dropna()
            fig = go.Figure(go.Scatter(x=s.index, y=s, name=sym))
            fig.add_scatter(x=s.index, y=s.rolling(50).mean(), name="50 DMA", line=dict(dash="dot"))
            fig.add_scatter(x=s.index, y=s.rolling(200).mean(), name="200 DMA", line=dict(dash="dash"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
    st.divider()
    options = [r["Index"] for _, r in t.iterrows() if r["key"] in cons]
    pick = st.selectbox("Sector to drill into (ranked by composite)", options)
    key = t.loc[t["Index"] == pick, "key"].iloc[0]
    sec_close = d["sec_close"].get(key)
    if sec_close is None or sec_close.dropna().empty:
        st.info("No price series for this index.")
        return
    tbl = en.stock_table(stocks, cons[key][0], sec_close.dropna())
    if tbl.empty:
        st.info("Not enough constituent price history.")
        return
    row = t[t["key"] == key].iloc[0]
    st.markdown(f"**{pick}** — {row['Signal']} · composite {row['Composite']:.0f} · "
                f"RRG {row['quadrant']} · PE Z {row['pe_z']:+.2f}" if not np.isnan(row["pe_z"])
                else f"**{pick}** — {row['Signal']} · RRG {row['quadrant']}")
    n_top = st.slider("Fetch latest-quarter results from Screener.in for the top N stocks", 0,
                      25 if st.session_state.get("owner") else 10, 0,
                      help="Off by default: one Screener page per stock, cached for 24 h.")
    if n_top:
        extra = []
        bar = st.progress(0.0, "Reading Screener.in…")
        for j, s in enumerate(tbl["Symbol"].head(n_top)):
            r = load_screener(s) if not DEMO else {}
            if r.get("blocked"):
                load_screener.clear()   # don't keep the refusal for 24 h
                st.warning(f"Screener.in is limiting requests, so the remaining stocks were skipped. {r['error']}.")
                break
            extra.append({"Symbol": s, "Stock PE": r.get("pe", np.nan), "Sales YoY %": r.get("sales_yoy", np.nan),
                          "Profit YoY %": r.get("pat_yoy", np.nan), "Quarter": r.get("q_label")})
            bar.progress((j + 1) / n_top, f"Reading Screener.in… {s}")
        bar.empty()
        tbl = tbl.merge(pd.DataFrame(extra), on="Symbol", how="left")
        if not np.isnan(row["pe"]):
            tbl["PE vs sector"] = tbl["Stock PE"] / row["pe"]
    if DEMO:
        from demo_data import demo_ownership
        own = demo_ownership(list(tbl["Symbol"]))
    else:
        own = ds.cached_ownership(list(tbl["Symbol"]))
    if own:
        def own_cols(sym):
            df = own.get(sym)
            if df is None or len(df) < 2:
                return pd.Series({"FII %": np.nan, "FII Δ1Q": np.nan, "DII %": np.nan, "DII Δ1Q": np.nan})
            f, dd = df["FII"].astype(float), df["DII"].astype(float) if "DII" in df else pd.Series([np.nan] * len(df))
            return pd.Series({"FII %": f.iloc[-1], "FII Δ1Q": f.iloc[-1] - f.iloc[-2],
                              "DII %": dd.iloc[-1], "DII Δ1Q": dd.iloc[-1] - dd.iloc[-2]})
        tbl = pd.concat([tbl, tbl["Symbol"].apply(own_cols)], axis=1)
    num = {c: "{:+.1f}" for c in ["1M %", "3M %", "Rel 3M vs sector", "Rel 1M vs sector", "From 52W high %",
                                  "Sales YoY %", "Profit YoY %"] if c in tbl} | {"Price": "{:,.2f}", "RS score": "{:.0f}",
                                                                              "Stock PE": "{:.1f}", "PE vs sector": "{:.2f}",
                                                                              "FII %": "{:.1f}", "DII %": "{:.1f}",
                                                                              "FII Δ1Q": "{:+.2f}", "DII Δ1Q": "{:+.2f}"}
    sty = tbl.style.background_gradient(subset=["RS score"], cmap="RdYlGn", vmin=0, vmax=100) \
        .format({k: v for k, v in num.items() if k in tbl}, na_rep="—")
    st.dataframe(sty, width="stretch", hide_index=True, height=min(60 + 35 * len(tbl), 720))
    st.download_button("⬇️ Download stock list (CSV)", tbl.to_csv(index=False).encode(),
                       f"{key.lower()}_stocks_{datetime.now():%Y%m%d}.csv", "text/csv")


def snapshot_zip() -> bytes:
    """Everything in the server cache as a seed snapshot (data/seed/*.parquet + manifest.json)."""
    import tempfile
    from pathlib import Path

    import store

    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        files = store.export_seed(Path(tmp) / "seed")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f"data/seed/{f.name}")
    return buf.getvalue()


def method_tab(d: dict, stt: en.Settings):
    t = d["table"]
    st.markdown(f"""
**How each sector is scored (0–100, higher = more attractive to rotate into)**

| Leg | Weight | What it measures |
|---|---|---|
| Valuation | {stt.weights['Valuation']} | PE Z-score vs the index's own {stt.pe_years}-year history → score = 50 − 25 × Z |
| Momentum | {stt.weights['Momentum']} | RRG vs benchmark: RS-Ratio (60%) and RS-Momentum (40%), half absolute position, half rank vs other indices |
| Breadth | {stt.weights['Breadth']} | % constituents above 50 & 200 DMA, near-52W-high minus near-low, 1-month change |
| Earnings | {stt.weights['Earnings']} | Implied index EPS (price ÷ PE) YoY growth + acceleration vs 3 months ago, ranked |
| Flows | {stt.weights['Flows']} | FPI net equity buying by sector over the last {stt.fpi_periods} fortnights as % of FPI holdings (NSDL), 70%, plus the 1-quarter change in FII + DII ownership of the constituents, 30% |

A leg with no data is left out and the remaining weights are rescaled. **Confidence** is High when at least four
legs are present, agree (spread < 18 points) and breadth covers ≥ 80% of constituents; Medium with three
legs; otherwise Low.

**Signal** combines the RRG quadrant with the valuation zone — e.g. *Improving + cheap* = early rotate-in,
*Weakening + rich* = rotate out, *Lagging + cheap* = value watch (wait for the turn).
**Tier** comes from the composite (≥ 65 Overweight, ≥ 52 Accumulate, ≥ 40 Neutral, else Underweight) but is
held back when it would contradict the signal (red → at most Neutral, orange or yellow → at most Accumulate,
green → at least Neutral), so *Overweight* always needs price confirmation.

**PE Z-score zones:** ≤ −1.5 deeply undervalued · −1.5 to −0.75 undervalued · ±0.75 fair · +0.75 to +1.5
overvalued · ≥ +1.5 deeply overvalued.
""")
    st.markdown("**Data sources used this run**")
    st.caption(f"Constituent stock prices: {d['status'].get('stocks_src', '—')}")
    st.dataframe(t[["Index", "price_src", "pe_src", "cons_src", "pe_obs", "coverage"]].rename(columns={
        "price_src": "Prices", "pe_src": "PE history", "cons_src": "Constituents", "pe_obs": "PE days",
        "coverage": "Breadth coverage"}).style.format({"Breadth coverage": "{:.0%}"}),
        width="stretch", hide_index=True)
    errs = d["status"].get("pe_err", {}) | {f"{k} (constituents)": v for k, v in d["status"].get("cons_err", {}).items()}
    if errs:
        with st.expander(f"⚠️ {len(errs)} live fetches failed"):
            for k, v in errs.items():
                st.write(f"**{k}** — {v}")
            st.caption("NSE and niftyindices often block cloud servers outside India. A data snapshot in the "
                       "repo (below) lets the app start from saved data and only top it up.")
    import store
    n_seed, seed_age = store.seed_info()
    st.caption(f"Warm-start snapshot in the repo: {n_seed} datasets, {store.age_text(seed_age)} old.")
    if not DEMO and st.session_state.get("owner"):
        st.markdown("**Data snapshot (owner)**")
        st.caption("After a reboot or sleep the app starts from this snapshot and downloads only the days since. "
                   "Update it weekly: prepare and download it here (best on your own PC, where NSE is reachable), "
                   "unzip, and upload the data/seed folder to GitHub — or run `python make_snapshot.py`.")
        if st.button("Prepare snapshot from the current server data"):
            st.session_state["snap"] = snapshot_zip()
        if st.session_state.get("snap"):
            st.download_button("⬇️ Download data snapshot (data/seed)", st.session_state["snap"],
                               f"sector_rotation_seed_{datetime.now():%Y%m%d}.zip", "application/zip")


def footer():
    st.markdown(f"""<hr><div class="small">
<b>Sector Rotation Screener</b> · Developed by {AUTHOR}. Data: NSE / niftyindices.com, Yahoo Finance, Screener.in.
For education and research only — not investment advice. Not SEBI-registered. Data may be delayed or incomplete;
verify before acting. © {datetime.now():%Y} {AUTHOR}.</div>""", unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================
def main():
    hero()
    stt, cfg = sidebar()
    if not cfg["groups"]:
        st.warning("Pick at least one index group in the sidebar.")
        st.stop()
    try:
        d = build(stt, cfg)
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong while building the board: {e}")
        st.stop()
    t = d["table"]
    st.caption(f"Benchmark: {BENCHMARKS[cfg['bench_key']].label} · {len(t)} indices · "
               f"last price date {d['bench'].index.max():%d %b %Y}")
    labels = [*t["price_src"], *t["pe_src"], *t["cons_src"], d["status"].get("stocks_src", "")]
    n_cached = sum("live fetch failed" in str(x) for x in labels)
    if n_cached:
        st.info(f"{n_cached} data series could not be refreshed just now and are shown from the last good "
                "download (see **Method & data**). The app retries automatically later.")
    kpis(t)
    tabs = st.tabs(["🧭 Rotation board", "🔄 RRG", "💰 Valuation", "🌡️ Breadth & earnings", "💸 Flows",
                    "🔍 Sector drill-down", "ℹ️ Method & data"])
    with tabs[0]:
        board(t)
    with tabs[1]:
        grp = st.multiselect("Show groups", cfg["groups"], default=cfg["groups"], key="rrg_groups")
        keep = set(t.loc[t["Group"].isin(grp), "Index"])
        rrg_chart({k: v for k, v in d["tails"].items() if k in keep}, t)
    with tabs[2]:
        valuation_tab(d, stt)
    with tabs[3]:
        breadth_earn_tab(d)
    with tabs[4]:
        flows_tab(d, stt)
    with tabs[5]:
        drilldown(d)
    with tabs[6]:
        method_tab(d, stt)
    footer()


if __name__ == "__main__":
    main()
