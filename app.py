"""Sector Rotation Screener — NSE sectoral, thematic and size indices.

Four-leg rotation model: Valuation (PE Z-score), Momentum (RRG vs benchmark),
Breadth (constituents above 50/200 DMA) and Earnings (implied index EPS growth),
with a drill-down into each sector's strongest stocks.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_screener(symbol: str) -> dict:
    return ds.fetch_screener_company(symbol)


# ============================================================================
# Sidebar
# ============================================================================
def sidebar() -> tuple[en.Settings, dict]:
    sb = st.sidebar
    sb.markdown(f"### 🧭 Settings\n<span class='small'>by {AUTHOR}</span>", unsafe_allow_html=True)
    bench_key = sb.selectbox("Benchmark", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k].label)
    groups = sb.multiselect("Index groups", GROUPS, default=GROUPS)
    with sb.expander("Valuation", expanded=False):
        pe_years = st.select_slider("PE history lookback (years)", [1, 2, 3, 5, 7, 10], value=5)
        cons_only = st.checkbox("Only use PE from Apr 2021 (consolidated basis)", value=True,
                                help="NSE switched index PE from standalone to consolidated earnings in April 2021; "
                                     "older PE is not comparable.")
        z_cheap = st.slider("Cheap if Z ≤", -2.0, 0.0, -0.75, 0.05)
        z_rich = st.slider("Rich if Z ≥", 0.0, 2.0, 0.75, 0.05)
        upload = st.file_uploader("Optional PE history CSV (Date, Index, PE, PB, DivYield)", type="csv",
                                  help="Fills in any index whose PE history could not be fetched live.")
    with sb.expander("Momentum (RRG)", expanded=False):
        rrg_len = st.slider("RS-Ratio length (weeks)", 6, 26, 10)
        rrg_mom = st.slider("RS-Momentum lookback (weeks)", 1, 8, 4)
        tail = st.slider("Tail length (weeks)", 3, 15, 6)
    with sb.expander("Leg weights", expanded=False):
        w = {l: st.slider(l, 0, 50, en.DEFAULT_WEIGHTS[l], 5) for l in en.LEGS}
    do_breadth = sb.toggle("Compute breadth & drill-down (downloads constituent prices)", value=True)
    last = st.session_state.get("last_refresh", 0.0)
    wait = REFRESH_COOLDOWN - (time.time() - last)
    if sb.button("🔄 Refresh live data", width="stretch", disabled=wait > 0,
                 help="Re-checks the sources. Only data older than its refresh interval is downloaded again "
                      "(prices 6 h, PE 12 h, constituents 7 days)."):
        st.session_state["last_refresh"] = time.time()
        st.cache_data.clear()
        st.rerun()
    if wait > 0:
        sb.caption(f"Refresh available again in {wait / 60:.0f} min — protects the data sources from repeat downloads.")
    cfg = dict(bench_key=bench_key, groups=groups, upload=upload, do_breadth=do_breadth)
    return en.Settings(pe_years=pe_years, consolidated_only=cons_only, rrg_len=rrg_len, rrg_mom=rrg_mom,
                       tail=tail, weights=w, z_cheap=z_cheap, z_rich=z_rich), cfg


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
        cons, cons_err = {}, {}
        stocks, stocks_src = pd.DataFrame(), "—"
        if cfg["do_breadth"]:
            box.write("Index constituents (niftyindices → NSE)…")
            cons, cons_err = load_constituents(keys)
            syms = sorted({s for k, (df, _) in cons.items() for s in df["Symbol"]})
            if syms:
                box.write(f"Prices for {len(syms)} constituent stocks (yfinance)…")
                stocks, stocks_src = load_stock_prices(tuple(syms))
        box.update(label="Data ready", state="complete")

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
    status.update(pe_err=pe_err, cons_err=cons_err, stocks_src=stocks_src)
    return dict(table=table, tails=rrg_tails, prices=prices, sec_close=sec_close, pe=pe, cons=cons, stocks=stocks,
                bench=bench, status=status, sel=sel)


# ============================================================================
# UI pieces
# ============================================================================
def hero():
    st.markdown(f"""
<div class="hero">
  <h1>🧭 Sector Rotation Screener</h1>
  <p>NSE sectoral, thematic and size indices scored on Valuation · Momentum · Breadth · Earnings —
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
        "Breadth": t["Breadth_score"], "Earnings": t["Earnings_score"],
        "PE": t["pe"], "PE Z": t["pe_z"], "Valuation zone": t["val_label"], "RRG": t["quadrant"],
        "3M %": t["ret_3m"], "Rel 3M %": t["rel_3m"], "% > 200 DMA": t["pct_200"],
        "EPS YoY %": t["eps_yoy"],
    })
    for c in show.columns:
        if c not in ("Index", "Group", "Signal", "Tier", "Confidence", "Valuation zone", "RRG"):
            show[c] = pd.to_numeric(show[c], errors="coerce").astype(float)
    score_cols = ["Composite", "Valuation", "Momentum", "Breadth", "Earnings"]
    sty = (show.style
           .background_gradient(subset=score_cols, cmap="RdYlGn", vmin=0, vmax=100)
           .background_gradient(subset=["PE Z"], cmap="RdYlGn_r", vmin=-2, vmax=2)
           .format({c: "{:.0f}" for c in score_cols} | {"PE": "{:.1f}", "PE Z": "{:+.2f}", "3M %": "{:+.1f}",
                    "Rel 3M %": "{:+.1f}", "% > 200 DMA": "{:.0f}", "EPS YoY %": "{:+.1f}"}, na_rep="—"))
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
    n_top = st.slider("Fetch latest-quarter results from Screener.in for the top N stocks", 0, 25, 0,
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
    num = {c: "{:+.1f}" for c in ["1M %", "3M %", "Rel 3M vs sector", "Rel 1M vs sector", "From 52W high %",
                                  "Sales YoY %", "Profit YoY %"] if c in tbl} | {"Price": "{:,.2f}", "RS score": "{:.0f}",
                                                                              "Stock PE": "{:.1f}", "PE vs sector": "{:.2f}"}
    sty = tbl.style.background_gradient(subset=["RS score"], cmap="RdYlGn", vmin=0, vmax=100) \
        .format({k: v for k, v in num.items() if k in tbl}, na_rep="—")
    st.dataframe(sty, width="stretch", hide_index=True, height=min(60 + 35 * len(tbl), 720))
    st.download_button("⬇️ Download stock list (CSV)", tbl.to_csv(index=False).encode(),
                       f"{key.lower()}_stocks_{datetime.now():%Y%m%d}.csv", "text/csv")


def snapshot_zip(d: dict) -> bytes:
    """Constituent lists + PE history in the same format the app reads from data/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, (df, _) in d["cons"].items():
            z.writestr(f"data/constituents/{INDEX_BY_KEY[k].csv}",
                       df.rename(columns={"Company": "Company Name"}).to_csv(index=False))
        frames = [p.reset_index().assign(Index=INDEX_BY_KEY[k].nse_name) for k, p in d["pe"].items()]
        if frames:
            pe = pd.concat(frames).rename(columns={"index": "Date", "pe": "PE", "pb": "PB", "dy": "DivYield"})
            z.writestr("data/pe_history.csv", pe[["Date", "Index", "PE", "PB", "DivYield"]].to_csv(index=False))
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

A leg with no data is left out and the remaining weights are rescaled. **Confidence** is High when all four
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
            st.caption("NSE and niftyindices often block cloud servers outside India. Run the app locally once, "
                       "download the data snapshot below and commit its data/ folder to the repo — the app then "
                       "falls back to those files when the live fetch fails.")
    if d["cons"] or d["pe"]:
        st.download_button("⬇️ Download data snapshot (constituents + PE history)", snapshot_zip(d),
                           f"sector_rotation_data_{datetime.now():%Y%m%d}.zip", "application/zip")


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
    tabs = st.tabs(["🧭 Rotation board", "🔄 RRG", "💰 Valuation", "🌡️ Breadth & earnings",
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
        drilldown(d)
    with tabs[5]:
        method_tab(d, stt)
    footer()


if __name__ == "__main__":
    main()
