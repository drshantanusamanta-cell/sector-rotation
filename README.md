# 🧭 Sector Rotation Screener

A Streamlit app that scores NSE sectoral, thematic and size indices on five legs and shows where
money may be rotating next, with a drill-down into each sector's strongest stocks.

Developed by **Dr Shantanu Samanta** — Physician · Quantitative Market Analyst & Trading-Systems Developer · Kolkata, India.

## What it does

| Leg | Source | What it measures |
|---|---|---|
| **Valuation** | niftyindices.com PE/PB/DY history | PE Z-score vs the index's own history (default 5 years, consolidated basis from Apr 2021) |
| **Momentum** | Yahoo Finance / niftyindices prices | Relative Rotation Graph vs Nifty 50 (or Nifty 500): Leading, Weakening, Lagging, Improving |
| **Breadth** | Constituents (niftyindices) + Yahoo prices | % of stocks above 50 & 200 DMA, near highs vs lows, 1-month change |
| **Earnings** | Index price ÷ PE | Implied index EPS growth YoY and whether it is accelerating |
| **Flows** | NSDL fortnightly FPI data + Screener shareholding | FPI net equity buying by sector (last 6 fortnights, as % of FPI holdings) and the quarterly change in FII + DII ownership of the constituents |

Default weights: Valuation 25, Momentum 30, Breadth 15, Earnings 15, Flows 15 (owner-adjustable).

### Flows tab

- **FPI by sector:** NSDL publishes foreign portfolio investors' net investment by sector every fortnight.
  The app reads the latest 12 reports (cached a day) and shows net buying per sector, a fortnight-by-
  fortnight heatmap, and the Flows score per index. NSDL uses the same sector names as NSE's industry
  classification, so each index is mapped through its constituents' industries — bank indices share the
  Financial Services figure, and broad indices get a blend.
- **FII / DII ownership:** market-cap-weighted FII and DII holding % of each index's constituents, with the
  change over 1 and 4 quarters, from the shareholding table on Screener.in. It needs one Screener page per
  stock, so it is loaded in the background: by `make_snapshot.py` (~20 minutes the first time, then only
  stocks older than three weeks) or by the owner's **Update ownership data** button (about 80 stocks per
  click). Visitors never trigger these downloads. The drill-down shows each stock's FII/DII % and change.

Each sector gets a 0–100 composite, a tier (Overweight / Accumulate / Neutral / Underweight),
a confidence rating, and an action signal from the RRG quadrant × valuation zone
(e.g. *Improving + cheap = rotate in early*, *Weakening + rich = rotate out*).

**Sector drill-down** ranks constituents by relative strength vs their sector, and can pull the latest
quarter's sales and profit growth plus stock PE from Screener.in for the top N names.
The stock search works like the Forward Matrix and Fair Value apps: an exact symbol opens directly,
otherwise you pick from a list of probable scrips.

Covered: Auto, Bank, Private Bank, PSU Bank, Financial Services, FMCG, IT, Pharma, Healthcare, Metal,
Realty, Media, Consumer Durables, Oil & Gas, Energy, Infrastructure, Commodities, India Consumption,
PSE, MNC, India Defence, Nifty Next 50, Midcap 150, Smallcap 250, Nifty 500.

## Files

```
sector-rotation/
├── app.py              Streamlit UI
├── engine.py           scoring maths (no network)
├── data_sources.py     live fetchers + parsers
├── net.py              paced, retried HTTP with per-site cool-down
├── store.py            disk cache (download once, then top up) + warm-start seed
├── make_snapshot.py    builds data/seed/ on your PC
├── sectors.py          index list, Yahoo tickers, niftyindices names
├── demo_data.py        synthetic data for the tests only
├── tests/test_core.py  offline tests (pytest)
├── data/               optional fallback files (see data/README.md)
├── requirements.txt
└── .streamlit/config.toml
```

No API keys are needed.

## Deploy on Streamlit Community Cloud

1. Create a new **public** GitHub repo (e.g. `sector-rotation`) → *uploading an existing file* →
   drag in everything from this folder (including the `.streamlit`, `data` and `tests` folders) → **Commit**.
2. Go to **share.streamlit.io** → **New app** → pick the repo → main file `app.py` → **Deploy**.
3. The first load takes about 3 minutes (about 550 stock price series plus 24 PE histories).
   After that the app starts in seconds and only tops up what is due (see below).

### Owner settings (password)

Visitors see only **Benchmark** and **Index groups**. Everything else — PE lookback, Z thresholds,
RRG lengths, leg weights, breadth toggle, PE CSV upload, **Refresh live data** and the data snapshot —
sits behind a password in the sidebar's **🔒 Owner** box. Visitors can pull Screener results for up to
10 stocks at a time; the owner up to 25.

1. Streamlit Cloud → your app → **Settings → Secrets**, paste:
   `OWNER_PASSWORD = "your-long-password"` → Save. (Locally: copy `.streamlit/secrets.toml.example`
   to `.streamlit/secrets.toml`; it is git-ignored.)
2. Unlock in the sidebar. Changes apply to your session only, until you click
   **Defaults for all visitors → Save these as the default**. That lasts until the app restarts; to make
   it permanent, download `owner_settings.json` there and upload it to the repo's `data/` folder.
3. Three wrong passwords lock the box for 2 minutes. Without a password set, the owner box is disabled.

### Warm start after sleep or reboot

Streamlit Cloud sleeps apps that have had no visitors for a while; waking can start with an empty
cache. If the repo has a snapshot in `data/seed/`, the app starts from it and downloads only the days
since, instead of everything.

- On your own PC: `python make_snapshot.py` (about 3–5 minutes), then upload the `data/seed` folder to
  GitHub, replacing the old one. Or, as owner, use **Method & data → Prepare snapshot**.
- Refresh the snapshot weekly — after 7 days the app does its weekly full Yahoo re-download anyway.

### If NSE blocks the cloud server

Streamlit Cloud runs on US servers, and NSE / niftyindices sometimes refuse them. When that happens
the Method & data tab lists the failed fetches, and the app still works on Yahoo prices (momentum),
but valuation, earnings and breadth go missing. Fix:

1. Run the app on your own PC once (below).
2. Method & data → **Download data snapshot**.
3. Unzip it and upload its `data/` folder to the GitHub repo. The app uses those files whenever the
   live fetch fails. Refresh the snapshot every week or two.

You can also upload a PE CSV (`Date, Index, PE, PB, DivYield`) from the sidebar.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests (no internet needed): `pip install pytest && pytest -q tests`

## How the app avoids being blocked or throttled

- **Downloads once, then tops up.** Everything fetched is kept in a disk cache (`.cache/`). Later visits
  reuse it and download only what is missing: a few new days of prices or PE, never the whole history
  again. Constituents are re-checked weekly, prices every 6 h, PE every 12 h, Screener pages daily, and
  Yahoo prices get one full re-download a week. A warm start takes seconds instead of ~3 minutes.
- **Paced like a single visitor.** Requests to each site are spaced out (niftyindices ≥ 0.8 s,
  NSE ≥ 1.2 s, Screener ≥ 2 s, with a little jitter), across all threads, over one keep-alive session.
  Yahoo is fetched in batches of 50 with a pause between batches.
- **Backs off when asked.** HTTP 429 and server errors are retried with growing waits, honouring
  Retry-After. If a site still answers 403/429, the app leaves it alone for 15 minutes rather than
  trying again for every index. Screener drill-down stops at the first refusal.
- **Never goes blank.** If a live fetch fails, the last good copy is used, labelled "cached … ago",
  and a note appears above the board.
- **Refresh button has a 10-minute cool-down** per visitor, and it only re-downloads data that is
  actually due.

These measures keep the load on each source low; they don't disguise the app. The cache folder is
in `.gitignore`, so it is never uploaded to GitHub.

## Notes and limits

- NSE changed index PE from standalone to consolidated earnings in April 2021; by default only PE from
  that date is used, so Z-scores are not distorted.
- Implied EPS jumps when an index's constituents change or when members swing between profit and loss
  (Media, Metal). The earnings leg is ranked and clipped to limit this, but treat it as a guide.
- Yahoo has long history only for some NSE indices; the rest use niftyindices prices, or an equal-weight
  proxy of constituents as a last resort (then the earnings leg is skipped for that index).
- niftyindices.com and Screener.in have no official API; their page layouts can change. Index names for
  the PE history are set in `sectors.py` (checked October 2026).

For education and research only — not investment advice. Not SEBI-registered.
© Dr Shantanu Samanta.
