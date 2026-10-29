"""Synthetic data used only by the offline tests (SRS_DEMO=1). Never shown to real users."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sectors import INDEX_BY_KEY


def _walk(n, drift, vol, seed, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def demo_bundle(keys: list[str], bench_key: str):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 6)
    n = len(idx)
    cons, index_px, pe = {}, {}, {}
    stock_cols = {}
    for k_i, key in enumerate(sorted(set(keys) | {bench_key})):
        d = INDEX_BY_KEY[key]
        syms = [f"{key[:4]}{j:02d}" for j in range(12)]
        cons[key] = (pd.DataFrame({"Symbol": syms, "Company": [f"{d.label} Co {j}" for j in range(12)],
                                   "Industry": d.label}), "demo")
        members = []
        for j, s in enumerate(syms):
            px = _walk(n, 0.0003 + 0.0001 * (k_i % 5 - 2), 0.015, 1000 * k_i + j)
            stock_cols[s] = px
            members.append(px / px[0])
        ix = 1000 * np.mean(members, axis=0)
        index_px[key] = pd.Series(ix, idx)
        eps = 1000 / 20 * np.exp(np.linspace(0, 0.25 + 0.05 * (k_i % 4), n))
        pe[key] = pd.DataFrame({"pe": ix / eps, "pb": ix / eps / 6, "dy": 1.2}, index=idx)
    stocks = pd.DataFrame(stock_cols, index=idx)
    return cons, pd.DataFrame(index_px), pe, stocks
