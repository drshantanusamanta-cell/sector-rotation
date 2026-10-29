"""Small on-disk cache so each piece of data is downloaded once, then topped up.

Survives app restarts and the in-memory Streamlit cache being cleared, which is
what keeps repeated visits (or the Refresh button) from re-downloading everything.
On Streamlit Cloud the folder lasts until the app is rebooted or redeployed.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("SRS_CACHE_DIR", Path(__file__).parent / ".cache"))


def _path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return CACHE_DIR / f"{safe}.pkl"


def load(key: str):
    """Return (object, age in seconds) or (None, inf)."""
    p = _path(key)
    try:
        with p.open("rb") as f:
            payload = pickle.load(f)
        return payload["obj"], time.time() - payload["ts"]
    except Exception:  # noqa: BLE001  (missing, partial or old-format file)
        return None, float("inf")


def save(key: str, obj, ts: float | None = None) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"ts": time.time() if ts is None else ts, "obj": obj}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _path(key))   # atomic, so a crash never leaves half a file
    except Exception:  # noqa: BLE001  read-only disk etc. — caching is best-effort
        pass


def age_text(seconds: float) -> str:
    if seconds == float("inf"):
        return "never"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.0f} h"
    return f"{seconds / 86400:.0f} days"
