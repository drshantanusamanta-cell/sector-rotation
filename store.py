"""Small on-disk cache so each piece of data is downloaded once, then topped up.

Survives app restarts and the in-memory Streamlit cache being cleared, which is
what keeps repeated visits (or the Refresh button) from re-downloading everything.
On Streamlit Cloud the folder lasts until the app is rebooted, redeployed or woken
from sleep. To avoid a cold start then, the cache can be *seeded* from a snapshot
committed to the repo (data/seed/): anything missing from the cache is read from
the seed, keeps the seed's own timestamp, and is then topped up as usual.

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import tempfile
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("SRS_CACHE_DIR", Path(__file__).parent / ".cache"))
SEED_DIR = Path(os.getenv("SRS_SEED_DIR", Path(__file__).parent / "data" / "seed"))
MANIFEST = "manifest.json"


def _path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return CACHE_DIR / f"{safe}.pkl"


def load(key: str):
    """Return (object, age in seconds) or (None, inf). Falls back to the repo seed."""
    p = _path(key)
    try:
        with p.open("rb") as f:
            payload = pickle.load(f)
        return payload["obj"], time.time() - payload["ts"]
    except Exception:  # noqa: BLE001  (missing, partial or old-format file)
        return load_seed(key)


# ============================================================================
# Seed snapshot (parquet files + manifest.json, safe to commit to GitHub)
# ============================================================================
_MANIFEST_CACHE: dict = {"mtime": None, "data": {}}


def _manifest() -> dict:
    m = SEED_DIR / MANIFEST
    try:
        mt = m.stat().st_mtime
    except OSError:
        return {}
    if _MANIFEST_CACHE["mtime"] != mt:
        try:
            _MANIFEST_CACHE.update(mtime=mt, data=json.loads(m.read_text()))
        except Exception:  # noqa: BLE001
            _MANIFEST_CACHE.update(mtime=mt, data={})
    return _MANIFEST_CACHE["data"]


def load_seed(key: str):
    import pandas as pd

    entry = _manifest().get("items", {}).get(key)
    if not entry:
        return None, float("inf")
    try:
        df = pd.read_parquet(SEED_DIR / entry["file"])
    except Exception:  # noqa: BLE001
        return None, float("inf")
    kind = entry["kind"]
    if kind == "cons":
        obj = (df, entry.get("label", "snapshot"))
    elif kind == "series":
        obj = df.iloc[:, 0]
    elif kind == "yf":
        obj = {"px": df, "full_at": entry.get("full_at", entry["ts"])}
    else:
        obj = df
    return obj, time.time() - entry["ts"]


def export_seed(out_dir: Path) -> list[Path]:
    """Write every cached dataset (except per-stock Screener pages) as parquet + manifest."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, files = {}, []
    for p in sorted(CACHE_DIR.glob("*.pkl")):
        key = p.stem
        if key.startswith("scr"):
            continue
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)
        except Exception:  # noqa: BLE001
            continue
        obj, ts = payload["obj"], payload["ts"]
        entry = {"ts": ts, "file": f"{key}.parquet"}
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], pd.DataFrame):
            df, entry["kind"] = obj[0], "cons"
            entry["label"] = str(obj[1]).split(" · ")[0]
        elif isinstance(obj, dict) and "px" in obj:
            df, entry["kind"], entry["full_at"] = obj["px"], "yf", obj.get("full_at", ts)
        elif isinstance(obj, pd.Series):
            df, entry["kind"] = obj.to_frame("value"), "series"
        elif isinstance(obj, pd.DataFrame):
            df, entry["kind"] = obj, "frame"
        else:
            continue
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(out_dir / entry["file"])
        files.append(out_dir / entry["file"])
        items[key] = entry
    (out_dir / MANIFEST).write_text(json.dumps({"created": time.time(), "items": items}, indent=1))
    files.append(out_dir / MANIFEST)
    return files


def seed_info() -> tuple[int, float]:
    """(number of datasets in the repo seed, age in seconds of the snapshot)."""
    m = _manifest()
    if not m:
        return 0, float("inf")
    return len(m.get("items", {})), time.time() - m.get("created", 0)


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
