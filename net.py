"""Polite HTTP layer: per-site pacing, retries with back-off, and a cool-down
when a site signals it is being overloaded (HTTP 403 / 429).

Every request to NSE, niftyindices and Screener goes through `request()`.
The aim is to look like one careful visitor, not a burst of traffic:

* one shared, keep-alive session per site (cookies and connections reused)
* a minimum gap between requests to the same site, with a little jitter,
  enforced across all threads
* automatic retries on 429 / 5xx with exponential back-off, honouring
  the server's Retry-After header
* if a site still answers 403 / 429, it is left alone for a cool-down
  period instead of being hammered again by the next index in the loop

Developed by Dr. Shantanu Samanta.
"""
from __future__ import annotations

import random
import threading
import time
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = 20

# minimum seconds between two requests to the same host
MIN_GAP = {
    "www.niftyindices.com": 0.8,
    "www.nseindia.com": 1.2,
    "www.screener.in": 2.0,
    "www.fpi.nsdl.co.in": 1.5,
}
DEFAULT_GAP = 0.3
COOLDOWN_S = 15 * 60          # leave a site alone this long after 403/429

_lock = threading.Lock()
_next_slot: dict[str, float] = {}
_cool_until: dict[str, float] = {}
_sessions: dict[str, requests.Session] = {}


class Blocked(Exception):
    """The site is refusing or rate-limiting us; it is in cool-down."""


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def cooling(host: str) -> float:
    """Seconds left in this host's cool-down (0 if none)."""
    with _lock:
        return max(0.0, _cool_until.get(host, 0.0) - time.time())


def start_cooldown(host: str, seconds: float = COOLDOWN_S) -> None:
    with _lock:
        _cool_until[host] = max(_cool_until.get(host, 0.0), time.time() + seconds)


def reset() -> None:
    """Forget pacing and cool-downs (used by tests)."""
    with _lock:
        _next_slot.clear()
        _cool_until.clear()
        _sessions.clear()


def throttle(host: str) -> None:
    """Block until this host's next request slot. Slots are reserved under a lock,
    so parallel threads queue up one gap apart instead of firing together."""
    gap = MIN_GAP.get(host, DEFAULT_GAP)
    with _lock:
        now = time.monotonic()
        slot = max(now, _next_slot.get(host, 0.0))
        _next_slot[host] = slot + gap + random.uniform(0, gap * 0.3)
    wait = slot - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def session(host: str, headers: dict | None = None) -> requests.Session:
    with _lock:
        s = _sessions.get(host)
        if s is None:
            s = requests.Session()
            retry = Retry(total=3, connect=2, read=2, backoff_factor=2.0,
                          status_forcelist=(429, 500, 502, 503, 504),
                          allowed_methods=frozenset({"GET", "POST"}),
                          respect_retry_after_header=True, raise_on_status=False)
            adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            s.headers.update({"User-Agent": UA, "Accept-Language": "en-IN,en;q=0.9"})
            _sessions[host] = s
        if headers:
            s.headers.update(headers)
        return s


def request(method: str, url: str, *, headers: dict | None = None, timeout: float = TIMEOUT,
            **kw) -> requests.Response:
    """Paced, retried request. Raises Blocked if the host is cooling down or refuses us."""
    host = _host(url)
    left = cooling(host)
    if left:
        raise Blocked(f"{host} is rate-limiting this server; pausing requests for {left / 60:.0f} more min")
    throttle(host)
    s = session(host)
    try:
        r = s.request(method, url, headers=headers, timeout=timeout, **kw)
    except requests.exceptions.RetryError as e:
        start_cooldown(host)
        raise Blocked(f"{host} kept returning errors; pausing requests for {COOLDOWN_S // 60} min") from e
    if r.status_code in (403, 429):
        wait = COOLDOWN_S
        ra = r.headers.get("Retry-After", "")
        if ra.isdigit():
            wait = max(wait, int(ra))
        start_cooldown(host, wait)
        raise Blocked(f"{host} answered HTTP {r.status_code}; pausing requests for {wait // 60} min")
    return r


def get(url: str, **kw) -> requests.Response:
    return request("GET", url, **kw)


def post(url: str, **kw) -> requests.Response:
    return request("POST", url, **kw)
