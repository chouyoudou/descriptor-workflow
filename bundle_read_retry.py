"""Bounded recovery of transient failures inside one read-only file operation.

Used by bundle_control.read_repository_file, never by a candidate, whole task,
publication transaction or mutating API request. No paths, bodies or credentials
are logged. Original exceptions and all content/identity checks are preserved.
"""
from __future__ import annotations

from email.utils import parsedate_to_datetime
from functools import wraps
import http.client
import math
import random
import socket
import time
import urllib.error

MAX_ATTEMPTS = 4
MAX_RETRY_SLEEP_SECONDS = 120.0
TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


def _header(headers, name):
    # HTTPMessage is case-insensitive; plain mappings in tests need not be.
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return str(value).strip()
    return None


def _retry_after(value):
    if value is None:
        return None
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                return None
            delay = date.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(delay, 0.0) if math.isfinite(delay) else None


def _delay(exc, failure_index):
    backoff = 2 ** failure_index + random.uniform(0.0, 0.4)
    if isinstance(exc, urllib.error.HTTPError):
        headers = exc.headers
        after_value = _header(headers, "Retry-After")
        after = _retry_after(after_value)
        exhausted = _header(headers, "X-RateLimit-Remaining") == "0"
        # A generic 403 is an authorization failure, not evidence of throttling.
        limited = exc.code == 429 or (exc.code == 403 and (after is not None or exhausted))
        if exc.code not in TRANSIENT_HTTP and not limited:
            return None
        # A malformed server hint is not permission to retry earlier than asked.
        if after_value is not None and after is None:
            return None
        waits = [backoff]
        if after is not None:
            waits.append(after)
        if exhausted:
            try:
                reset = float(_header(headers, "X-RateLimit-Reset"))
            except (ValueError, TypeError):
                return None
            if not math.isfinite(reset):
                return None
            waits.append(max(reset - time.time() + 1.0, 0.0))
        if limited and after is None and not exhausted:
            waits.append(60.0 * 2 ** failure_index)
        return max(waits)
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        return backoff if reason.errno == socket.EAI_AGAIN else None
    if isinstance(reason, (TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return backoff
    return None


def retry_read(function):
    """Decorate a known read-only operation; never reuse for write transactions.

    At most four attempts, with at most 120 seconds of additional sleeping per
    file read. Existing per-request timeouts still apply. A longer server wait
    fails without clipping the hint or issuing an early retry. This is a finite
    transport policy, not a scientific budget or a PRO reasoning-attempt quota.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        slept = 0.0
        for attempt in range(MAX_ATTEMPTS):
            try:
                return function(*args, **kwargs)
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.IncompleteRead) as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                delay = _delay(exc, attempt)
                if delay is None or delay > MAX_RETRY_SLEEP_SECONDS - slept:
                    raise
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                time.sleep(delay)
                slept += delay
        raise AssertionError("unreachable")
    return wrapped
