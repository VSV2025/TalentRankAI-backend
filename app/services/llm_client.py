"""
Shared Groq LLM client — sliding-window rate limiting, retry-on-429, prompt cache.
Both scoring.py and debate.py import from here so throttling is consistent
across all pipeline layers and across consecutive runs (same process window).
"""
import hashlib
import logging
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt-hash result cache ───────────────────────────────────────────────────
_cache: dict[str, str] = {}

# ── Per-model retry schedules after successive 429s ───────────────────────────
# Fast model (25 RPM): window clears quickly — short waits are fine.
# Reasoning model (4 RPM / ~6K TPM): window needs longer to clear.
_FAST_RETRY_WAITS = (5, 10, 20, 40)
_SLOW_RETRY_WAITS = (15, 30, 60, 120)


def _get_retry_waits(model: str) -> tuple:
    m = model.lower()
    if any(tag in m for tag in ("70b", "versatile", "8x7b", "opus", "large")):
        return _SLOW_RETRY_WAITS
    return _FAST_RETRY_WAITS


# ── Sliding-window rate limiter ────────────────────────────────────────────────
class _SlidingWindowLimiter:
    """
    Tracks call timestamps in a rolling 60-second window.
    Blocks until making one more call won't exceed max_per_minute.
    This correctly handles bursts AND cross-run interference (previous
    run's calls are visible in the window until they age out).
    """

    def __init__(self, max_per_minute: int, window: float = 60.0):
        self._max = max_per_minute
        self._window = window
        self._ts: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, label: str = "") -> None:
        with self._lock:
            self._evict()
            if len(self._ts) >= self._max:
                # Wait for the oldest call to age out of the window
                wait = (self._ts[0] + self._window) - time.monotonic() + 0.1
                if wait > 0:
                    logger.info(
                        f"[rate_limit] {label}: {len(self._ts)}/{self._max} calls in 60s "
                        f"window — pausing {wait:.1f}s"
                    )
                    self._lock.release()
                    try:
                        time.sleep(wait)
                    finally:
                        self._lock.acquire()
                    self._evict()
            self._ts.append(time.monotonic())

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()


# Per-model limiters — created lazily.
# Groq free tier: 30 RPM per model, but 70b also has ~6K TPM.
# We use conservative limits: 25 RPM for fast, 4 RPM for 70b (TPM-limited).
_limiters: dict[str, _SlidingWindowLimiter] = {}
_limiter_registry_lock = threading.Lock()


def _get_limiter(model: str) -> _SlidingWindowLimiter:
    with _limiter_registry_lock:
        if model not in _limiters:
            m = model.lower()
            if any(tag in m for tag in ("70b", "versatile", "8x7b", "opus", "large")):
                # TPM-constrained: ~6K TPM / ~1500 tokens per call ≈ 4 safe calls/min
                _limiters[model] = _SlidingWindowLimiter(max_per_minute=4)
            else:
                # RPM-constrained: stay safely under 30 RPM
                _limiters[model] = _SlidingWindowLimiter(max_per_minute=25)
        return _limiters[model]


# ── Public API ─────────────────────────────────────────────────────────────────

def call_groq(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Rate-limited, cached, retry-on-429 call to Groq.
    Returns the text response, or None only after all retries are exhausted.
    """
    ck = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    if ck in _cache:
        logger.debug(f"[llm_cache] hit {model}")
        return _cache[ck]

    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        logger.error("[groq] openai package not installed — pip install openai")
        return None

    limiter = _get_limiter(model)
    client = OpenAI(api_key=api_key, base_url=base_url)
    retry_waits = _get_retry_waits(model)

    for attempt in range(len(retry_waits) + 1):
        limiter.acquire(label=model)
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip() if resp.choices else None
            if text:
                _cache[ck] = text
            return text

        except RateLimitError as e:
            if attempt >= len(retry_waits):
                logger.error(
                    f"[groq] 429 on {model}: all {len(retry_waits) + 1} attempts exhausted — "
                    f"caller will use heuristic fallback"
                )
                return None
            # Use Retry-After header when Groq provides it; otherwise use our schedule
            wait = retry_waits[attempt]
            try:
                wait = max(wait, int(e.response.headers.get("retry-after", wait)))
            except Exception:
                pass
            logger.warning(
                f"[groq] 429 on {model} — retry {attempt + 1}/{len(retry_waits)} "
                f"in {wait}s (rate limit window clearing)"
            )
            time.sleep(wait)

        except Exception as e:
            logger.error(f"[groq] call failed ({model}): {e}")
            return None

    return None


def clear_cache() -> None:
    """Wipe the in-process prompt cache."""
    _cache.clear()
    logger.info("[llm_cache] cleared")
