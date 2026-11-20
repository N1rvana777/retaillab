"""Token-bucket rate limiter middleware.

Protects the API from abuse and ensures the quant_engine
doesn't get overwhelmed with concurrent optimization requests.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token bucket rate limiter.

    For production with multiple workers, replace with Redis-based
    sliding window (see redis rate limiting patterns).
    """

    def __init__(self, app, requests_per_minute: int = 60):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._rpm = requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Check rate limit before processing request."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._buckets[client_ip] = [
            t for t in self._buckets[client_ip] if now - t < 60
        ]

        if len(self._buckets[client_ip]) >= self._rpm:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        self._buckets[client_ip].append(now)
        return await call_next(request)
