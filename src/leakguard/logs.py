"""One JSON log line per request — metadata only.

The function accepts only named metadata fields, never message text or headers.
That makes "the guard leaks data into its own logs" impossible by construction,
instead of relying on every caller remembering not to log sensitive values.
"""

import json
import logging

logger = logging.getLogger("leakguard")


def log_request(
    *,
    request_id: str,
    status: int,
    stream: bool,
    upstream_ms: float | None,
    total_ms: float,
) -> None:
    record = {
        "event": "request",
        "request_id": request_id,
        "status": status,
        "stream": stream,
        "upstream_ms": None if upstream_ms is None else round(upstream_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    logger.info(json.dumps(record))
