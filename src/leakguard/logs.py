"""One JSON log line per request — metadata only.

The functions accept only named metadata fields, never message text or headers.
That makes "the guard leaks data into its own logs" impossible by construction,
instead of relying on every caller remembering not to log sensitive values.
"""

import json
import logging
from collections.abc import Mapping

logger = logging.getLogger("leakguard")


def log_request(
    *,
    request_id: str,
    status: int,
    stream: bool,
    upstream_ms: float | None,
    total_ms: float,
    findings: Mapping[str, int] | None = None,  # e.g. {"PHONE:mask": 2}
    blocked: bool = False,
    detection_ms: float | None = None,
) -> None:
    record = {
        "event": "request",
        "request_id": request_id,
        "status": status,
        "stream": stream,
        "blocked": blocked,
        "findings": dict(findings or {}),
        "detection_ms": None if detection_ms is None else round(detection_ms, 2),
        "upstream_ms": None if upstream_ms is None else round(upstream_ms, 1),
        "total_ms": round(total_ms, 1),
    }
    logger.info(json.dumps(record))


def log_ner_disabled() -> None:
    logger.warning(json.dumps({"event": "ner_disabled",
                               "detail": "LEAKGUARD_NER=off: person names are not detected"}))


def log_detection_failed(*, request_id: str, failure_mode: str) -> None:
    logger.warning(json.dumps(
        {"event": "detection_failed", "request_id": request_id, "failure_mode": str(failure_mode)}
    ))
