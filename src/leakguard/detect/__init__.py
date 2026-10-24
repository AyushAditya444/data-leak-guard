"""Detection: find sensitive data in text and return it as Spans."""

from leakguard.detect.engine import build_detector, detect_rules

__all__ = ["build_detector", "detect_rules"]
