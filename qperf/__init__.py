"""qperf: deterministic performance model for Quasar binary_ng eltwise (reader -> compute -> writer)."""

ENGINE_VERSION = "0.2.0"
SCHEMA_VERSION = "1"

from .api import analyze  # noqa: E402

__all__ = ["analyze", "ENGINE_VERSION", "SCHEMA_VERSION"]
