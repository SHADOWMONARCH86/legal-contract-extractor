"""
Centralised logging configuration.
Outputs structured logs (JSON-friendly format in production, human-readable in dev).
"""

import logging
import os
import sys

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" | "json"

_FMT_TEXT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    if _LOG_FORMAT == "json":
        try:
            import pythonjsonlogger.jsonlogger as jsonlogger  # type: ignore

            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt=_DATE_FMT,
            )
        except ImportError:
            formatter = logging.Formatter(_FMT_TEXT, datefmt=_DATE_FMT)
    else:
        formatter = logging.Formatter(_FMT_TEXT, datefmt=_DATE_FMT)
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False
    return logger
