"""Logging configuration."""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int, optional
        Logging level.

    Returns
    -------
    Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
