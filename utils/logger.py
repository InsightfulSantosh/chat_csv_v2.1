import logging
import sys
import os

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup and configure logger with the specified name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if not logger.level or logger.level == logging.NOTSET:
        logger.setLevel(logging.DEBUG)
    return logger
