import os
import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_console_stream_safe() -> None:
    """Prevents UnicodeEncodeError log noise on Windows consoles (cp1252) when messages contain symbols."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def get_logger(name: str = "predictive_maintenance", log_level: str = None, log_file: str = "logs/app.log") -> logging.Logger:
    """
    Returns a configured Python logger instance with console and file output.
    Relative log paths are anchored to the project root so logs land in the same
    place regardless of the directory the process was started from.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    _make_console_stream_safe()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = _PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
