import logging
import sys
import io
from pathlib import Path

# Create the logger
logger = logging.getLogger("qsga")
logger.setLevel(logging.INFO)

# Make sure we don't duplicate handlers if the module is re-imported
if not logger.handlers:
    # Console formatter and handler
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Memory string stream handler to capture all log messages during the run
    log_capture_stream = io.StringIO()
    memory_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    memory_handler = logging.StreamHandler(log_capture_stream)
    memory_handler.setFormatter(memory_formatter)
    memory_handler.setLevel(logging.INFO)
    logger.addHandler(memory_handler)
else:
    # Find existing memory handler if it exists
    log_capture_stream = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h.stream, (type(sys.stdout), type(sys.stderr))):
            if isinstance(h.stream, io.StringIO):
                log_capture_stream = h.stream
                break
    if log_capture_stream is None:
        log_capture_stream = io.StringIO()

def reset_log_capture() -> None:
    """Clear the captured logs in memory."""
    global log_capture_stream
    if log_capture_stream is not None:
        log_capture_stream.seek(0)
        log_capture_stream.truncate(0)

def get_captured_logs() -> str:
    """Retrieve the log messages captured so far."""
    if log_capture_stream is not None:
        return log_capture_stream.getvalue()
    return ""

def configure_file_logging(log_file_path: Path) -> None:
    """Write captured logs to log_file_path and add a FileHandler for future logs."""
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write what's currently in memory
    captured_content = get_captured_logs()
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(captured_content)

    # 2. Reset the captured logs to avoid duplicating them if configure_file_logging is called again
    reset_log_capture()

    # 3. Check if a FileHandler is already active for this path to avoid duplicates
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # 4. Add the new FileHandler
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
