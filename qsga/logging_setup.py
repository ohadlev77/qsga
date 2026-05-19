import logging
import re
import sys
import io
from pathlib import Path

_TIMING_PATTERN = re.compile(r'\[(.+?)\] execution time: ([\d.]+) seconds')

class TimingHandler(logging.Handler):
    """Captures timing log messages emitted by the time_it decorator."""

    def __init__(self):
        super().__init__()
        self._records: list[tuple[float, str, str]] = []  # (duration, qualname, formatted_line)

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        match = _TIMING_PATTERN.search(msg)
        if match:
            duration = float(match.group(2))
            formatted = self.format(record) if self.formatter else msg
            self._records.append((duration, match.group(1), formatted))

    def get_top_n(self, n: int) -> list[tuple[float, str, str]]:
        return sorted(self._records, key=lambda x: x[0], reverse=True)[:n]

    def reset(self) -> None:
        self._records.clear()


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

    # Timing handler to collect execution-time log entries
    timing_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    timing_handler = TimingHandler()
    timing_handler.setFormatter(timing_formatter)
    timing_handler.setLevel(logging.INFO)
    logger.addHandler(timing_handler)
else:
    # Find existing memory handler if it exists
    log_capture_stream = None
    timing_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h.stream, (type(sys.stdout), type(sys.stderr))):
            if isinstance(h.stream, io.StringIO):
                log_capture_stream = h.stream
        if isinstance(h, TimingHandler):
            timing_handler = h
    if log_capture_stream is None:
        log_capture_stream = io.StringIO()
    if timing_handler is None:
        timing_handler = TimingHandler()
        timing_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        timing_handler.setLevel(logging.INFO)
        logger.addHandler(timing_handler)

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

def write_timing_report(report_path: Path, top_n: int = 100) -> None:
    """Write the top-n longest execution times to report_path, sorted longest to shortest."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    top_records = timing_handler.get_top_n(top_n)
    actual_n = len(top_records)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Top {actual_n} longest execution times (sorted longest to shortest)\n")
        f.write("=" * 70 + "\n\n")
        for rank, (_, __, formatted) in enumerate(top_records, 1):
            f.write(f"#{rank:<4} {formatted}\n")


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
