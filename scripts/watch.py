"""Watch the ./inbox/ directory and auto-ingest any new .doc/.docx files."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from law_rag import config, pipeline

log = logging.getLogger("watch")

VALID_EXT = {".doc", ".docx"}


class IngestHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        if event.is_directory:
            return
        self._queue(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        self._queue(event.dest_path)

    def _queue(self, p: str) -> None:
        path = Path(p)
        if path.suffix.lower() not in VALID_EXT:
            return
        if path.name.startswith("."):  # macOS metadata files
            return
        self._pending[str(path)] = time.time()

    def drain(self) -> None:
        now = time.time()
        ready = [p for p, t in self._pending.items() if now - t > 2.0]
        for p in ready:
            self._pending.pop(p, None)
            path = Path(p)
            if not path.exists():
                continue
            try:
                log.info("Auto-ingesting %s", path.name)
                result = pipeline.ingest_file(path)
                log.info("  done: %d Khoản", result["n_clauses"])
            except Exception as e:  # noqa: BLE001
                log.exception("Failed to ingest %s: %s", path, e)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = IngestHandler()
    obs = Observer()
    obs.schedule(handler, str(config.INBOX_DIR), recursive=False)
    obs.start()
    log.info("Watching %s — drop .doc/.docx files to ingest. Ctrl-C to stop.", config.INBOX_DIR)

    # Pick up files already present at startup
    for f in config.INBOX_DIR.iterdir():
        if f.suffix.lower() in VALID_EXT and not f.name.startswith("."):
            handler._queue(str(f))

    try:
        while True:
            time.sleep(1.0)
            handler.drain()
    except KeyboardInterrupt:
        log.info("Stopping watcher.")
    finally:
        obs.stop()
        obs.join()


if __name__ == "__main__":
    main()
