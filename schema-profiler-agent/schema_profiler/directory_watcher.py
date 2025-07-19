import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

logger = logging.getLogger(__name__)

class TdsxFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".tdsx"):
            logger.info(f"New TDSX file detected: {event.src_path}")
            self.callback(Path(event.src_path))

class DirectoryWatcher:
    def __init__(self, directory_to_watch, callback):
        self.directory_to_watch = directory_to_watch
        self.callback = callback
        self.observer = Observer()

    def run(self):
        event_handler = TdsxFileHandler(self.callback)
        self.observer.schedule(event_handler, self.directory_to_watch, recursive=False)
        self.observer.start()
        logger.info(f"Started watching directory: {self.directory_to_watch}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
