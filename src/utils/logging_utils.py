import csv
import os

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class SimpleCSVLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._init = False

    def append(self, row: dict):
        write_header = not os.path.isfile(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

