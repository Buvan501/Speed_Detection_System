import os
import csv
from threading import Lock

class CSVLogger:
    def __init__(self, path='exports/logs/speed_log.csv'):
        self.path = path
        self.lock = Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp','id','speed_kmph','bbox'])
                writer.writeheader()

    def log(self, row: dict):
        with self.lock:
            with open(self.path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp','id','speed_kmph','bbox'])
                writer.writerow(row)
