import csv
import asyncio

class DataStreamer:
    def __init__(self, path="model/data/stream_source.csv", window_size=30, delay=1.0):
        self.path = path
        self.window_size = window_size
        self.delay = delay

        # preload CSV into memory so we don't restart from top every loop
        self.data = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append([
                    float(row["hr"]),
                    float(row["rr"]),
                    float(row["spo2"])
                ])

        if not self.data:
            raise ValueError(f"No data found in {self.path}")

    async def stream(self):
        """Yield sliding windows continuously from preloaded data."""
        i = 0
        while True:
            # rolling last N rows
            window = self.data[max(0, i - self.window_size + 1): i + 1]

            if len(window) == self.window_size:
                yield window  # this keeps your latest window logic intact

            await asyncio.sleep(self.delay)

            i += 1
            if i >= len(self.data):  
                i = 0
