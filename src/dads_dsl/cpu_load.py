from __future__ import annotations

import math
import multiprocessing as mp
import os
import threading
import time

import psutil


def _burn_cpu(stop_event: mp.Event) -> None:
    value = 0.0
    while not stop_event.is_set():
        value = math.sin(value + 1.0) * math.cos(value + 2.0)


class CpuLoadController:
    def __init__(self, target_percent: float, tolerance: float = 5.0, interval: float = 0.5, max_workers: int | None = None):
        if target_percent < 0 or target_percent > 100:
            raise ValueError("target_percent must be between 0 and 100.")
        self.target_percent = float(target_percent)
        self.tolerance = float(tolerance)
        self.interval = float(interval)
        self.max_workers = int(max_workers or max(os.cpu_count() or 1, 1))
        self._workers: list[tuple[mp.Process, mp.Event]] = []
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self.samples: list[float] = []

    def __enter__(self) -> "CpuLoadController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self.target_percent <= 0:
            self.samples.append(psutil.cpu_percent(interval=0.0))
            return
        initial = max(1, min(self.max_workers, int(round(self.max_workers * self.target_percent / 100.0))))
        for _ in range(initial):
            self._start_worker()
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=max(self.interval * 2, 1.0))
        while self._workers:
            self._stop_worker()

    def ramp(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def stats(self) -> dict[str, float]:
        if not self.samples:
            sample = psutil.cpu_percent(interval=0.0)
            self.samples.append(sample)
        return {
            "cpu_load_target": self.target_percent,
            "cpu_load_avg": sum(self.samples) / len(self.samples),
            "cpu_load_min": min(self.samples),
            "cpu_load_max": max(self.samples),
        }

    def _start_worker(self) -> None:
        stop_event = mp.Event()
        process = mp.Process(target=_burn_cpu, args=(stop_event,))
        process.daemon = True
        process.start()
        self._workers.append((process, stop_event))

    def _stop_worker(self) -> None:
        process, stop_event = self._workers.pop()
        stop_event.set()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    def _monitor(self) -> None:
        while not self._monitor_stop.is_set():
            sample = psutil.cpu_percent(interval=self.interval)
            self.samples.append(sample)
            lower = self.target_percent - self.tolerance
            upper = self.target_percent + self.tolerance
            if sample < lower and len(self._workers) < self.max_workers:
                self._start_worker()
            elif sample > upper and len(self._workers) > 1:
                self._stop_worker()
