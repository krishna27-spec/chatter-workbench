"""A fixed-size ring buffer of the newest samples, shared between the acquisition thread (writes) and the GUI
(reads). One lock; writes and reads copy, so neither side ever sees a half-written block. (Part 1b may replace it
with C++ behind the same methods.)"""
from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, n_ch: int, capacity: int):
        self.buf = np.zeros((n_ch, capacity), dtype=np.float32)
        self.capacity = capacity
        self.total = 0                      # samples written since the start (per channel)
        self._lock = threading.Lock()

    def write(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float32)
        n_new = block.shape[1]
        if n_new > self.capacity:                   # only the newest capacity samples can be kept
            block = block[:, -self.capacity:]
        k = block.shape[1]
        with self._lock:
            i = (self.total + n_new - k) % self.capacity
            first = min(k, self.capacity - i)
            self.buf[:, i:i + first] = block[:, :first]
            self.buf[:, :k - first] = block[:, first:]
            self.total += n_new

    def latest(self, n: int) -> tuple[np.ndarray, int]:
        """(the newest min(n, available) samples in time order, index of the first one since the start)."""
        with self._lock:
            n = min(n, self.total, self.capacity)
            end = self.total % self.capacity
            idx = (np.arange(end - n, end)) % self.capacity
            return self.buf[:, idx].copy(), self.total - n
