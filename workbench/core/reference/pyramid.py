"""Min/max pyramid: draw millions of samples on a ~2000-pixel lane without losing a single spike.

A 10 min run is 12 000 000 samples per channel, but a lane is about 1920 pixels wide, so each pixel covers
~6250 samples. Averaging them (or keeping every 6250th sample) would erase a one-sample spike, which is exactly
what chatter onset and DAQ glitches look like. Instead each pixel shows the min and the max of its samples, drawn as
one vertical stroke. That is lossless for what the eye sees: every sample lies inside some pixel's stroke.
This is "min/max decimation", which pyqtgraph calls 'peak' downsampling; Jugel et al. (2014) prove that per pixel
column the first, last, min and max values (M4) are enough for an error-free line chart
(PVLDB 7(10), doi:10.14778/2732951.2732953). We keep min and max only, because we draw vertical strokes.

Scanning 12 M samples on every zoom or pan is too slow (~10-30 ms in numpy, the frame budget is 16 ms), so the
mins and maxes of fixed blocks are precomputed once, like mipmaps in graphics (Williams 1983, "Pyramidal
parametrics", SIGGRAPH '83, Computer Graphics 17(3) 1-11):
    level 0 = x itself; level k = min and max of every block of `block**k` samples (16, 256, 4096, ...).
Each level is `block` times shorter, so all levels together cost 1/(block-1) of x for the mins and the same for the
maxes: 2/15 = 13 % extra memory (measured 13.3 %).

A pixel is then split into whole blocks, coarsest first (a segment-tree style range query): at most
2*(block-1) entries per level, so the cost per pixel grows with log(samples), not with samples. The answer is
exact (identical to scanning every sample), which test_pyramid.py checks against brute force.

Part 1b replaces this class with C++ behind the same names; this file stays as the reference it is tested against.
"""
from __future__ import annotations

import numpy as np


class MinMaxPyramid:
    """Precomputed block minima and maxima of one channel.

    x      1-D samples (not copied; level 0 is a view of it). NaN means "no data" and is skipped (fmin/fmax).
    block  samples per block at level 1; level k blocks hold block**k samples. 16 keeps levels few (6 for 12 M)
           and the edge work per pixel small (< 2*15 per level).
    """

    def __init__(self, x: np.ndarray, block: int = 16):
        if x.ndim != 1:
            raise ValueError(f"expected 1-D samples, got shape {x.shape}")
        if block < 2:
            raise ValueError("block must be at least 2")
        self.x = x
        self.block = block
        self.mins = [x]
        self.maxs = [x]
        while len(self.mins[-1]) > 1:
            starts = np.arange(0, len(self.mins[-1]), block)      # the last block may be partial
            self.mins.append(np.fmin.reduceat(self.mins[-1], starts))
            self.maxs.append(np.fmax.reduceat(self.maxs[-1], starts))

    @property
    def levels(self) -> int:
        return len(self.mins)

    def envelope(self, i0: int, i1: int, n_px: int) -> tuple[np.ndarray, np.ndarray]:
        """Min and max of samples [i0, i1) for each of n_px pixels.

        Pixel p covers samples [e_p, e_p+1) with e_p = i0 + p*(i1-i0)//n_px. When there are no more samples than
        pixels (zoomed right in), there is nothing to reduce and the raw samples come back as both min and max.
        """
        n = len(self.x)
        if not (0 <= i0 < i1 <= n) or n_px < 1:
            raise ValueError(f"need 0 <= i0 < i1 <= {n} and n_px >= 1, got i0={i0}, i1={i1}, n_px={n_px}")
        if i1 - i0 <= n_px:
            return self.x[i0:i1], self.x[i0:i1]

        e = i0 + np.arange(n_px + 1, dtype=np.int64) * (i1 - i0) // n_px
        a, b = e[:-1], e[1:]
        lo = np.full(n_px, np.nan, dtype=self.x.dtype)
        hi = np.full(n_px, np.nan, dtype=self.x.dtype)

        # Coarsest useful level: the biggest block that still fits in the narrowest pixel.
        width = (i1 - i0) // n_px
        top = 0
        while top + 1 < self.levels and self.block ** (top + 1) <= width:
            top += 1

        # Walk from coarse to fine. At level k a pixel's aligned part is blocks [L, R) (in level-k block units).
        # The part already covered by level k+1 is cut out, so each sample is used exactly once.
        prev_l = prev_r = None
        for k in range(top, -1, -1):
            size = self.block ** k
            L = -(-a // size)                     # ceil(a / size)
            R = b // size
            valid = L < R
            if prev_l is None:
                parts = [(L, R)]
            else:
                inner_l = np.where(prev_valid, prev_l * self.block, R)    # level k+1 blocks in level-k units
                inner_r = np.where(prev_valid, prev_r * self.block, R)
                parts = [(L, np.minimum(inner_l, R)), (inner_r, R)]
            for s, t in parts:
                s = np.where(valid, s, 0)
                t = np.where(valid, t, 0)
                seg_lo, seg_hi = self._segments(k, s, t)
                lo = np.fmin(lo, seg_lo)
                hi = np.fmax(hi, seg_hi)
            prev_l, prev_r, prev_valid = L, R, valid
        return lo, hi

    def _segments(self, k: int, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """fmin/fmax of level k over [s_p, t_p) for every pixel at once; NaN where the range is empty.

        Each range is short (a few blocks), so a (pixels x width) index table is gathered and reduced in one go.
        Indices past t are clamped to t-1 (a repeat of an element already in the range, harmless for min/max).
        """
        mins, maxs = self.mins[k], self.maxs[k]
        out_lo = np.full(len(s), np.nan, dtype=mins.dtype)
        out_hi = np.full(len(s), np.nan, dtype=maxs.dtype)
        has = t > s
        if not has.any():
            return out_lo, out_hi
        s, t = s[has], t[has]
        width = int((t - s).max())
        idx = np.minimum(s[:, None] + np.arange(width), (t - 1)[:, None])
        out_lo[has] = np.fmin.reduce(mins[idx], axis=1)
        out_hi[has] = np.fmax.reduce(maxs[idx], axis=1)
        return out_lo, out_hi
