"""The Steps engine: a chain of operations on a recording, where the original is never changed.

The opened file is the root step. Every other step is (op, params, input step) and its output is a new Recording
with the same fs, t0, tables and events, and new channel arrays. Outputs are computed the first time something
asks for them and then cached; changing a step's parameters clears its cache and that of every step built on it
(the SolidWorks feature-tree idea from the spec, §5). Ops live one per file in core/ops/ and register themselves
with @op, exactly like file formats with @reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from workbench.core import registry
from workbench.core.recording import Channel, Recording


@dataclass
class OpInfo:
    name: str
    title: str
    description: str
    defaults: dict
    fn: Callable[..., Recording]


OPS: dict[str, OpInfo] = {}


def op(name: str, title: str, description: str = "", **defaults):
    """Register fn(rec, **params) -> Recording as an op; keyword arguments are the parameters and their defaults."""
    def register(fn):
        OPS[name] = OpInfo(name, title, description, dict(defaults), fn)
        return fn
    return register


def load_ops(package: str = "workbench.core.ops") -> list[str]:
    """Import every op module (same loader and problem reporting as the file formats)."""
    return registry.load_formats(package)


def map_channels(rec: Recording, fn: Callable[[np.ndarray], np.ndarray], step: str = "") -> Recording:
    """A new Recording with fn applied to each channel's data (kept float32), everything else carried over."""
    channels = [Channel(c.name, c.unit, np.asarray(fn(c.data), dtype=np.float32)) for c in rec.channels]
    meta = {**rec.meta, "step": step} if step else dict(rec.meta)
    return replace(rec, channels=channels, meta=meta)


@dataclass(eq=False)
class Step:
    op: str | None
    params: dict = field(default_factory=dict)
    input: Step | None = None
    _rec: Recording | None = None                     # only the root holds the file's Recording
    _cache: Recording | None = field(default=None, repr=False)
    _children: list[Step] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.op is None:
            return
        if self.op not in OPS:
            raise KeyError(f"no op '{self.op}'; known: {', '.join(sorted(OPS)) or 'none loaded'}")
        info = OPS[self.op]
        unknown = set(self.params) - set(info.defaults)
        if unknown:
            raise TypeError(f"{info.title} has no parameter {', '.join(sorted(unknown))}; "
                            f"it takes {', '.join(info.defaults) or 'none'}")
        self.params = {**info.defaults, **self.params}
        if self.input is not None:
            self.input._children.append(self)

    @classmethod
    def root(cls, rec: Recording) -> Step:
        return cls(None, _rec=rec)

    @property
    def title(self) -> str:
        return "File" if self.op is None else OPS[self.op].title

    @property
    def is_cached(self) -> bool:
        return self.op is None or self._cache is not None

    def output(self) -> Recording:
        if self.op is None:
            return self._rec
        if self._cache is None:
            self._cache = OPS[self.op].fn(self.input.output(), **self.params)
        return self._cache

    def set_params(self, **params) -> None:
        bad = set(params) - set(OPS[self.op].defaults)
        if bad:
            raise TypeError(f"{self.title} has no parameter {', '.join(sorted(bad))}")
        self.params.update(params)
        self.invalidate()

    def invalidate(self) -> None:
        self._cache = None
        for child in self._children:
            child.invalidate()
