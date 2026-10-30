"""Every NI module the driver knows, real and simulated, with its pins; plus the suggestions ni_hw.py used to make.

The cDAQ chassis names modules by slot (cDAQ2Mod1, …), so names change when a module moves. The Live workspace lists
what is attached now and the user picks; suggestions only pre-fill the choice and never pick a simulated module
(NI MAX can hold simulated devices: they answer like real ones, but nothing reaches a pin).
"""
from __future__ import annotations

from dataclasses import dataclass, field

INPUT_PREF = ("NI 9252", "NI 9775", "NI 9222")       # the accelerometers are on the NI 9252 since 2026-10-23
OUTPUT_PREF = ("NI 9263",)


@dataclass
class Module:
    name: str                 # e.g. cDAQ2Mod1
    product: str              # e.g. NI 9252
    chassis: str
    slot: int
    simulated: bool
    ai: list[str] = field(default_factory=list)     # physical channel names, e.g. cDAQ2Mod1/ai0
    ao: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.name}  {self.product}" + ("  (simulated)" if self.simulated else "")


def list_modules() -> tuple[list[Module], str | None]:
    """(modules, problem). No driver or an error → ([], a sentence saying why); never raises."""
    try:
        from nidaqmx.system import System
    except ImportError:
        return [], "NI-DAQmx (Python package nidaqmx) is not installed"
    try:
        out = []
        for d in System.local().devices:
            try:
                slot = d.compact_daq_slot_num
                chassis = d.compact_daq_chassis_device.name
            except Exception:                       # the chassis itself, or a non-cDAQ device
                continue
            out.append(Module(d.name, d.product_type, chassis, slot, bool(d.is_simulated),
                              list(d.ai_physical_chans.channel_names), list(d.ao_physical_chans.channel_names)))
        return sorted(out, key=lambda m: (m.chassis, m.slot)), None
    except Exception as e:                          # driver not running, USB unplugged mid-query, …
        return [], f"NI-DAQmx driver: {type(e).__name__}: {e}".splitlines()[0]


def _suggest(mods: list[Module], prefs: tuple[str, ...], kind: str) -> Module | None:
    cands = [m for m in mods if getattr(m, kind) and not m.simulated]
    rank = lambda m: (prefs.index(m.product) if m.product in prefs else len(prefs), m.chassis, m.slot)
    return min(cands, key=rank) if cands else None


def suggest_input(mods: list[Module]) -> Module | None:
    return _suggest(mods, INPUT_PREF, "ai")


def suggest_output(mods: list[Module]) -> Module | None:
    return _suggest(mods, OUTPUT_PREF, "ao")
