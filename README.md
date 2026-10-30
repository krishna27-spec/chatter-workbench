# Chatter Workbench

Desktop software for chatter detection and spindle speed control in micromilling. It opens accelerometer recordings
(`.lvm`, `.csv`), shows time, spectrum and spectrogram views, and runs a live rig through NI hardware: three trained models
vote on no cutting, stable or chatter every 25.6 ms, and a Liao & Young (1996) rule computes a new spindle speed.

## Download

**Windows:** get `ChatterWorkbench-Setup-<version>.exe` from [Releases](../../releases/latest) and run it.
The installer is not code-signed, so Windows may warn about an unknown publisher: choose *More info*, then *Run anyway*.
Live use needs the free NI-DAQmx driver. Opening recordings does not.

**From source (Python 3.12):**

```
git clone https://github.com/krishna27-spec/chatter-workbench.git
cd chatter-workbench
python -m venv .venv
.venv/bin/pip install -r packaging/requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python -m workbench
```

## What is here

| Folder | Contents |
|---|---|
| `workbench/` | The desktop app: `app/` (Qt) on top of `core/` (data, formats, maths) |
| `models/` | The three shipped models (extra trees, random forest, RBF SVM) |
| `rig/` | Live tools: scope, realtime loop, speed control, NI hardware, local dashboard |
| `pipeline/` | Signal cleaning, features, labels |
| `training/`, `tools/` | Training and plotting scripts |
| `tests/`, `packaging/` | Tests and the installer build |

Recordings and labels are not part of this repository, so the scripts in `training/` need your own data in `data/`.
Speed control is off by default and has not yet been shown to suppress chatter on a machine.

Tests: `python -m pytest`
