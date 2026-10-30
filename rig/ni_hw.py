"""Find the NI modules at start-up, so nobody has to type slot names.

The cDAQ chassis names its modules by slot (cDAQ2Mod1, cDAQ2Mod2, ...); move a module to another
slot, or plug in another chassis, and the names change. detect() asks the NI-DAQmx driver what is
attached and picks
  input   the accelerometer module: NI 9252, else NI 9775, else NI 9222, else any module with
          analog inputs; channels ai0..ai2 (X, Y, Z). The accelerometers are on the NI 9252 now
          (2026-10-23); reorder INPUT_PREF if they move.
  output  the analog output module: NI 9263, else any module with analog outputs;
          ao0 = spindle speed, ao1 = feed override
Simulated devices (NI MAX can hold them; they answer like real modules but nothing reaches a pin)
are never picked automatically; name one explicitly (--device cDAQ1Mod4) to use it on purpose.

Used by scope.py and realtime.py: --device auto, --ao-channel auto, --feed-channel auto.
Run on its own to see what would be used:  .venv/bin/python -m rig.ni_hw
"""
INPUT_PREF = ['NI 9252', 'NI 9775', 'NI 9222']
OUTPUT_PREF = ['NI 9263']


def modules():
    """Every C Series module the driver knows: list of dicts (name, product, chassis, slot, simulated, n_ai, n_ao)."""
    from nidaqmx.system import System
    out = []
    for d in System.local().devices:
        try:
            slot = d.compact_daq_slot_num
            chassis = d.compact_daq_chassis_device.name
        except Exception:                       # the chassis itself, or a non-cDAQ device
            continue
        out.append(dict(name=d.name, product=d.product_type, chassis=chassis, slot=slot,
                        simulated=bool(d.is_simulated),
                        n_ai=len(d.ai_physical_chans), n_ao=len(d.ao_physical_chans)))
    return out


def _pick(mods, prefs, key):
    cands = [m for m in mods if m[key] > 0 and not m['simulated']]
    rank = lambda m: (prefs.index(m['product']) if m['product'] in prefs else len(prefs), m['chassis'], m['slot'])
    return min(cands, key=rank) if cands else None


def detect():
    """{'input': module or None, 'output': module or None, 'all': [modules]}."""
    mods = modules()
    return dict(input=_pick(mods, INPUT_PREF, 'n_ai'), output=_pick(mods, OUTPUT_PREF, 'n_ao'), all=mods)


def lookup(name, mods=None):
    """The module dict for a device name such as cDAQ2Mod1 (None if the driver does not list it)."""
    mods = modules() if mods is None else mods
    return next((m for m in mods if m['name'] == name), None)


def resolve(a, log=print):
    """Fill in a.device / a.ao_channel / a.feed_channel where they say 'auto'. Returns a summary dict
    (what is read and written where) for printing and for the web page. Leaves a replay alone."""
    hw = dict(input=None, output=None, all=[], spindle=None, feed=None, error='')
    if getattr(a, 'replay', None):
        return hw
    try:
        found = detect()
    except Exception as e:                       # no NI driver / no hardware
        found = dict(input=None, output=None, all=[])
        hw['error'] = f'NI driver: {type(e).__name__}: {e}'.splitlines()[0]
    hw['all'] = found['all']
    if a.device == 'auto':
        if found['input'] is None:
            raise SystemExit('No real NI module with analog inputs found. Is the chassis plugged in (USB) and '
                             'powered? (.venv/bin/python -m rig.ni_hw lists what the driver sees)')
        a.device = found['input']['name']
    hw['input'] = dict(lookup(a.device, found['all']) or dict(name=a.device), channels=f'{a.device}/{a.channels}')

    ao_mod = None
    for attr in ('ao_channel', 'feed_channel'):
        if getattr(a, attr, None) == 'auto':
            if found['output'] is None:
                raise SystemExit('No real NI module with analog outputs (NI 9263) found for --'
                                 + attr.replace('_', '-') + '. Is the chassis plugged in and powered?')
            setattr(a, attr, f"{found['output']['name']}/{'ao0' if attr == 'ao_channel' else 'ao1'}")
    if getattr(a, 'control', False):
        ao_mod = lookup(a.ao_channel.split('/')[0], found['all'])
        hw['output'] = dict(ao_mod or dict(name=a.ao_channel.split('/')[0]))
        hw['spindle'] = a.ao_channel
        hw['feed'] = a.feed_channel
    for m in (hw['input'], hw['output']):
        if m and m.get('simulated'):
            log(f"WARNING: {m['name']} ({m.get('product')}) is a SIMULATED device: nothing reaches a real pin.")
    log(describe(hw))
    return hw


def describe(hw):
    def mod(m):
        if not m:
            return '-'
        where = f" in {m['chassis']} slot {m['slot']}" if 'slot' in m else ''
        return f"{m.get('product', '?')}{where}{' (SIMULATED)' if m.get('simulated') else ''}"
    s = f"Input: {mod(hw['input'])}, reading {hw['input']['channels']}" if hw['input'] else 'Input: replayed file'
    if hw['output']:
        s += f"\nOutput: {mod(hw['output'])}, spindle on {hw['spindle']}, " \
             f"feed {'on ' + hw['feed'] if hw['feed'] else 'not sent'}"
    return s


if __name__ == '__main__':
    f = detect()
    for m in f['all']:
        tag = ' <- input' if m is f['input'] else ' <- output' if m is f['output'] else ''
        print(f"{m['name']:12s} {m['product']:8s} {m['chassis']} slot {m['slot']}  ai {m['n_ai']}  ao {m['n_ao']}"
              f"{'  simulated' if m['simulated'] else ''}{tag}")
