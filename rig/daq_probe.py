"""See what NI hardware is attached, and read any analog input channels with chosen settings.

List every chassis and module with what it can do (channels, ranges, rates, terminal
configurations, measurement types):
    .venv/bin/python -m rig.daq_probe

Read channels for a few seconds, print per-channel statistics, optionally save an .lvm:
    .venv/bin/python -m rig.daq_probe --read cDAQ1Mod3/ai0:2 --fs 20000 --range 10 --seconds 5
    .venv/bin/python -m rig.daq_probe --read cDAQ1Mod1/ai0,cDAQ1Mod1/ai3 --term DIFF --out test.lvm
    .venv/bin/python -m rig.daq_probe --read cDAQ1Mod2/ai0 --iepe 0.002 --coupling AC     (IEPE modules only)

The rate the module actually runs at can differ from the one asked for (delta-sigma
modules only support certain rates); the script prints the real one.
"""
import argparse

import numpy as np


def show(label, get):
    try:
        v = get()
    except Exception as e:                      # property not supported or device absent
        v = f'n/a ({type(e).__name__})'
    print(f'    {label:24s} {v}')


def list_hardware():
    from nidaqmx.system import System
    s = System.local()
    print('NI-DAQmx driver', s.driver_version)
    devs = list(s.devices)
    if not devs:
        print('no devices known to the driver')
    for d in devs:
        print(f'\n{d.name}')
        show('product', lambda: d.product_type)
        show('serial', lambda: hex(d.serial_num))
        show('slot', lambda: d.compact_daq_slot_num)
        show('simulated', lambda: d.is_simulated)
        show('AI channels', lambda: d.ai_physical_chans.channel_names)
        show('AI measurement types', lambda: [m.name for m in d.ai_meas_types])
        show('AI ranges (V)', lambda: d.ai_voltage_rngs)
        show('AI max rate 1 ch (S/s)', lambda: d.ai_max_single_chan_rate)
        show('AI max rate multi (S/s)', lambda: d.ai_max_multi_chan_rate)
        show('AI simultaneous', lambda: d.ai_simultaneous_sampling_supported)
        show('AI couplings', lambda: [c.name for c in d.ai_couplings])
        show('terminal configs ch0', lambda: [t.name for t in d.ai_physical_chans[0].ai_term_cfgs])
        show('AO channels', lambda: d.ao_physical_chans.channel_names)


def read(chans, fs, vrange, seconds, term, iepe, coupling, out):
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, Coupling, ExcitationSource, TerminalConfiguration
    with nidaqmx.Task() as task:
        kw = dict(min_val=-vrange, max_val=vrange)
        if term:
            kw['terminal_config'] = TerminalConfiguration[term]
        ch = task.ai_channels.add_ai_voltage_chan(chans, **kw)
        if iepe:                                  # constant-current excitation for an IEPE sensor
            ch.ai_excit_src = ExcitationSource.INTERNAL
            ch.ai_excit_val = iepe
        if coupling:
            ch.ai_coupling = Coupling[coupling]
        n = int(fs * seconds)
        task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=n)
        real_fs = task.timing.samp_clk_rate
        print(f'asked {fs:g} S/s, module runs at {real_fs:g} S/s; reading {seconds} s')
        data = np.atleast_2d(np.array(task.read(number_of_samples_per_channel=n, timeout=seconds + 10)))
        names = [c.name for c in task.ai_channels]
    print(f'{"channel":22s} {"mean V":>10s} {"std mV":>9s} {"min V":>9s} {"max V":>9s}  clipped')
    for name, v in zip(names, data):
        clip = np.mean(np.abs(v) >= 0.999 * vrange)
        print(f'{name:22s} {v.mean():10.5f} {1000 * v.std():9.3f} {v.min():9.4f} {v.max():9.4f}  {clip:.1%}')
    if out:
        t = np.arange(data.shape[1]) / real_fs
        np.savetxt(out, np.column_stack([t, data.T]), fmt='%.7g', delimiter='\t')
        print('saved', out, '(time, then one column per channel)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--read', help='physical channels, e.g. cDAQ1Mod3/ai0:2')
    ap.add_argument('--fs', type=float, default=20_000)
    ap.add_argument('--range', type=float, default=10.0, help='+/- volts; pick the smallest that fits the signal')
    ap.add_argument('--seconds', type=float, default=2.0)
    ap.add_argument('--term', choices=['DEFAULT', 'RSE', 'NRSE', 'DIFF', 'PSEUDO_DIFF'], help='terminal configuration')
    ap.add_argument('--iepe', type=float, help='IEPE excitation current in A, e.g. 0.002 (only IEPE modules)')
    ap.add_argument('--coupling', choices=['AC', 'DC', 'GND'])
    ap.add_argument('--out', help='save the samples as a tab-separated .lvm-style text file')
    a = ap.parse_args()
    if a.read:
        read(a.read, a.fs, a.range, a.seconds, a.term, a.iepe, a.coupling, a.out)
    else:
        list_hardware()


if __name__ == '__main__':
    main()
