"""Set the NI 9263 outputs by hand: type a voltage, it is written at once.

    .venv/bin/python -m rig.ao_set                  # finds the NI 9263 itself
    .venv/bin/python -m rig.ao_set --readback ai3   # also measure ao1 back on the NI 9252 (wire AO1 -> AI3, COM -> COM)

At the prompt:
    4          feed (ao1) = 4 V
    f 4.5      feed (ao1) = 4.5 V
    s 1.429    spindle (ao0) = 1.429 V
    r          read back the voltage again (needs --readback)
    q          quit: feed back to 5 V (100 %), spindle left as it is, channels released

Every value is clamped to 0..10 V. Nothing is written to the spindle unless you type `s`.
Do not run this while scope.py has Start output on (both would want the NI 9263).
"""
import argparse
import sys

import numpy as np

from rig import ni_hw

V_MIN, V_MAX = 0.0, 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--module', default='auto', help='NI 9263 module name (auto: found at start)')
    ap.add_argument('--readback', help='spare input on the NI 9252 wired to AO1, e.g. ai3')
    ap.add_argument('--feed-exit', type=float, default=5.0, help='feed voltage written on quit (default 5 V = 100 %%)')
    a = ap.parse_args()

    import nidaqmx
    found = ni_hw.detect()
    out_mod = found['output']['name'] if a.module == 'auto' and found['output'] else a.module
    if out_mod == 'auto':
        sys.exit('No real NI 9263 found. Is the chassis plugged in?')
    rb = None
    if a.readback:
        if not found['input']:
            sys.exit('No input module found for --readback')
        rb = nidaqmx.Task()
        rb.ai_channels.add_ai_voltage_chan(f"{found['input']['name']}/{a.readback}", min_val=-10, max_val=10)

    # until `s` is typed only ao1 is open (one-channel task), so the spindle input is never touched
    task = feed_task = None
    spindle = None
    feed = a.feed_exit
    print(f'{out_mod}: ao0 = spindle, ao1 = feed. Type a number for the feed, "s <V>" for the spindle, q to quit.')

    def put_feed(v):
        nonlocal feed_task, task
        if spindle is None:
            if task is not None:
                task.close()
                task = None
            if feed_task is None:
                feed_task = nidaqmx.Task()
                feed_task.ao_channels.add_ao_voltage_chan(f'{out_mod}/ao1', min_val=V_MIN, max_val=V_MAX)
                feed_task.start()
            feed_task.write(v)
        else:
            task.write([spindle, v])

    def put_spindle(v):
        nonlocal feed_task, task
        if feed_task is not None:
            feed_task.close()
            feed_task = None
        if task is None:
            task = nidaqmx.Task()
            task.ao_channels.add_ao_voltage_chan(f'{out_mod}/ao0', min_val=V_MIN, max_val=V_MAX)
            task.ao_channels.add_ao_voltage_chan(f'{out_mod}/ao1', min_val=V_MIN, max_val=V_MAX)
            task.start()
        task.write([v, feed])

    def readback():
        if rb is None:
            return ''
        v = np.mean(rb.read(number_of_samples_per_channel=200))
        return f'   measured on {a.readback}: {v:.4f} V'

    try:
        while True:
            try:
                line = input('> ').strip().lower()
            except EOFError:
                break
            if not line:
                continue
            if line in ('q', 'quit', 'exit'):
                break
            if line == 'r':
                print(readback() or '   no --readback channel')
                continue
            parts = line.split()
            try:
                if parts[0] in ('s', 'f'):
                    which, v = parts[0], float(parts[1])
                else:
                    which, v = 'f', float(parts[0])
            except (ValueError, IndexError):
                print('   type a number (feed), "f 4", "s 1.429", r or q')
                continue
            v_set = float(np.clip(v, V_MIN, V_MAX))
            note = '' if v_set == v else f' (clamped from {v})'
            if which == 'f':
                feed = v_set
                put_feed(feed)
                print(f'   feed ao1 = {feed:.3f} V{note}{readback()}')
            else:
                spindle = v_set
                put_spindle(spindle)
                print(f'   spindle ao0 = {spindle:.3f} V ({spindle * 14000:.0f} rpm at 14000 rpm/V){note}')
    except KeyboardInterrupt:
        print()
    finally:
        feed = float(np.clip(a.feed_exit, V_MIN, V_MAX))
        put_feed(feed)
        print(f'quit: feed ao1 = {feed:.3f} V' + ('' if spindle is None else f', spindle ao0 left at {spindle:.3f} V')
              + ', channels released')
        for t in (task, feed_task, rb):
            if t is not None:
                t.close()


if __name__ == '__main__':
    main()
