"""List USB serial adapters or identify LK motor IDs using read-only queries.

Protocol reference: OmniUmiX/omniumix/third_party/LK_MOTOR/motor/{protocol,motor}.py.
No enabling, movement, zero reset, ID change or baud-rate change commands are sent.
Requires pyserial, already used by the gripper driver.
"""
import argparse
import math
from pathlib import Path
import sys
import time


def request_frame(command, motor_id):
    if command not in (0x92, 0x9C):
        raise ValueError('Only angle/status read commands are permitted')
    header = bytes((0x3E, command, motor_id, 0))
    return header + bytes((sum(header) & 0xFF,))


def find_reply(buffer, command, motor_id, payload_size):
    """Find a complete, matching, checksummed frame despite noise or USB echo."""
    size = 5 + payload_size + 1
    for offset in range(max(0, len(buffer) - size + 1)):
        frame = bytes(buffer[offset:offset + size])
        if frame[:4] != bytes((0x3E, command, motor_id, payload_size)):
            continue
        if (sum(frame[:4]) & 0xFF) != frame[4]:
            continue
        if (sum(frame[5:-1]) & 0xFF) != frame[-1]:
            continue
        return frame
    return None


def query(port, command, motor_id, timeout):
    port.reset_input_buffer()
    port.write(request_frame(command, motor_id))
    deadline = time.monotonic() + timeout
    data = bytearray()
    payload_size = 8 if command == 0x92 else 7
    while time.monotonic() < deadline:
        data.extend(port.read(min(max(port.in_waiting, 1), 256)))
        reply = find_reply(data, command, motor_id, payload_size)
        if reply is not None:
            return reply, bytes(data)
        if len(data) > 4096:
            break
    return None, bytes(data)


def scan(port, ids, timeout, verbose=False):
    found = []
    for motor_id in ids:
        replies = []
        for command in (0x92, 0x9C):
            reply, received = query(port, command, motor_id, timeout)
            replies.append(reply)
            if verbose:
                print(f'ID={motor_id:2d} cmd=0x{command:02X} '
                      f'{"VALID" if reply else "NO VALID REPLY"}: '
                      f'{received[:128].hex(" ") or "no bytes received"}', flush=True)
            time.sleep(0.01)
        if all(reply is not None for reply in replies):
            print(f'FOUND motor ID {motor_id}: angle and status replies validated', flush=True)
            found.append(motor_id)
        elif any(reply is not None for reply in replies):
            print(f'PARTIAL motor ID {motor_id}: one valid reply; '
                  'address responds but the full feedback check failed', flush=True)
    return found


def parse_ids(value):
    try:
        ids = set()
        for item in value.split(','):
            limits = item.split('-')
            if len(limits) == 1:
                ids.add(int(item))
            elif len(limits) == 2:
                first, last = map(int, limits)
                if first > last:
                    raise ValueError()
                ids.update(range(first, last + 1))
            else:
                raise ValueError()
        if not ids or min(ids) < 1 or max(ids) > 32:
            raise ValueError()
        return sorted(ids)
    except ValueError:
        raise argparse.ArgumentTypeError('Use IDs 1-32, for example 1-32 or 1,2,5')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List adapters without opening any port')
    parser.add_argument('--port', help='Probe only this selected motor adapter, e.g. /dev/ttyUSB1')
    parser.add_argument('--baud', type=int, default=460800, help='Probe baud rate; default matches OmniUmiX')
    parser.add_argument('--ids', type=parse_ids, default=parse_ids('1-32'))
    parser.add_argument('--timeout', type=float, default=0.15, help='Seconds per query, default 0.15')
    parser.add_argument('--verbose', action='store_true', help='Show received bytes for every probe')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.baud <= 0:
        parser.error('--timeout and --baud must be positive and finite')
    import serial
    from serial.tools import list_ports
    if args.list or not args.port:
        ports = list(list_ports.comports())
        if not ports:
            print('No serial adapters detected.')
        for info in sorted(ports, key=lambda p: p.device):
            print(f'{info.device}: {info.description}\n  {info.hwid}')
        for alias in sorted(Path('/dev/serial/by-id').glob('*')):
            print(f'Stable path: {alias} -> {alias.resolve()}')
        if not args.port:
            print('Select the motor adapter, then rerun with --port. No ports were opened.')
            return 0
    print(f'Probing {args.port}, {args.baud} baud, IDs {args.ids}; read commands only.', flush=True)
    # 8 data bits, no parity, 1 stop bit: the existing driver's defaults.
    options = dict(port=args.port, baudrate=args.baud, timeout=min(args.timeout, .02),
                   write_timeout=1, bytesize=serial.EIGHTBITS,
                   parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    if sys.platform.startswith('linux'):
        options['exclusive'] = True
    with serial.Serial(**options) as port:
        found = scan(port, args.ids, args.timeout, args.verbose)
    if not found:
        print('No fully validated motor found at this baud rate. This does not prove '
              'the adapter is wrong: check power, wiring, baud rate and other port users.')
        return 2
    print(f'Confirmed port={args.port}, baud={args.baud}, motor IDs={found}')
    if args.baud != 460800:
        print('This differs from the reference driver default; configure the driver baud rate too.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
