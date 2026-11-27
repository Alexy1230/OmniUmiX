"""Project-local LK reply framing fix; the installed motor package is unmodified.

The observed FTDI byte stream contains bytes before a valid 0x3E reply. Keep
all upstream motor commands/conversions but locate and validate reply frames.
Supports both direct pyserial and the installed PySerialTransport wrapper.
"""
import time

from motor.motor import LkMotor as _PackageLkMotor
from motor.protocol import build_frame, MotorTimeoutError


class LkMotor(_PackageLkMotor):
    # Matches the read-only discovery tool's successful receive deadline.
    # Successful reads return immediately, without waiting for this duration.
    reply_timeout_s = 0.15

    def send_command(self, cmd: int, data=None, expect_reply_len: int = 0) -> bytes:
        payload = [] if data is None else data
        if expect_reply_len <= 0:
            # Preserve upstream behavior for torque/enable/stop/zero commands.
            return super().send_command(cmd, payload, expect_reply_len)

        self.ser.reset_input_buffer()
        self.ser.write(build_frame(cmd, self.motor_id, payload))
        # __getattr__ on PySerialTransport does not forward assignments. Change
        # the actual serial object's timeout, then restore it even on failure.
        serial_port = getattr(self.ser, '_serial', self.ser)
        original_timeout = serial_port.timeout
        deadline = time.monotonic() + self.reply_timeout_s
        buffer = bytearray()
        diagnostic = bytearray()
        try:
            while time.monotonic() < deadline:
                serial_port.timeout = min(0.02, max(0.0, deadline - time.monotonic()))
                chunk = self.ser.read(1)
                if not chunk:
                    continue
                buffer.extend(chunk)
                if len(diagnostic) < 128:
                    diagnostic.extend(chunk[:128 - len(diagnostic)])
                while buffer:
                    if buffer[0] != 0x3E:
                        del buffer[0]
                        continue
                    if len(buffer) < 5:
                        break
                    length = buffer[3]
                    frame_length = 5 + length + (1 if length else 0)
                    if not (
                        buffer[1] == cmd
                        and buffer[2] == self.motor_id
                        and (sum(buffer[:4]) & 0xFF) == buffer[4]
                        and frame_length == expect_reply_len
                    ):
                        del buffer[0]
                        continue
                    if len(buffer) < frame_length:
                        break
                    response = bytes(buffer[:frame_length])
                    if length and (sum(response[5:-1]) & 0xFF) != response[-1]:
                        del buffer[0]
                        continue
                    return response
            raise MotorTimeoutError(
                f'No valid LK reply: cmd=0x{cmd:02X}, motor_id={self.motor_id}, '
                f'expected_length={expect_reply_len}, '
                f'received={bytes(diagnostic).hex(" ") or "<none>"}'
            )
        finally:
            serial_port.timeout = original_timeout
