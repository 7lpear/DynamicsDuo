"""Lost ACK must be distinguished from a lost write using exact register readback."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from calibration_io import verified_calibration_writes


class Bus:
    def __init__(self, mode='ok'):
        self.value = 0
        self.mode = mode
        self.writes = []

    def read(self, *args, **kwargs):
        return self.value

    def write(self, register, motor, value, **kwargs):
        self.writes.append((register, motor, value))
        if self.mode == 'device_error':
            raise RuntimeError('Device reports voltage/protection error')
        if self.mode not in ('not_applied', 'false_ack'):
            self.value = value
        if self.mode in ('lost_ack', 'not_applied'):
            raise ConnectionError('There is no status packet!')


class Tests(unittest.TestCase):
    @patch('calibration_io.time.sleep')
    def test_lost_ack_but_negative_offset_stored_is_verified_once(self, sleep):
        bus = Bus('lost_ack')
        original = bus.write
        output = io.StringIO()
        with redirect_stdout(output), verified_calibration_writes(bus):
            bus.write('Homing_Offset', 'wrist_flex', -1712)
        self.assertEqual(bus.value, -1712)
        self.assertEqual(len(bus.writes), 1)
        self.assertIn('READBACK VERIFIED', output.getvalue())
        self.assertEqual(bus.write, original)

    @patch('calibration_io.time.sleep')
    def test_failed_write_or_wrong_readback_stops_after_bounded_retries(self, sleep):
        for mode in ('not_applied', 'false_ack'):
            bus = Bus(mode)
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify'), verified_calibration_writes(bus):
                bus.write('Homing_Offset', 'wrist_flex', -1712)
            self.assertEqual(len(bus.writes), 3)
            self.assertEqual(bus.value, 0)

    def test_unchanged_register_is_not_rewritten(self):
        bus = Bus()
        with verified_calibration_writes(bus):
            bus.write('Torque_Enable', 'wrist_flex', 0)
        self.assertEqual(bus.writes, [])

    def test_device_reported_error_is_not_silently_retried(self):
        bus = Bus('device_error')
        with self.assertRaisesRegex(RuntimeError, 'protection error'), verified_calibration_writes(bus):
            bus.write('Homing_Offset', 'wrist_flex', -1712)
        self.assertEqual(len(bus.writes), 1)

    def test_calibration_wrapper_blocks_motion_and_torque_enable(self):
        for register, value in (('Goal_Position', 2047), ('Torque_Enable', 1)):
            bus = Bus()
            with self.assertRaisesRegex(RuntimeError, 'cannot enable'), verified_calibration_writes(bus):
                bus.write(register, 'wrist_flex', value)
            self.assertEqual(bus.writes, [])


if __name__ == '__main__':
    unittest.main()
