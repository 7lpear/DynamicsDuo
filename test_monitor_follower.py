"""Monitor and telemetry checks using a fake bus with no write methods."""
from contextlib import redirect_stdout, redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from follower_feedback import JOINTS, read_diagnostics
from monitor_follower import main, monitor


class Bus:
    is_calibrated = True
    is_connected = False

    def __init__(self):
        self.disconnect_options = None

    def connect(self):
        self.is_connected = True

    def disconnect(self, **kwargs):
        self.disconnect_options = kwargs
        self.is_connected = False

    def sync_read(self, register, **kwargs):
        return dict.fromkeys(JOINTS,120 if register=='Present_Voltage' else 0)


class Tests(unittest.TestCase):
    def test_diagnostics_units_and_missing_or_nonfinite_feedback(self):
        self.assertEqual(read_diagnostics(Bus())['elbow_flex']['voltage_v'],12.)
        for values, message in (({},'Incomplete'),(dict.fromkeys(JOINTS,float('nan')),'Nonfinite')):
            bus = SimpleNamespace(sync_read=Mock(return_value=values))
            with self.assertRaisesRegex(RuntimeError,message):
                read_diagnostics(bus)

    def test_monitor_preserves_torque_and_only_reads(self):
        bus,viewer = Bus(),Mock()
        robot = SimpleNamespace(bus=bus,calibration={'present':True})
        clock = [0.]
        def sleep(seconds):
            clock[0] += seconds
        with patch('monitor_follower.time.monotonic',side_effect=lambda:clock[0]), \
             patch('monitor_follower.time.sleep',side_effect=sleep), redirect_stdout(io.StringIO()):
            monitor(robot,viewer,duration=.3,fps=10.)
        self.assertEqual(viewer.publish.call_count,3)
        self.assertEqual(bus.disconnect_options,{'disable_torque':False})
        self.assertFalse(bus.is_connected)

    def test_calibration_read_and_viewer_failures_close_bus_without_changing_torque(self):
        for failure in ('calibration','read','viewer'):
            with self.subTest(failure=failure):
                bus,viewer = Bus(),Mock()
                robot = SimpleNamespace(bus=bus,calibration={} if failure=='calibration' else {'present':True})
                if failure=='read':
                    bus.sync_read = Mock(side_effect=ConnectionError('packet missing'))
                if failure=='viewer':
                    viewer.publish.side_effect = RuntimeError('viewer failed')
                with redirect_stdout(io.StringIO()), self.assertRaises((RuntimeError,ConnectionError)):
                    monitor(robot,viewer,duration=.1)
                self.assertEqual(bus.disconnect_options,{'disable_torque':False})

    def test_invalid_cli_values_fail_before_opening_viewer_or_hardware(self):
        for args in (['--duration','nan'],['--duration','0'],['--fps','31'],['--viewer-port','0']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), patch('monitor_follower.FollowerViewer') as viewer:
                main(args)
            viewer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
