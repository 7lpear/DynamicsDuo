"""Recalibration startup must not configure or enable the motors."""

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from calibrate_follower import recalibrate


@dataclass
class Calibration:
    range_min: int = 100
    range_max: int = 3000
    homing_offset: int = 10


class Bus:
    motors = {"shoulder_pan": object()}
    is_connected = False
    is_calibrated = False

    def __init__(self, torque=0):
        self.torque = torque
        self.events = []

    def connect(self):
        self.is_connected = True
        self.events.append("connect")

    def read(self, register, motor, **kwargs):
        assert register == "Torque_Enable"
        return self.torque

    def read_calibration(self):
        return {"shoulder_pan": Calibration()}

    def write(self, *args, **kwargs):
        raise AssertionError('Mock calibration does not issue any motor writes')

    def disconnect(self, disable_torque=True):
        self.events.append(("disconnect", disable_torque))
        self.is_connected = False


class Tests(unittest.TestCase):
    def run_case(self, torque=0, interrupt=False):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            original = {"old": "measured configuration"}
            fpath = root / "follower.json"
            fpath.write_text(json.dumps(original), encoding="utf-8")
            bus = Bus(torque)
            robot = SimpleNamespace(bus=bus, calibration=original, calibration_fpath=fpath, id="follower")

            def calibrate():
                # Must not offer to write the old file to the device.
                self.assertEqual(robot.calibration, {})
                self.assertEqual(json.loads(next((root/"backups").glob("*/pc_calibration.json")).read_text()), original)
                self.assertTrue(next((root/"backups").glob("*/device_calibration.json")).exists())
                bus.events.append("calibrate")
                if interrupt:
                    raise KeyboardInterrupt()
                robot.calibration = {"new": "measured configuration"}
                bus.is_calibrated = True

            robot.calibrate = calibrate
            if torque:
                with self.assertRaisesRegex(RuntimeError, "torque is ON"):
                    recalibrate(robot, root/"backups")
                self.assertFalse((root/"backups").exists())
                self.assertNotIn("calibrate", bus.events)
            elif interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    recalibrate(robot, root/"backups")
                self.assertIs(robot.calibration, original)
                failure = next((root/"backups").glob("*/calibration_failure.json"))
                self.assertEqual(json.loads(failure.read_text())['state'], 'incomplete')
                self.assertEqual(json.loads(fpath.read_text()), original)
            else:
                recalibrate(robot, root/"backups")
                self.assertIn("calibrate", bus.events)
            self.assertEqual(bus.events[-1], ("disconnect", False))
            self.assertFalse(bus.is_connected)

    def test_backup_before_fresh_calibration(self):
        self.run_case()

    def test_powered_motor_stops_without_torque_change(self):
        self.run_case(torque=1)

    def test_interruption_closes_bus(self):
        self.run_case(interrupt=True)

    def test_leader_entrypoint_uses_selected_teleoperation_profile(self):
        from calibrate_leader import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leader = SimpleNamespace(calibration_fpath=root/'my_leader.json')
            config = Mock(side_effect=lambda **kwargs:kwargs)
            module = SimpleNamespace(SO101LeaderConfig=config, SO101Leader=Mock(return_value=leader))
            with patch.dict(sys.modules, {'lerobot.teleoperators.so_leader':module}), \
                    patch('calibrate_leader.recalibrate') as calibrate:
                main(['--port','COM6','--calibration-dir',str(root),'--backup-dir',str(root/'backup')])
            self.assertEqual(config.call_args.kwargs['id'], 'my_leader')
            self.assertEqual(config.call_args.kwargs['port'], 'COM6')
            self.assertEqual(config.call_args.kwargs['calibration_dir'], root)
            self.assertTrue(config.call_args.kwargs['use_degrees'])
            calibrate.assert_called_once_with(leader, root/'backup')
            self.assertEqual(json.loads((root/'my_leader.source.json').read_text())['mode'], 'manual_leader_recalibration')


if __name__ == "__main__":
    unittest.main()
