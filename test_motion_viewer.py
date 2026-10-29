"""Hardware-free checks for motion cleanup and local telemetry."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

from follower_viewer import FollowerViewer
from test_follower_motion import test_joint as run_joint


class FakeBus:
    def __init__(self, tracking_error=0):
        self.position = 2048
        self.tracking_error = tracking_error
        self.enabled = False
        self.goals = []

    def read(self, register, motor, **kwargs):
        return 0 if register == "Operating_Mode" else self.position + (self.tracking_error if self.enabled else 0)

    def write(self, register, motor, value, **kwargs):
        self.position = value
        self.goals.append(value)

    def enable_torque(self, motor, **kwargs):
        self.enabled = True

    def disable_torque(self, motor, **kwargs):
        self.enabled = False

    def sync_read(self, register, **kwargs):
        return {"shoulder_pan": (self.position - 2048) * 360 / 4096}


class Tests(unittest.TestCase):
    calibration = {"shoulder_pan": SimpleNamespace(range_min=0, range_max=4095)}

    def test_local_endpoint_and_assets(self):
        viewer = FollowerViewer(0, False)
        try:
            viewer.publish({"shoulder_pan": 8.5}, "Testing")
            with urlopen(viewer.url + "/state") as response:
                state = json.load(response)
                self.assertEqual(state["positions"]["shoulder_pan"], 8.5)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            for path in ["/", "/app.js", "/model/urdf/so101_new_calib.urdf", "/vendor/three.module.js"]:
                with urlopen(viewer.url + path) as response:
                    self.assertEqual(response.status, 200)
        finally:
            viewer.close()

    @patch("test_follower_motion.time.sleep")
    def test_motion_returns_and_releases(self, _sleep):
        bus = FakeBus()
        viewer = FollowerViewer(0, False)
        try:
            self.assertTrue(run_joint(bus, self.calibration, "shoulder_pan", viewer, True))
            self.assertFalse(bus.enabled)
            self.assertEqual(bus.position, 2048)
            self.assertLessEqual(max(abs(goal - 2048) for goal in bus.goals), 114)
            self.assertEqual(viewer.state["positions"]["shoulder_pan"], 0)
        finally:
            viewer.close()

    @patch("test_follower_motion.time.sleep")
    def test_tracking_fault_releases(self, _sleep):
        bus = FakeBus(24)
        with self.assertRaisesRegex(RuntimeError, "Tracking error"):
            run_joint(bus, self.calibration, "shoulder_pan", automatic=True)
        self.assertFalse(bus.enabled)

    @patch("test_follower_motion.time.sleep")
    def test_telemetry_fault_releases(self, _sleep):
        bus = FakeBus()
        class BrokenViewer:
            def sample(self, *args):
                raise RuntimeError("Telemetry failed")
        with self.assertRaisesRegex(RuntimeError, "Telemetry failed"):
            run_joint(bus, self.calibration, "shoulder_pan", BrokenViewer(), True)
        self.assertFalse(bus.enabled)

    def test_near_limit_skips_without_commands(self):
        bus = FakeBus()
        bus.position = 10
        self.assertTrue(run_joint(bus, self.calibration, "shoulder_pan", automatic=True))
        self.assertEqual(bus.goals, [])


if __name__ == "__main__":
    unittest.main()
