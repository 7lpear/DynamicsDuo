"""Hardware-free verification of coordinated motion and fault cleanup."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from follower_demo import NAMES, make_plan, make_wide_plan, run_demo


class Bus:
    def __init__(self, fault=False):
        self.positions = dict.fromkeys(NAMES, 2048)
        self.powered = set()
        self.commands = []
        self.fault = fault

    def read(self, register, motor, **kwargs):
        return int(motor in self.powered) if register == "Torque_Enable" else 0

    def sync_read(self, register, **kwargs):
        positions = self.positions.copy()
        if self.fault and self.powered:
            positions["elbow_flex"] += 100
        return positions

    def sync_write(self, register, values, **kwargs):
        self.positions = values.copy()
        self.commands.append(values.copy())

    def enable_torque(self, motors, **kwargs):
        self.powered.update(motors)

    def disable_torque(self, motors, **kwargs):
        self.powered.difference_update(motors)


class Tests(unittest.TestCase):
    calibration = {motor: SimpleNamespace(range_min=0, range_max=4095) for motor in NAMES}

    def test_wide_plan_preserves_margins_and_increases_range(self):
        start = dict.fromkeys(NAMES, 2048)
        start["shoulder_lift"] = 0
        start["gripper"] = 4095
        plan, _ = make_wide_plan(start, self.calibration, 0.85)
        for _, target in plan[:-1]:
            for motor, value in target.items():
                margin = 123 if motor == "gripper" else 114
                self.assertTrue(margin <= value <= 4095 - margin)
        for motor in NAMES:
            values = [target[motor] for _, target in plan[:-1]]
            self.assertGreater(max(values) - min(values), 3000)
        self.assertEqual(plan[-1][1], start)

    def test_wide_invalid_fraction_rejected(self):
        for fraction in (0.1, 1.0, float('nan')):
            with self.assertRaises(ValueError):
                make_wide_plan(dict.fromkeys(NAMES, 2048), self.calibration, fraction)

    @patch("follower_demo.time.sleep")
    @patch("builtins.input", return_value="")
    def test_wide_hardware_run_disabled_without_commands(self, _input, sleep):
        bus = Bus()
        with self.assertRaisesRegex(RuntimeError, 'Wide motion disabled'):
            run_demo(bus, self.calibration, automatic=True, wide=True, speed=40, range_fraction=0.85)
        self.assertFalse(bus.powered)
        self.assertEqual(bus.positions, dict.fromkeys(NAMES, 2048))
        self.assertEqual(bus.commands, [])
        _input.assert_not_called()
        sleep.assert_not_called()

    @patch("follower_demo.time.sleep")
    def test_wide_tracking_fault_still_releases_all_joints(self, _sleep):
        bus = Bus(fault=True)
        with self.assertRaisesRegex(RuntimeError, "Wide motion disabled"):
            run_demo(bus, self.calibration, automatic=True, wide=True, speed=30)
        self.assertFalse(bus.powered)

    def test_scaled_plan_respects_all_limits(self):
        start = dict.fromkeys(NAMES, 2048)
        start["shoulder_pan"] = 180
        plan, _, scales = make_plan(start, self.calibration, 1.5)
        self.assertEqual(scales["shoulder_pan"], 1)
        self.assertTrue(all(target["shoulder_pan"] >= start["shoulder_pan"] for _, target in plan))
        for _, target in plan:
            for motor, value in target.items():
                margin = 123 if motor == "gripper" else 80
                self.assertTrue(margin <= value <= 4095 - margin)
        self.assertEqual(plan[-1][1], start)

    def test_limit_rejection_has_no_motor_commands(self):
        bus = Bus()
        bus.positions["elbow_flex"] = -1
        with self.assertRaisesRegex(RuntimeError, "outside"):
            run_demo(bus, self.calibration, automatic=True)
        self.assertEqual(bus.commands, [])
        self.assertFalse(bus.powered)

    def test_endpoint_starts_use_available_direction_and_return(self):
        for endpoint in (0, 4095):
            start = dict.fromkeys(NAMES, 2048)
            start["shoulder_lift"] = endpoint
            start["gripper"] = endpoint
            plan, _, _ = make_plan(start, self.calibration, 1.5)
            for _, target in plan:
                self.assertTrue(all(0 <= value <= 4095 for value in target.values()))
            for motor in ("shoulder_lift", "gripper"):
                values = [target[motor] for _, target in plan]
                self.assertGreater(max(values) - min(values), 200)
                if endpoint == 0:
                    self.assertTrue(all(value >= endpoint for value in values))
                else:
                    self.assertTrue(all(value <= endpoint for value in values))
            self.assertEqual(plan[-1][1], start)

    @patch("follower_demo.time.sleep")
    @patch("builtins.input", return_value="")
    def test_all_joints_move_return_and_release(self, _input, sleep):
        bus = Bus()
        run_demo(bus, self.calibration, automatic=True)
        self.assertEqual(bus.commands[0], dict.fromkeys(NAMES, 2048))
        self.assertEqual(bus.positions, dict.fromkeys(NAMES, 2048))
        self.assertFalse(bus.powered)
        for motor in NAMES:
            values = [command[motor] for command in bus.commands]
            self.assertGreater(max(values) - min(values), 300)
        # Every commanded step stays below the specified peak velocity (+ rounding tick).
        movement = bus.commands[1:]
        delays = [call.args[0] for call in sleep.call_args_list if call.args[0] != 0.4]
        _, factors, _ = make_plan(dict.fromkeys(NAMES, 2048), self.calibration)
        previous = bus.commands[0]
        for goal, delay in zip(movement, delays):
            for motor in NAMES:
                self.assertLessEqual(abs(goal[motor] - previous[motor]), factors[motor] * 12 * delay + 1.1)
            previous = goal
        _input.assert_called_once()

    @patch("follower_demo.time.sleep")
    def test_tracking_fault_releases_every_joint(self, _sleep):
        bus = Bus(fault=True)
        with self.assertRaisesRegex(RuntimeError, "Tracking error"):
            run_demo(bus, self.calibration, automatic=True)
        self.assertFalse(bus.powered)

    @patch("follower_demo.time.sleep")
    def test_partial_torque_enable_failure_releases_every_joint(self, _sleep):
        bus = Bus()
        def fail(motors, **kwargs):
            bus.powered.add(motors[0])
            raise RuntimeError("enable failed")
        bus.enable_torque = fail
        with self.assertRaisesRegex(RuntimeError, "enable failed"):
            run_demo(bus, self.calibration, automatic=True)
        self.assertFalse(bus.powered)


if __name__ == "__main__":
    unittest.main()
