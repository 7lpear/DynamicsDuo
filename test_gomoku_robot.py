"""Hardware-free calibration, path geometry, and read-only serial contract checks."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gomoku_robot.__main__ import main, read_state
from gomoku_robot.geometry import cell, fit_session, template


def measured_session():
    session = template()
    session["setup_id"] = "test-fixed-setup"
    # A board rotated in the base XY plane, with image rows toward base -Y.
    angle = np.radians(30)
    column = 22 * np.array([np.cos(angle), np.sin(angle), 0])
    row = 22 * np.array([np.sin(angle), -np.cos(angle), 0])
    for point in session["fit_points"] + session["validation_points"]:
        col, r = cell(point["coordinate"], 9)
        point["xyz_mm"] = (np.array([80, -20, 12]) + col*column + r*row).tolist()
    return session


class GeometryTests(unittest.TestCase):
    def test_all_81_cells_and_upward_normal(self):
        session = measured_session()
        frame = fit_session(session)
        a1 = np.asarray(session["fit_points"][0]["xyz_mm"])
        i1 = np.asarray(session["fit_points"][1]["xyz_mm"])
        a9 = np.asarray(session["fit_points"][3]["xyz_mm"])
        for row in range(9):
            for col in range(9):
                expected = a1 + col*(i1-a1)/8 + row*(a9-a1)/8
                np.testing.assert_allclose(frame.surface(f"{chr(65+col)}{row+1}"), expected, atol=1e-9)
        np.testing.assert_allclose(frame.normal, [0, 0, 1], atol=1e-9)
        self.assertLess(frame.report["max_validation_error_mm"], 1e-9)

    def test_grid_labels_match_vision(self):
        self.assertEqual(cell("A1", 9), (0, 0))
        self.assertEqual(cell("e5", 9), (4, 4))
        self.assertEqual(cell("I9", 9), (8, 8))
        for invalid in ("J1", "A0", "A01", "A10", "1A", "A-1", " A1", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cell(invalid, 9)

    def test_holdout_error_rejects_without_refitting(self):
        session = measured_session()
        session["validation_points"][0]["xyz_mm"][0] += 4
        with self.assertRaisesRegex(ValueError, "validation error"):
            fit_session(session)

    def test_corner_warp_rejected(self):
        session = measured_session()
        session["fit_points"][2]["xyz_mm"][2] += 12
        with self.assertRaisesRegex(ValueError, "Calibration rejected"):
            fit_session(session)

    def test_corner_label_swap_rejected(self):
        session = measured_session()
        points = session["fit_points"]
        points[1]["xyz_mm"], points[2]["xyz_mm"] = points[2]["xyz_mm"], points[1]["xyz_mm"]
        with self.assertRaises(ValueError):
            fit_session(session)

    def test_reject_incomplete_and_invalid_inputs(self):
        mutations = [
            lambda s: s.update(setup_id=None),
            lambda s: s.update(units="m"),
            lambda s: s.update(frame="camera"),
            lambda s: s.update(coordinate_convention="A1=bottom_left"),
            lambda s: s.update(board_size=True),
            lambda s: s.update(max_error_mm=float("nan")),
            lambda s: s.update(max_error_mm=6),
            lambda s: s["fit_points"][0].update(xyz_mm=None),
            lambda s: s["fit_points"][0].update(xyz_mm=[0, 0, float("inf")]),
            lambda s: s["fit_points"][0].update(xyz_mm=[True, 0, 0]),
            lambda s: s["fit_points"].pop(),
            lambda s: s["fit_points"].append(deepcopy(s["fit_points"][0])),
            lambda s: s.update(validation_points=[]),
            lambda s: s.update(validation_points=[deepcopy(s["fit_points"][0])]),
        ]
        for mutate in mutations:
            session = measured_session()
            mutate(session)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                fit_session(session)
        with self.assertRaises(ValueError):
            fit_session(template())

    def test_tilted_plane_hover_uses_normal_and_stays_above_plane(self):
        session = measured_session()
        angle = np.radians(10)
        rotate = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
        for point in session["fit_points"] + session["validation_points"]:
            point["xyz_mm"] = (rotate @ point["xyz_mm"]).tolist()
        frame = fit_session(session)
        for initial_height in (8, 90):
            start = frame.surface("A1") + initial_height * frame.normal
            plan = frame.hover_plan("E5", start, 30)
            waypoints = [np.array(p["xyz_mm"]) for p in plan["waypoints"]]
            np.testing.assert_allclose(waypoints[1]-start,
                                       max(0, 30-initial_height)*frame.normal, atol=1e-9)
            for index in range(1, len(waypoints)-1):
                for ratio in np.linspace(0, 1, 15):
                    point = (1-ratio)*waypoints[index] + ratio*waypoints[index+1]
                    self.assertGreaterEqual((point-frame.basis[0]) @ frame.normal, 30-1e-9)
            np.testing.assert_allclose(waypoints[-1], frame.surface("E5")+30*frame.normal)
        with self.assertRaises(ValueError):
            frame.hover_plan("E5", frame.surface("E5")-frame.normal)
        for invalid in (0, -5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                frame.target("E5", invalid)


class ReadOnlyTests(unittest.TestCase):
    def fake_module(self, fail_read=False, calibrated=True):
        events = []

        class Bus:
            is_connected = False
            is_calibrated = calibrated

            def connect(self):
                events.append("connect")
                self.is_connected = True

            def sync_read(self, register, **kwargs):
                events.append(("sync_read", register))
                if fail_read:
                    raise RuntimeError("read failed")
                return dict.fromkeys(("shoulder_pan", "shoulder_lift", "elbow_flex",
                                      "wrist_flex", "wrist_roll", "gripper"), 0)

            def read(self, register, motor, **kwargs):
                return 0

            def disconnect(self, disable_torque=True):
                events.append(("disconnect", disable_torque))
                self.is_connected = False

        module = SimpleNamespace(SO101FollowerConfig=lambda **kw: kw,
                                 SO101Follower=lambda config: SimpleNamespace(bus=Bus(), calibration={"valid": True}))
        return module, events

    def test_read_only_and_disconnect_never_changes_torque(self):
        for fail, calibrated in ((False, True), (True, True), (False, False)):
            module, events = self.fake_module(fail, calibrated)
            with patch.dict("sys.modules", {"lerobot.robots.so_follower": module}):
                if fail or not calibrated:
                    with self.assertRaises((RuntimeError, ValueError)):
                        read_state("TEST", "test")
                else:
                    snapshot = read_state("TEST", "test")
                    self.assertIn("NOT Cartesian", snapshot["position_units"])
            self.assertEqual(events[-1], ("disconnect", False))


class CommandTests(unittest.TestCase):
    def test_init_does_not_overwrite_and_incomplete_session_blocks_target(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/"session.json"
            self.assertEqual(main(["init", "--out", str(path)]), 0)
            original = path.read_bytes()
            with self.assertRaises(SystemExit):
                main(["init", "--out", str(path)])
            self.assertEqual(original, path.read_bytes())
            with self.assertRaises(SystemExit):
                main(["target", "--session", str(path)])

    def test_check_and_plan_revalidate_source(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path, output = Path(tmp)/"session.json", Path(tmp)/"plan.json"
            session = measured_session()
            path.write_text(json.dumps(session), encoding="utf-8")
            self.assertEqual(main(["check", "--session", str(path)]), 0)
            self.assertEqual(main(["plan-hover", "--session", str(path), "--coordinate", "E5",
                                   "--start-xyz-mm", "80", "-20", "80", "--out", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["mode"], "offline_cartesian_preview")
            session["validation_points"][0]["xyz_mm"][0] += 10
            path.write_text(json.dumps(session), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(["target", "--session", str(path)])

    def test_demo_explicitly_synthetic(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(["demo"]), 0)
        self.assertTrue(json.loads(stream.getvalue())["synthetic"])


if __name__ == "__main__":
    unittest.main()
