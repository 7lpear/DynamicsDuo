"""Prepare board-to-robot calibration; preview commands never move the arm."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .geometry import fit_session, template
from follower_feedback import JOINTS as MOTORS


def write_new(path, value):
    """Never silently overwrite a measured calibration or pose."""
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def read_state(port, robot_id):
    # Lazy import: geometry commands work in the vision-only environment too.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
    bus = robot.bus
    try:
        # Robot.connect/configure can change motor settings. Only open the bus.
        bus.connect()
        if not robot.calibration or not bus.is_calibrated:
            raise ValueError("Robot calibration is missing or mismatched; state was not saved")
        positions = {k: float(v) for k, v in bus.sync_read("Present_Position", num_retry=2).items()}
        if set(positions) != set(MOTORS) or not all(math.isfinite(v) for v in positions.values()):
            raise ValueError("Incomplete or nonfinite joint state; state was not saved")
        state = {
            "mode": "read_only_joint_snapshot",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "port": port, "robot_id": robot_id,
            "positions": positions,
            "position_units": "body=degrees; gripper=percent; NOT Cartesian xyz",
            "raw_registers": {
                motor: {register: int(bus.read(register, motor, normalize=False, num_retry=2))
                        for register in ("Torque_Enable", "Present_Voltage", "Present_Temperature", "Status")}
                for motor in MOTORS
            },
        }
        return state
    finally:
        if bus.is_connected:
            # Preserve the initial torque state, including when reads fail.
            bus.disconnect(disable_torque=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an empty measurement session (no sample positions)")
    init.add_argument("--size", type=int, default=9)
    init.add_argument("--out", type=Path, default=Path("calibration/robot_board.session.json"))
    for command in ("check", "target", "plan-hover"):
        sub = commands.add_parser(command, help="Offline geometry only; does not open USB")
        sub.add_argument("--session", type=Path, default=Path("calibration/robot_board.session.json"))
        sub.add_argument("--out", type=Path)
        if command != "check":
            sub.add_argument("--coordinate", default="E5")
            sub.add_argument("--hover-mm", type=float, default=30)
        if command == "plan-hover":
            sub.add_argument("--start-xyz-mm", type=float, nargs=3, required=True,
                             help="Measured current tool centre in the SAME robot-base frame")
    state = commands.add_parser("read-state", help="Explicit read-only USB connection; no motion or torque writes")
    state.add_argument("--port", required=True, help="Current robot USB port; check it before connecting")
    state.add_argument("--robot-id", default="follower")
    state.add_argument("--out", type=Path)
    commands.add_parser("demo", help="Synthetic 9x9 geometry example, no hardware or measured calibration")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            write_new(args.out, template(args.size))
            print(f"Created {args.out}. Fill setup_id and measured xyz_mm, then run check.")
            return 0
        if args.command == "demo":
            session = template()
            session["setup_id"] = "SYNTHETIC_EXAMPLE_ONLY"
            for point in session["fit_points"] + session["validation_points"]:
                col = ord(point["coordinate"][0])-65
                row = int(point["coordinate"][1:])-1
                point["xyz_mm"] = [100+20*col, -80+20*row, 10]
            frame = fit_session(session)
            result = {"synthetic": True, "calibration": frame.report,
                      "preview": frame.hover_plan("E5", [100, -80, 80])}
        elif args.command == "read-state":
            if args.out is not None and args.out.exists():
                raise FileExistsError(f"Output already exists: {args.out}")
            result = read_state(args.port, args.robot_id)
        else:
            session = json.loads(args.session.read_text(encoding="utf-8-sig"))
            # Re-fit and validate the measured source EVERY time, not a stale saved report.
            frame = fit_session(session)
            if args.command == "check":
                result = frame.report
            elif args.command == "target":
                result = frame.target(args.coordinate, args.hover_mm)
            else:
                result = frame.hover_plan(args.coordinate, args.start_xyz_mm, args.hover_mm)
        if getattr(args, "out", None) is not None:
            write_new(args.out, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"STOP: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
