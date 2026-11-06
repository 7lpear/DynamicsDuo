"""Manual SO-101 recalibration without Robot.connect/configure or torque enable."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from calibration_io import verified_calibration_writes


def recalibrate(robot, backup_root):
    bus = robot.bus
    try:
        # Robot.connect() calls configure(), which re-enables torque.
        # Open only the bus so no stale goal can be activated on startup.
        bus.connect()
        for motor in bus.motors:
            if bus.read("Torque_Enable", motor, num_retry=2) != 0:
                raise RuntimeError("Motor torque is ON. Support the arm, power-cycle motor power, "
                                   "then retry. This command did not change torque.")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = Path(backup_root) / f"{robot.id}_{stamp}"
        backup.mkdir(parents=True, exist_ok=False)
        if robot.calibration_fpath.exists():
            shutil.copy2(robot.calibration_fpath, backup / "pc_calibration.json")
        device_calibration = bus.read_calibration()
        (backup / "device_calibration.json").write_text(
            json.dumps({name: asdict(value) for name, value in device_calibration.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Backed up PC/device calibration: {backup}")
        print("Keep the arm supported throughout manual calibration.")
        print("This command never enables torque or sends Goal_Position.")
        # Use LeRobot's mid-range and sweep prompts; bypass old-file reuse.
        previous = robot.calibration
        try:
            # A backup already exists; bypass the reuse prompt and always MEASURE.
            robot.calibration = {}
            print("Fresh calibration: old-file reuse is disabled.")
            with verified_calibration_writes(bus):
                robot.calibrate()
        except BaseException as error:
            robot.calibration = previous
            (backup / 'calibration_failure.json').write_text(
                json.dumps({'state':'incomplete', 'error':str(error),
                            'note':'Motor registers may be partially changed. Complete calibration before teleoperation.'}, indent=2)+'\n',
                encoding='utf-8',
            )
            print(f'CALIBRATION INCOMPLETE. Original PC/device backup preserved at {backup}. Recalibrate before motion.')
            raise
        if not bus.is_calibrated:
            raise RuntimeError("Saved calibration does not match device; do not move the arm")
        print(f"PASS: PC/device calibration match: {robot.calibration_fpath}")
        print("Finished with torque OFF. Continue supporting the arm.")
    finally:
        if bus.is_connected:
            # Preserve torque state. calibrate() only DISABLES torque.
            bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default="follower")
    parser.add_argument("--backup-dir", type=Path, default=Path(__file__).resolve().parent / "calibration/robot_backups")
    args = parser.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, use_degrees=True))
    recalibrate(robot, args.backup_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("STOP: Calibration interrupted. Keep arm supported and recalibrate before motion.")
        raise SystemExit(1)
    except Exception as error:
        print(f"STOP: {error}")
        raise SystemExit(1)
