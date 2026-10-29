"""SO101 joint checks and coordinated motion demo with optional 3D telemetry."""
import argparse
import math
import time
from follower_viewer import FollowerViewer
from follower_demo import NAMES, POSES, run_demo, make_wide_plan, segment_duration
from types import SimpleNamespace


def test_joint(bus, calibration, motor, viewer=None, automatic=False):
    print(f"\n--- {motor} ---")
    print("Support the arm throughout; this joint releases immediately after the test.")
    print("Keep fingers out of joints and gripper. MOVE=test, Enter=skip, Q=quit.")
    choice = "MOVE" if automatic else input(f"{motor}> ").strip().upper()
    if choice == "Q":
        return False
    if choice != "MOVE":
        print("SKIP")
        return True
    if bus.read("Operating_Mode", motor) != 0:
        raise RuntimeError(f"{motor} is not in position mode.")
    # Read after the prompt so manual repositioning cannot leave a stale target.
    start = int(bus.read("Present_Position", motor, normalize=False, num_retry=2))
    cal = calibration[motor]
    span = cal.range_max - cal.range_min
    margin = 57 if motor != "gripper" else max(5, round(span * 0.02))
    distance = round(4095 * 10 / 360) if motor != "gripper" else max(1, round(span * 0.10))
    low, high = cal.range_min + margin, cal.range_max - margin
    if not low <= start <= high:
        print("SKIP: near calibrated limit. With torque off, gently reposition away from the end and rerun.")
        return True
    direction = 1 if start <= (low + high) / 2 else -1
    clearance = high - start if direction == 1 else start - low
    distance = min(distance, clearance)
    target = start + direction * distance
    if not low <= target <= high:
        print("SKIP: insufficient calibrated clearance.")
        return True
    try:
        bus.disable_torque(motor, num_retry=2)
        start = int(bus.read("Present_Position", motor, normalize=False, num_retry=2))
        if not low <= start <= high or abs(target - start) > distance + 5:
            raise RuntimeError("Joint shifted while releasing torque. Support it and retry.")
        bus.write("Goal_Position", motor, start, normalize=False)
        bus.enable_torque(motor, num_retry=2)
        for begin, end in ((start, target), (target, start)):
            for step in range(1, 101):
                goal = round(begin + (end - begin) * step / 100)
                bus.write("Goal_Position", motor, goal, normalize=False)
                time.sleep(0.05)
                actual = int(bus.read("Present_Position", motor, normalize=False, num_retry=2))
                if abs(actual - goal) > 23:
                    raise RuntimeError("Tracking error exceeded 2 motor degrees; stopping.")
                if viewer:
                    viewer.sample(bus, f"Moving: {motor}")
            time.sleep(0.4)
            actual = int(bus.read("Present_Position", motor, normalize=False, num_retry=2))
            print(f"Target={end}, actual={actual}, error={abs(actual-end)} ticks")
            if abs(actual - end) > 11:
                raise RuntimeError("Target not reached within 1 motor degree; stopping.")
        print(f"PASS: {motor} moved and returned.")
    finally:
        try:
            bus.disable_torque(motor, num_retry=2)
            print(f"{motor}: torque OFF. Keep arm supported.")
        except Exception as exc:
            raise RuntimeError("Cannot release torque. Disconnect motor power now.") from exc
    if viewer:
        viewer.sample(bus, f"Finished: {motor}; torque OFF")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = NAMES
    parser.add_argument("--joint", choices=names + ["remaining", "all"], default="remaining")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--robot-id", default="follower")
    parser.add_argument("--viewer", action="store_true", help="Show measured joint positions in a local 3D viewer")
    parser.add_argument("--viewer-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="Animate the viewer without importing LeRobot or opening USB")
    parser.add_argument("--duration", type=float, default=30, help="Simulation duration in seconds")
    parser.add_argument("--auto", action="store_true", help="Run the selected joint check without MOVE prompts; support the other joints")
    parser.add_argument("--demo", action="store_true", help="Move all six joints together, then return to the starting pose")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Demo motion scale, 0.2..1.5 (default 1)")
    parser.add_argument("--speed", type=float, default=None, help="Peak command speed; --load-aware defaults to 6 and caps at 8, ordinary demo defaults to 12")
    parser.add_argument("--wide", action="store_true", help="Virtual wide preview only; hardware wide motion is disabled after tracking fault")
    parser.add_argument("--analyze-load", action="store_true", help="Read-only position, static gravity estimate and servo diagnostics; no torque enable")
    parser.add_argument("--load-aware", action="store_true", help="Conservative base/wrist/gripper motion; fixed shoulder/elbow, static-load filter")
    parser.add_argument("--range-fraction", type=float, default=0.7, help="Fraction of travel after endpoint margins for --wide, 0.2..0.85")
    parser.add_argument("--cycles", type=int, default=1, help="Demo repetitions, 1..5")
    args = parser.parse_args()
    args.demo = args.demo or args.wide or args.load_aware
    if args.wide and not args.simulate:
        parser.error("--wide disabled after tracking fault. Use --analyze-load first.")
    if args.load_aware and args.simulate:
        parser.error("--load-aware requires measured pose; use --analyze-load for read-only diagnostics.")
    if args.speed is None:
        args.speed = 6.0 if args.load_aware else 25.0 if args.wide else 12.0
    if not math.isfinite(args.amplitude) or not 0.2 <= args.amplitude <= 1.5:
        parser.error("--amplitude must be 0.2..1.5")
    if not math.isfinite(args.speed) or not 3 <= args.speed <= 40:
        parser.error("--speed must be 3..40")
    if not math.isfinite(args.range_fraction) or not 0.2 <= args.range_fraction <= 0.85:
        parser.error("--range-fraction must be 0.2..0.85")
    if not 1 <= args.cycles <= 5:
        parser.error("--cycles must be 1..5")
    if args.simulate:
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("--duration must be positive")
        viewer = FollowerViewer(args.viewer_port, not args.no_browser)
        try:
            preview_segments = []
            if args.wide:
                # Illustrative virtual calibration, never opens serial or imports LeRobot.
                virtual_start = dict.fromkeys(names, 2048)
                virtual_calibration = {name: SimpleNamespace(range_min=0, range_max=4095) for name in names}
                virtual_plan, factors = make_wide_plan(virtual_start, virtual_calibration, args.range_fraction)
                previous = virtual_start
                timeline = 0.0
                for label, target in virtual_plan:
                    duration = segment_duration(previous, target, factors, args.speed)
                    preview_segments.append((timeline, duration, previous, target))
                    timeline += duration
                    previous = target
            started = time.monotonic()
            while time.monotonic() - started < args.duration:
                elapsed = time.monotonic() - started
                if args.wide:
                    phase = elapsed % timeline
                    for begin_time, duration, begin, end in preview_segments:
                        if phase < begin_time + duration:
                            t = max(0, (phase - begin_time) / duration)
                            blend = 10*t**3 - 15*t**4 + 6*t**5
                            raw = {name: begin[name] + (end[name] - begin[name]) * blend for name in names}
                            positions = {name: (raw[name] - 2048) / factors[name] for name in names}
                            positions["gripper"] = raw["gripper"] / 4095 * 100
                            break
                elif args.demo:
                    # Preview the same relative choreography; simulation starts at zero / gripper 50%.
                    preview = [POSES[-1], *POSES]
                    segment = int(elapsed / 5) % (len(preview) - 1)
                    t = (elapsed % 5) / 5
                    blend = 10*t**3 - 15*t**4 + 6*t**5
                    begin, end = preview[segment][1], preview[segment + 1][1]
                    positions = {name: (begin[i] + (end[i] - begin[i]) * blend) * args.amplitude
                                 for i, name in enumerate(names)}
                    positions["gripper"] += 50
                else:
                    positions = {name: 10 * math.sin(elapsed + index * 0.5) for index, name in enumerate(names)}
                    positions["gripper"] = 50 + 30 * math.sin(elapsed)
                viewer.publish(positions, "Simulation: virtual calibration; no robot connected", source="simulation")
                time.sleep(0.05)
        finally:
            viewer.close()
        return
    # Simulation/help work even when the robot environment is unavailable.
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise RuntimeError("LeRobot is not installed in this Python environment. See MOTION_VIEWER.md.") from exc
    selected = names if args.joint == "all" else names[1:] if args.joint == "remaining" else [args.joint]
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, use_degrees=True))
    bus = robot.bus
    viewer = FollowerViewer(args.viewer_port, not args.no_browser) if args.viewer else None
    try:
        # Do not call Robot.connect(): it enables torque on all motors.
        bus.connect()
        if not robot.calibration or not bus.is_calibrated:
            raise RuntimeError("Calibration missing or mismatched.")
        if args.analyze_load:
            from follower_load import report_load
            report_load(bus)
            return
        print("Current positions (degrees; gripper is percent):")
        for name, value in bus.sync_read("Present_Position", num_retry=2).items():
            print(f"  {name}: {value:.2f}")
        # Require a released arm rather than unexpectedly releasing loaded joints.
        for name in names:
            if bus.read("Torque_Enable", name, num_retry=2) != 0:
                raise RuntimeError(f"{name} is powered. Support arm, turn motor power off/on, then retry.")
        if viewer:
            viewer.sample(bus, "Ready: waiting for joint test")
        if args.demo:
            run_demo(bus, robot.calibration, viewer, args.auto, args.amplitude, args.speed, args.cycles,
                     args.wide, args.range_fraction, args.load_aware)
            print("Demo session finished.")
            return
        print("Test: body joints up to 10 degrees; gripper up to 10% of calibrated travel.")
        print("Only the selected joint is powered. Support other joints without blocking motion.")
        for motor in selected:
            if not test_joint(bus, robot.calibration, motor, viewer, args.auto):
                break
        print("Session finished. Check PASS/SKIP results above.")
    finally:
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            if viewer:
                viewer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:
        print(f"STOP: {exc}")
        raise SystemExit(1)
