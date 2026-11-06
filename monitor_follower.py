"""Display measured SO-101 telemetry without torque, calibration or motion writes."""
import argparse
from contextlib import ExitStack
import math
import time

from follower_feedback import JOINTS, read_diagnostics
from follower_viewer import FollowerViewer


def monitor(robot, viewer, duration=300., fps=10.):
    bus = robot.bus
    try:
        # Robot.connect() configures and powers motors. Open only the serial bus.
        bus.connect()
        if not robot.calibration or not bus.is_calibrated:
            raise RuntimeError('Calibration missing or mismatched; read-only monitor stopped')
        print('READ ONLY: no torque, calibration or motion writes. Ctrl+C stops.')
        deadline = time.monotonic()+duration
        next_diagnostics = 0.
        diagnostics = None
        diagnostics_at = None
        while time.monotonic() < deadline:
            started = time.monotonic()
            if started >= next_diagnostics:
                diagnostics = read_diagnostics(bus)
                diagnostics_at = time.time()
                next_diagnostics = time.monotonic()+.5
            positions = bus.sync_read('Present_Position', num_retry=2)
            if set(positions) != set(JOINTS):
                raise RuntimeError('Incomplete follower positions')
            viewer.publish(positions, 'Read-only follower monitor',
                           diagnostics=diagnostics, diagnostics_timestamp=diagnostics_at)
            time.sleep(max(0.,1/fps-(time.monotonic()-started)))
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM10')
    parser.add_argument('--robot-id', default='follower')
    parser.add_argument('--duration', type=float, default=300.)
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--viewer-port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and positive')
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 30:
        parser.error('--fps must be 1..30')
    if not 1 <= args.viewer_port <= 65535:
        parser.error('--viewer-port must be 1..65535')
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(port=args.port,id=args.robot_id,use_degrees=True))
    with ExitStack() as resources:
        viewer = FollowerViewer(port=args.viewer_port,open_browser=not args.no_browser)
        resources.callback(viewer.close)
        monitor(robot,viewer,args.duration,args.fps)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Read-only monitor stopped.')
    except Exception as error:
        print(f'STOP: {error}')
        raise SystemExit(1)
