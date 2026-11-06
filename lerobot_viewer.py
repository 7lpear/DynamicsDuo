"""Read-only 3D display for the native LeRobot teleoperation loop.

Uses the observation already read by LeRobot. Never opens USB or writes motors.
"""
import math
import logging
import time

from follower_viewer import FollowerViewer
from follower_feedback import JOINTS


def read_with_retry(read, label):
    """Retry only failed reads, at most three attempts within a short window.

    Never reuse old leader positions or retry motor writes. A persistent error
    propagates to native session cleanup and torque release.
    """
    deadline = time.monotonic()+.25
    for attempt in range(1,4):
        try:
            return read()
        except ConnectionError as error:
            if attempt == 3 or time.monotonic() >= deadline:
                raise ConnectionError(f'{label}: communication failed after {attempt} attempts; {error}') from error
            logging.warning('%s: read failed, retry %s/2: %s',label,attempt,error)
            time.sleep(.01)
            if time.monotonic() >= deadline:
                raise ConnectionError(f'{label}: communication failed after {attempt} attempts; {error}') from error


def open_viewer(robot, enabled=True, port=8765, open_browser=True):
    if not enabled or robot.name != 'so_follower':
        return None
    return FollowerViewer(port=port, open_browser=open_browser)


def positions_in_degrees(robot, values):
    positions = {name: float(values[name+'.pos']) for name in JOINTS}
    if not robot.config.use_degrees:
        for name in JOINTS[:-1]:
            cal = robot.calibration[name]
            positions[name] *= (cal.range_max-cal.range_min)*360/4095/200
    if not all(math.isfinite(value) for value in positions.values()):
        raise ValueError('Nonfinite follower observation; cannot display measured motion')
    return positions


def publish_observation(viewer, robot, observation, sent_action):
    if viewer is None:
        return
    viewer.publish(
        positions_in_degrees(robot, observation),
        'LeRobot teleoperation: measured follower motion',
        source='actual',
        command_positions=positions_in_degrees(robot, sent_action),
    )


def disconnect_device(device):
    """Also release a serial bus left open by an interrupted native connect()."""
    if device.is_connected:
        device.disconnect()
    elif getattr(getattr(device, 'bus', None), 'is_connected', False):
        device.bus.disconnect()
        for camera in getattr(device, 'cameras', {}).values():
            if camera.is_connected:
                camera.disconnect()
