"""Native CLI integration checks with fake arms; never connects to USB."""
from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

from lerobot_viewer import JOINTS, open_viewer, positions_in_degrees, publish_observation, read_with_retry
from setup_lerobot_viewer import patched_source


class Viewer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def publish(self, positions, status, **kwargs):
        self.frames.append((positions,status,kwargs))

    def close(self):
        self.closed = True


class Device:
    name = 'so_follower'
    config = SimpleNamespace(use_degrees=True)
    def __init__(self, failure=None):
        self.is_connected = False
        self.failure = failure
        self.commands = []
        self.action_features = {n+'.pos':float for n in JOINTS}

    def connect(self):
        self.is_connected = True
        if self.failure:
            raise RuntimeError('connect failed')

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {n+'.pos':index for index,n in enumerate(JOINTS)}

    def get_action(self):
        return {n+'.pos':20.+index for index,n in enumerate(JOINTS)}

    def send_action(self, action):
        self.commands.append(action.copy())
        return action.copy()


class Tests(unittest.TestCase):
    def test_real_http_state_uses_follower_feedback_not_leader_target(self):
        robot = Device()
        with redirect_stdout(io.StringIO()):
            viewer = open_viewer(robot,port=0,open_browser=False)
        try:
            publish_observation(viewer,robot,robot.get_observation(),robot.get_action())
            with urlopen(viewer.url+'/state') as response:
                state = json.load(response)
            self.assertEqual(state['positions']['elbow_flex'],2.)
            self.assertEqual(state['feedback']['command_positions']['elbow_flex'],22.)
            self.assertEqual(state['source'],'actual')
        finally:
            viewer.close()

    def test_percent_normalization_converts_body_to_degrees_only(self):
        robot = Device()
        robot.config = SimpleNamespace(use_degrees=False)
        robot.calibration = {n:SimpleNamespace(range_min=1000,range_max=3000) for n in JOINTS}
        positions = positions_in_degrees(robot,dict.fromkeys((n+'.pos' for n in JOINTS),50.))
        self.assertAlmostEqual(positions['elbow_flex'],2000*360/4095/4)
        self.assertEqual(positions['gripper'],50.)

    def test_disabled_or_other_robot_does_not_open_viewer(self):
        with patch('lerobot_viewer.FollowerViewer') as factory:
            self.assertIsNone(open_viewer(Device(),enabled=False))
            self.assertIsNone(open_viewer(SimpleNamespace(name='other_robot')))
        factory.assert_not_called()

    def test_invalid_feedback_is_not_animated(self):
        with self.assertRaises(ValueError):
            positions_in_degrees(Device(),dict.fromkeys((n+'.pos' for n in JOINTS),float('nan')))

    def test_patch_is_reproducible_and_rejects_unexpected_sdk(self):
        source = '''import logging
class Config:
    fps: int = 60
def teleop_loop(
    display_compressed_images: bool = False,
):
    while True:
        obs = robot.get_observation()
        raw_action = teleop.get_action()
        _ = robot.send_action(robot_action_to_send)
def teleoperate():
    teleop.connect()
    robot.connect()
    try:
        teleop_loop(
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()

def main():
    pass
'''
        updated = patched_source(source)
        self.assertEqual(updated,patched_source(updated))
        with patch('setup_lerobot_viewer.ROOT',Path('C:/moved-project')):
            relocated = patched_source(updated)
        self.assertIn("sys.path.insert(0, 'C:\\\\moved-project')",relocated)
        from lerobot.scripts import lerobot_teleoperate as native
        installed = Path(native.__file__).read_text(encoding='utf-8')
        self.assertIn('publish_observation(follower_viewer, robot, obs, sent_action)',installed)
        self.assertIn('with ExitStack() as resources:',installed)
        with self.assertRaises(RuntimeError):
            patched_source('import logging\n')

    def test_transient_read_retries_and_returns_fresh_data(self):
        read = Mock(side_effect=[ConnectionError('packet missing'),{'elbow_flex.pos':25.}])
        with patch('lerobot_viewer.time.sleep'), self.assertLogs(level='WARNING'):
            self.assertEqual(read_with_retry(read,'Leader action'),{'elbow_flex.pos':25.})
        self.assertEqual(read.call_count,2)

    def test_persistent_read_is_bounded_and_noncommunication_errors_are_not_retried(self):
        read = Mock(side_effect=ConnectionError('missing packet'))
        with patch('lerobot_viewer.time.sleep'), self.assertLogs(level='WARNING'):
            with self.assertRaisesRegex(ConnectionError,'Leader action: communication failed after 3 attempts'):
                read_with_retry(read,'Leader action')
        self.assertEqual(read.call_count,3)
        read = Mock(side_effect=ValueError('bad calibration'))
        with self.assertRaises(ValueError):
            read_with_retry(read,'Leader action')
        self.assertEqual(read.call_count,1)

    def test_read_retry_window_does_not_expand_after_a_slow_failed_read(self):
        read = Mock(side_effect=ConnectionError('slow timeout'))
        with patch('lerobot_viewer.time.monotonic',side_effect=[0.,.3]):
            with self.assertRaisesRegex(ConnectionError,'after 1 attempts'):
                read_with_retry(read,'Leader action')
        self.assertEqual(read.call_count,1)

    def test_native_read_failure_never_sends_stale_leader_commands(self):
        from lerobot.scripts import lerobot_teleoperate as native
        leader,robot = Device(),Device()
        leader.get_action = Mock(side_effect=ConnectionError('missing packet'))
        identity = lambda pair:pair[0]
        with patch('lerobot_viewer.time.sleep'), self.assertLogs(level='WARNING'):
            with self.assertRaises(ConnectionError):
                native.teleop_loop(leader,robot,60,identity,identity,lambda obs:obs,duration=-1,follower_viewer=Viewer())
        self.assertEqual(robot.commands,[])

    def test_retry_does_not_start_if_window_expires_during_sleep(self):
        read = Mock(side_effect=ConnectionError('packet missing'))
        with patch('lerobot_viewer.time.monotonic',side_effect=[0.,.245,.255]), \
             patch('lerobot_viewer.time.sleep'), self.assertLogs(level='WARNING'):
            with self.assertRaisesRegex(ConnectionError,'after 1 attempts'):
                read_with_retry(read,'Leader action')
        self.assertEqual(read.call_count,1)

    def test_native_loop_keeps_original_actions_and_publishes_measured_observation(self):
        from lerobot.scripts import lerobot_teleoperate as native
        leader,robot,viewer = Device(),Device(),Viewer()
        identity = lambda pair:pair[0]
        with patch.object(native,'precise_sleep'), patch('builtins.input',side_effect=AssertionError('No start prompt')), redirect_stdout(io.StringIO()):
            native.teleop_loop(leader,robot,60,identity,identity,lambda obs:obs,duration=-1,follower_viewer=viewer)
        self.assertEqual(robot.commands,[leader.get_action()])
        self.assertEqual(viewer.frames[0][0]['elbow_flex'],2.)
        self.assertEqual(viewer.frames[0][2]['command_positions']['elbow_flex'],22.)
        self.assertFalse(leader.commands)

    def test_native_session_auto_opens_and_cleans_up_on_stop_and_failure(self):
        from lerobot.scripts import lerobot_teleoperate as native
        config = native.TeleoperateConfig(teleop=None,robot=None)
        for failure in ('none','connect','loop','interrupt','viewer'):
            with self.subTest(failure=failure):
                leader,robot,viewer = Device(),Device('connect' if failure=='connect' else None),Viewer()
                loop_error = KeyboardInterrupt() if failure=='interrupt' else RuntimeError('loop failed') if failure=='loop' else None
                with patch.object(native,'make_teleoperator_from_config',return_value=leader), \
                     patch.object(native,'make_robot_from_config',return_value=robot), \
                     patch.object(native,'make_default_processors',return_value=(None,None,None)), \
                     patch.object(native,'open_viewer',side_effect=RuntimeError('port busy') if failure=='viewer' else None,return_value=viewer) as factory, \
                     patch.object(native,'teleop_loop',side_effect=loop_error) as loop, \
                     patch.object(native,'init_logging'), patch('builtins.input',side_effect=AssertionError('No start prompt')):
                    if failure in ('connect','loop','viewer'):
                        with self.assertRaises(RuntimeError):
                            inspect.unwrap(native.teleoperate)(config)
                    else:
                        inspect.unwrap(native.teleoperate)(config)
                    factory.assert_called_once_with(robot,True,8765,True)
                    if failure not in ('connect','viewer'):
                        self.assertIs(loop.call_args.kwargs['follower_viewer'],viewer)
                self.assertFalse(robot.is_connected)
                self.assertFalse(leader.is_connected)
                self.assertEqual(viewer.closed,failure!='viewer')


if __name__ == '__main__':
    unittest.main()
