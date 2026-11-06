"""Install the read-only viewer hook into this environment's native CLI.

Run again after reinstalling LeRobot. No USB connection or motor writes.
"""
import ast
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
MARKER = '# so-arm-101: native teleoperation viewer'
RETRY_MARKER = '# so-arm-101: bounded read retries'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unsupported LeRobot teleoperate source; no files changed: '+old[:70])
    return source.replace(old,new,1)


def refresh_project_path(source):
    """A copied/moved environment must import this checkout's viewer module."""
    prefix = MARKER+'\n'
    if source.count(prefix) != 1:
        raise RuntimeError('Unsupported viewer hook marker; no files changed')
    old = source.split(prefix,1)[1].splitlines()[0]
    if not old.startswith('sys.path.insert(0, ') or not old.endswith(')'):
        raise RuntimeError('Unsupported viewer import path; no files changed')
    return replace_once(source,old+'\n',f'sys.path.insert(0, {str(ROOT)!r})\n')


def add_read_retries(source):
    if RETRY_MARKER in source:
        return source
    replacements = {
        'from lerobot_viewer import open_viewer, publish_observation, disconnect_device\n':
            'from lerobot_viewer import open_viewer, publish_observation, disconnect_device, read_with_retry\n'+RETRY_MARKER+'\n',
        '        obs = robot.get_observation()\n':
            '        obs = read_with_retry(robot.get_observation, "Follower observation")\n',
        '        raw_action = teleop.get_action()\n':
            '        raw_action = read_with_retry(teleop.get_action, "Leader action")\n',
    }
    for old,new in replacements.items():
        source = replace_once(source,old,new)
    ast.parse(source)
    return source


def patched_source(source):
    if MARKER in source:
        source = add_read_retries(refresh_project_path(source))
        ast.parse(source)
        return source

    def replace(old, new):
        nonlocal source
        source = replace_once(source,old,new)

    replace('import logging\n',
                 'import logging\nfrom contextlib import ExitStack\nimport sys\n'
                 +MARKER+'\n'
                 +f'sys.path.insert(0, {str(ROOT)!r})\n'
                 +'from lerobot_viewer import open_viewer, publish_observation, disconnect_device\n')
    replace('    fps: int = 60\n',
                 '    fps: int = 60\n'
                 '    follower_viewer: bool = True\n'
                 '    follower_viewer_port: int = 8765\n'
                 '    follower_viewer_browser: bool = True\n')
    replace('    display_compressed_images: bool = False,\n):',
                 '    display_compressed_images: bool = False,\n    follower_viewer=None,\n):')
    replace('        _ = robot.send_action(robot_action_to_send)\n',
                 '        sent_action = robot.send_action(robot_action_to_send)\n'
                 '        publish_observation(follower_viewer, robot, obs, sent_action)\n')
    start = source.index('    teleop.connect()\n    robot.connect()\n')
    end = source.index('\n\ndef main():', start)
    old = source[start:end]
    # Preserve the native loop arguments and action processors verbatim.
    loop_start = old.index('        teleop_loop(')
    loop_end = old.index('    except KeyboardInterrupt:')
    call = old[loop_start:loop_end]
    call = call.replace('            display_compressed_images=display_compressed_images,\n',
                        '            display_compressed_images=display_compressed_images,\n'
                        '            follower_viewer=viewer,\n')
    call = ''.join('    '+line if line.strip() else line for line in call.splitlines(keepends=True))
    new = (
        '    with ExitStack() as resources:\n'
        '        if cfg.display_data:\n'
        '            resources.callback(shutdown_visualization, cfg.display_mode)\n'
        '        viewer = open_viewer(robot, cfg.follower_viewer, cfg.follower_viewer_port, cfg.follower_viewer_browser)\n'
        '        if viewer is not None:\n'
        '            resources.callback(viewer.close)\n'
        '        resources.callback(disconnect_device, teleop)\n'
        '        resources.callback(disconnect_device, robot)\n'
        '        try:\n'
        '            teleop.connect()\n'
        '            robot.connect()\n'
        +call+
        '        except KeyboardInterrupt:\n'
        '            pass\n'
    )
    source = source[:start]+new+source[end:]
    ast.parse(source)
    return add_read_retries(source)


def main():
    spec = find_spec('lerobot.scripts.lerobot_teleoperate')
    if spec is None or spec.origin is None:
        raise RuntimeError('LeRobot is not installed in this Python environment')
    path = Path(spec.origin)
    source = path.read_text(encoding='utf-8')
    updated = patched_source(source)
    if updated == source:
        print('Native LeRobot viewer hook already installed:', path)
        return
    backup = ROOT/'logs'/'lerobot_teleoperate.original.py'
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        backup.write_text(source, encoding='utf-8')
    # Replace only after the full patch is validated and written successfully.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(updated)
        os.replace(temporary_path,path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print('Installed native LeRobot viewer hook:', path)


if __name__ == '__main__':
    main()
