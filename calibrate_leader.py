"""Manual SO-101 leader calibration into the teleoperation profile directory."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from calibrate_follower import recalibrate


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--leader-id', default='my_leader')
    parser.add_argument('--calibration-dir', type=Path, default=ROOT/'calibration/teleop_profiles')
    parser.add_argument('--backup-dir', type=Path, default=ROOT/'calibration/robot_backups')
    args = parser.parse_args(argv)
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    leader = SO101Leader(SO101LeaderConfig(port=args.port, id=args.leader_id,
                                         calibration_dir=args.calibration_dir, use_degrees=True))
    # Shared manual routine backs up PC/device values, never powers the arm,
    # and always measures anew rather than reusing an existing file.
    recalibrate(leader, args.backup_dir)
    leader.calibration_fpath.with_suffix('.source.json').write_text(
        json.dumps({'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                    'profile_id':args.leader_id, 'mode':'manual_leader_recalibration'}, indent=2)+'\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('STOP: Leader calibration interrupted. Support the arm and recalibrate before teleoperation.')
        raise SystemExit(1)
    except Exception as error:
        print(f'STOP: {error}')
        raise SystemExit(1)
