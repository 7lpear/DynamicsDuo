"""Portable SmolVLA dataset preparation, checks, training, and model export.

Run from any directory. Only ``train`` starts training. No command connects to a robot.
"""

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import stat
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile


BASE = Path(__file__).resolve().parent
DATASET_NAME = "gomoku_white_center_18_clean"
DATASET = BASE / "data" / DATASET_NAME


def read_info(root, expected=None):
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    for directory in ("data", "meta", "videos"):
        if not (root / directory).is_dir():
            raise ValueError(f"Missing dataset directory: {root / directory}")
    for key in ("total_episodes", "total_frames", "fps"):
        if not isinstance(info.get(key), (int, float)) or info[key] <= 0:
            raise ValueError(f"Invalid dataset metadata: {key}={info.get(key)}")
    if expected:
        for key in ("total_episodes", "total_frames", "fps"):
            if info[key] != expected[key]:
                raise ValueError(f"Dataset {key} mismatch: expected {expected[key]}, found {info[key]}")
    if "observation.images.top" not in info["features"]:
        raise ValueError("Dataset must contain the top camera")
    return info


def prepare_dataset(archive_path=None):
    manifest = json.loads((BASE / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    default_archive = BASE / "datasets" / manifest["archive"]
    archive = Path(archive_path).expanduser().resolve() if archive_path else default_archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Copy the recording ZIP from the recording computer to {archive}; see smolvla/README.md")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    known_archive = archive == default_archive.resolve()
    if known_archive and digest != manifest["sha256"]:
        raise ValueError("Dataset archive checksum mismatch; fetch the branch again")
    print(f"Archive SHA-256: {digest}")
    data_parent = BASE / "data"
    data_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            members = []
            for entry in zipped.infolist():
                parts = PurePosixPath(entry.filename).parts
                if any(part in ("__MACOSX", ".DS_Store") or part.startswith("._") for part in parts):
                    continue
                if "\\" in entry.filename or any(part == ".." or ":" in part for part in parts):
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Archive symlink is not supported: {entry.filename}")
                members.append(entry)
            infos = [PurePosixPath(e.filename) for e in members
                     if not e.is_dir() and PurePosixPath(e.filename).parts[-2:] == ("meta", "info.json")]
            if len(infos) != 1:
                raise ValueError(f"ZIP must contain exactly one dataset meta/info.json; found {len(infos)}")
            prefix = infos[0].parent.parent
            name = prefix.name if prefix != PurePosixPath(".") else archive.stem
            if not name or name in (".", ".."):
                raise ValueError("Cannot determine dataset folder name")
            output = data_parent / name
            expected = json.loads(zipped.read(str(infos[0])))
            if output.exists():
                read_info(output, expected)
                receipt = output / "source_archive.json"
                if receipt.is_file():
                    if json.loads(receipt.read_text())["sha256"] != digest:
                        raise FileExistsError(f"Existing dataset came from another ZIP: {output}")
                elif not known_archive:
                    raise FileExistsError(f"Existing dataset has no archive receipt; choose a different ZIP root or move it aside: {output}")
                print(f"Dataset already prepared: {output}")
                return output
            zipped.extractall(destination, members=members)
        extracted = destination / prefix
        info = read_info(extracted, manifest if known_archive else None)
        (extracted / "source_archive.json").write_text(json.dumps({
            "archive": archive.name,
            "sha256": digest,
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "known_manifest": known_archive,
        }, indent=2) + "\n")
        shutil.move(str(extracted), str(output))
    print(f"Dataset prepared: {output}")
    print(f"Episodes: {info['total_episodes']}, frames: {info['total_frames']}")
    return output


def resolve_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_runtime(requested):
    import torch

    installed = version("lerobot")
    if installed != "0.7.0":
        raise ValueError(f"Expected LeRobot 0.7.0, found {installed}")
    device = resolve_device(requested)
    print(f"Python: {sys.version.split()[0]} | LeRobot: {installed} | PyTorch: {torch.__version__}")
    print(f"Device: {device} | CUDA runtime: {torch.version.cuda}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch installation")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this Python/environment")
    tensor = torch.randn(128, 128, device=device)
    result = (tensor @ tensor).sum().item()
    print(f"Device computation OK: {result:.3f}")
    return device


def check_dataset(root):
    import pyarrow.parquet as pq
    import av

    info = read_info(root)
    table = pq.read_table(sorted((root / "meta" / "episodes").rglob("*.parquet")))
    indices = table["episode_index"].to_pylist()
    if sorted(indices) != list(range(info["total_episodes"])):
        raise ValueError(f"Unexpected episode indices: {indices}")
    if sum(table["length"].to_pylist()) != info["total_frames"]:
        raise ValueError("Episode lengths do not match total_frames")
    data = pq.read_table(sorted((root / "data").rglob("*.parquet")))
    if len(data) != info["total_frames"]:
        raise ValueError("Parquet frame count mismatch")
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    print(f"Tasks: {tasks}")
    for video in sorted((root / "videos").rglob("*.mp4")):
        with av.open(str(video)) as container:
            frame = next(container.decode(video=0))
            print(f"Video decode OK: {video.name} ({frame.width}x{frame.height})")
    print(f"Dataset OK: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps")


def training_command(args, device):
    root = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.path=lerobot/smolvla_base",
        f"--dataset.repo_id=local/{root.name}",
        f"--dataset.root={root}",
        "--dataset.video_backend=pyav",
        '--rename_map={"observation.images.top":"observation.images.camera1"}',
        "--policy.input_features=null",
        "--policy.load_vlm_weights=false",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        "--num_workers=0",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--log_freq=10",
        f"--save_freq={min(1000, args.steps)}",
        "--seed=7",
        "--trackio.enable=false",
        f"--output_dir={output}",
        "--job_name=gomoku_white_center_smolvla",
    ]


def train(args):
    root = args.dataset.expanduser().resolve()
    read_info(root)
    if args.dry_run:
        # Can be inspected without LeRobot/PyTorch installed. Never auto-select an unchecked GPU.
        device = "cuda" if args.device == "auto" else args.device
    else:
        device = check_runtime(args.device)
        if args.output.exists():
            raise ValueError(f"Output already exists; choose a new --output: {args.output}")
    command = training_command(args, device)
    print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
        print(f"Training finished. Export with: python {__file__} export --run {args.output}")


def export_model(args):
    run = args.run.expanduser().resolve()
    candidates = [p for p in (run / "checkpoints").glob("*/pretrained_model")
                  if p.parent.name.isdigit() and (p / "model.safetensors").is_file()]
    if not candidates:
        raise ValueError(f"No completed pretrained_model checkpoint in {run}")
    model = max(candidates, key=lambda p: int(p.parent.name))
    for filename in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (model / filename).is_file():
            raise ValueError(f"Incomplete checkpoint, missing {filename}")
    archive = args.output.expanduser().resolve()
    if archive.exists():
        raise FileExistsError(f"Export already exists: {archive}")
    if archive.suffix != ".zip":
        raise ValueError("Export output must end in .zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=model.parent, base_dir=model.name)
    print(f"Exported checkpoint {model.parent.name}: {archive}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Unpack a recording ZIP; defaults to the verified 18-episode archive")
    preparation.add_argument("--archive", type=Path, help="Recording ZIP, e.g. datasets/gomoku_white_center_50_all.zip")
    check = commands.add_parser("check", help="Check packages, actual GPU computation, and local dataset")
    check.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    check.add_argument("--dataset", type=Path, default=DATASET)
    training = commands.add_parser("train", help="Fine-tune the original SmolVLA base checkpoint")
    training.add_argument("--dataset", type=Path, default=DATASET)
    training.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    training.add_argument("--steps", type=int, default=5000)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--output", type=Path, default=BASE / "outputs" / datetime.now().strftime("white_center_%Y%m%d_%H%M%S"))
    training.add_argument("--dry-run", action="store_true")
    exporting = commands.add_parser("export", help="Zip the latest completed checkpoint, including processors")
    exporting.add_argument("--run", type=Path, required=True)
    exporting.add_argument("--output", type=Path, default=BASE / "exports" / "gomoku_smolvla.zip")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_dataset(args.archive)
    elif args.command == "check":
        check_runtime(args.device)
        check_dataset(args.dataset.expanduser().resolve())
    elif args.command == "train":
        train(args)
    elif args.command == "export":
        export_model(args)


if __name__ == "__main__":
    main()
