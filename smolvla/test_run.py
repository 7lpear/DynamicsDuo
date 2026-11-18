"""Archive preparation regressions; no model download, training, or robot access."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


spec = importlib.util.spec_from_file_location("smolvla_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RecordingArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        (self.base / "datasets").mkdir()
        self.addCleanup(patch.stopall)
        patch.object(runner, "BASE", self.base).start()
        (self.base / "datasets/manifest.json").write_text(json.dumps({
            "archive": "known.zip", "sha256": "not-the-checksum",
            "total_episodes": 18, "total_frames": 10794, "fps": 30,
        }))

    def archive(self, name="gomoku_white_center_50_all", extra=None, rootless=False):
        archive = self.base / "datasets" / f"{name}.zip"
        prefix = "" if rootless else name + "/"
        info = {"total_episodes": 50, "total_frames": 24223, "fps": 30,
                "features": {"observation.images.top": {"dtype": "video"}}}
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(prefix + "meta/info.json", json.dumps(info))
            zipped.writestr(prefix + "data/", "")
            zipped.writestr(prefix + "videos/", "")
            zipped.writestr(prefix + "meta/._info.json", "macOS metadata")
            zipped.writestr("__MACOSX/._info.json", "macOS metadata")
            if extra:
                zipped.writestr(*extra)
        return archive

    def test_custom_fifty_episode_archive_and_training_identity(self):
        archive = self.archive()
        output = runner.prepare_dataset(archive)
        self.assertEqual(output.name, "gomoku_white_center_50_all")
        self.assertEqual(runner.read_info(output)["total_frames"], 24223)
        self.assertFalse(list(output.rglob("._*")))
        receipt = json.loads((output / "source_archive.json").read_text())
        self.assertEqual(receipt["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        args = argparse.Namespace(dataset=output, output=self.base / "model", steps=10, batch_size=1)
        command = runner.training_command(args, "cuda")
        self.assertIn("--dataset.repo_id=local/gomoku_white_center_50_all", command)
        self.assertIn(f"--dataset.root={output.resolve()}", command)
        self.assertEqual(runner.prepare_dataset(archive), output)

    def test_known_archive_still_requires_manifest_checksum(self):
        self.archive("known")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            runner.prepare_dataset()

    def test_rejects_path_traversal(self):
        archive = self.archive(extra=("../escape", "unsafe"))
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            runner.prepare_dataset(archive)
        self.assertFalse((self.base / "escape").exists())

    def test_rejects_multiple_datasets(self):
        archive = self.archive(extra=("other/meta/info.json", "{}"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.prepare_dataset(archive)

    def test_refuses_changed_archive_over_existing_dataset(self):
        archive = self.archive()
        output = runner.prepare_dataset(archive)
        with zipfile.ZipFile(archive, "a") as zipped:
            zipped.writestr("extra.txt", "different archive")
        with self.assertRaisesRegex(FileExistsError, "another ZIP"):
            runner.prepare_dataset(archive)
        self.assertTrue((output / "meta/info.json").is_file())

    def test_supports_zip_with_metadata_at_top_level(self):
        archive = self.archive(rootless=True)
        output = runner.prepare_dataset(archive)
        self.assertTrue((output / "meta/info.json").is_file())
        self.assertEqual(output.name, archive.stem)


if __name__ == "__main__":
    unittest.main()
