"""Platform routing and file handling checks without opening cameras."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import cv2
import numpy as np
from camera_io import resolve_backend, open_camera, read_image, write_image, require_gui


class CameraIOTests(unittest.TestCase):
    def test_headless_opencv_is_rejected_before_camera_open(self):
        with patch("camera_io.cv2.getBuildInformation", return_value="  GUI: NONE\n"):
            with self.assertRaisesRegex(RuntimeError, "no GUI support"):
                require_gui()

    def test_platform_defaults(self):
        self.assertEqual(resolve_backend("auto", "win32"), ("dshow", cv2.CAP_DSHOW))
        self.assertEqual(resolve_backend("auto", "darwin"), ("avfoundation", cv2.CAP_AVFOUNDATION))

    def test_wrong_platform_backend_is_rejected(self):
        for name, platform in (("dshow", "darwin"), ("msmf", "darwin"), ("avfoundation", "win32")):
            with self.subTest(name=name, platform=platform), self.assertRaises(ValueError):
                resolve_backend(name, platform)

    def test_failed_open_releases_device(self):
        cap = Mock()
        cap.isOpened.return_value = False
        with patch("camera_io.cv2.VideoCapture", return_value=cap), self.assertRaises(RuntimeError):
            open_camera(3, "auto")
        cap.release.assert_called_once()

    def test_macos_open_uses_avfoundation(self):
        cap = Mock()
        cap.isOpened.return_value = True
        with patch("camera_io.sys.platform", "darwin"), patch("camera_io.cv2.VideoCapture", return_value=cap) as capture:
            self.assertIs(open_camera(1, "auto"), cap)
            capture.assert_called_once_with(1, cv2.CAP_AVFOUNDATION)

    def test_unicode_image_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "오목판_테스트.png"
            expected = np.full((20, 30, 3), 123, dtype=np.uint8)
            write_image(path, expected)
            np.testing.assert_array_equal(read_image(path), expected)


if __name__ == "__main__":
    unittest.main()
