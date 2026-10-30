"""Synthetic geometry tests; real camera/lighting validation is still needed."""

import unittest

import cv2
import numpy as np

from board_detection import detect_board


class BoardDetectionTests(unittest.TestCase):
    def test_grids_and_perspective(self):
        for size in (7, 8, 9, 15, 19):
            board = np.full((640, 640, 3), (120, 180, 220), np.uint8)
            for p in np.linspace(50, 590, size).round().astype(int):
                cv2.line(board, (50, p), (590, p), (35, 35, 35), 2)
                cv2.line(board, (p, 50), (p, 590), (35, 35, 35), 2)
            for corners in ([[100, 50], [740, 50], [740, 690], [100, 690]],
                            [[150, 65], [750, 105], [720, 670], [100, 690]]):
                with self.subTest(size=size, corners=corners):
                    matrix = cv2.getPerspectiveTransform(
                        np.float32([[0, 0], [639, 0], [639, 639], [0, 639]]),
                        np.float32(corners))
                    frame = cv2.warpPerspective(board, matrix, (900, 740),
                                                borderValue=(190, 190, 190))
                    result = detect_board(frame, sizes=(size,))
                    self.assertIsNotNone(result)
                    self.assertEqual(result[1], size)
                    expected = cv2.perspectiveTransform(np.float32([
                        [[50, 50]], [[590, 50]], [[590, 590]], [[50, 590]]]), matrix).reshape(4, 2)
                    error = np.linalg.norm(np.array(result[0]) - expected, axis=1).max()
                    self.assertLess(error, 6)

    def test_faint_white_grid(self):
        frame = np.full((740, 900, 3), 185, np.uint8)
        for p in np.linspace(100, 640, 9).round().astype(int):
            cv2.line(frame, (150, p), (690, p), (172, 172, 172), 2)
        for p in np.linspace(150, 690, 9).round().astype(int):
            cv2.line(frame, (p, 100), (p, 640), (172, 172, 172), 2)
        result = detect_board(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 9)

    def test_reject_background_and_board_outline(self):
        frame = np.full((740, 900, 3), 180, np.uint8)
        self.assertIsNone(detect_board(frame))
        cv2.rectangle(frame, (100, 70), (750, 650), (20, 20, 20), 3)
        self.assertIsNone(detect_board(frame))


if __name__ == "__main__":
    unittest.main()
