"""Regression checks for shared geometry and detection postprocessing; no hardware."""
import unittest
import cv2
import numpy as np
from board_calibration import load_calibration, rectify_frame
from detection_utils import filter_stone_boxes, suppress_duplicate_boxes


class VisionTests(unittest.TestCase):
    def test_calibrated_corners_map_to_grid_extents(self):
        config = load_calibration()
        points = np.float32([config["source_points"]])
        result = cv2.perspectiveTransform(points, np.array(config["homography"]))[0]
        np.testing.assert_allclose(result, [[50, 50], [850, 50], [850, 850], [50, 850]], atol=.01)

    def test_mismatched_camera_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs from calibration"):
            rectify_frame(np.zeros((20, 30, 3), dtype=np.uint8), load_calibration())

    def test_no_three_object_limit(self):
        boxes = [{"box": [50+c*100, 50+r*100, 115+c*100, 115+r*100], "score": .9}
                 for r in range(3) for c in range(3)]
        self.assertEqual(len(suppress_duplicate_boxes(filter_stone_boxes(boxes))), 9)

    def test_board_false_positive_and_duplicate_removed(self):
        boxes = [{"box": [0, 0, 900, 900], "score": .95},
                 {"box": [100, 100, 170, 170], "score": .8},
                 {"box": [102, 102, 172, 172], "score": .6}]
        result = suppress_duplicate_boxes(filter_stone_boxes(boxes))
        self.assertEqual(result, [boxes[1]])


if __name__ == "__main__":
    unittest.main()
