import unittest

import cv2
import numpy as np

from board_camera import grid_points
from stone_tracking import BLACK, WHITE, UNKNOWN, StoneTracker


class StoneTrackingTests(unittest.TestCase):
    def setUp(self):
        self.grid = grid_points(9)
        self.empty = np.full((800, 800, 3), 180, np.uint8)
        for p in np.rint(self.grid[0, :, 0]).astype(int):
            cv2.line(self.empty, (p, 48), (p, 751), (160, 160, 160), 2)
            cv2.line(self.empty, (48, p), (751, p), (160, 160, 160), 2)
        self.tracker = StoneTracker(self.grid)
        self.now = 0.0
        self.tracker.start_capture()
        self.feed(self.empty, 30)
        self.assertIsNotNone(self.tracker.reference)

    def feed(self, frame, count):
        for _ in range(count):
            self.now += 0.1
            self.tracker.update(frame, self.now)

    def stone(self, frame, row, col, value):
        result = frame.copy()
        center = tuple(np.rint(self.grid[row, col]).astype(int))
        cv2.circle(result, center, 23, (value,) * 3, -1)
        return result

    def test_add_remove_and_transient_rejection(self):
        stone = self.stone(self.empty, 4, 4, 35)
        self.feed(stone, 2)
        self.assertFalse(self.tracker.occupied.any())
        self.feed(self.empty, 20)
        self.assertFalse(self.tracker.events)
        self.feed(stone, 25)
        self.assertEqual(int(self.tracker.occupied.sum()), 1)
        self.assertTrue(self.tracker.occupied[4, 4])
        self.assertEqual(self.tracker.events[-1]["coordinate"], "E5")
        self.assertEqual(self.tracker.events[-1]["action"], "added")
        self.feed(self.empty, 25)
        self.assertFalse(self.tracker.occupied.any())
        self.assertEqual(self.tracker.events[-1]["action"], "removed")

    def test_white_stone_and_exposure(self):
        self.feed(np.clip(self.empty.astype(int) + 18, 0, 255).astype(np.uint8), 25)
        self.assertFalse(self.tracker.occupied.any())
        self.feed(self.stone(self.empty, 1, 6, 225), 25)
        self.assertTrue(self.tracker.occupied[1, 6])
        self.assertEqual(self.tracker.board[1, 6], WHITE)

    def test_black_white_at_once_and_export(self):
        stones = self.stone(self.stone(self.empty, 2, 3, 35), 5, 6, 225)
        # Reflection on a black stone should not flip its median core color.
        x, y = np.rint(self.grid[2, 3]).astype(int)
        cv2.circle(stones, (x + 3, y - 3), 4, (245, 245, 245), -1)
        self.feed(stones, 25)
        self.assertEqual(self.tracker.board[2, 3], BLACK)
        self.assertEqual(self.tracker.board[5, 6], WHITE)
        self.assertEqual(int(self.tracker.occupied.sum()), 2)
        state = self.tracker.state()
        self.assertEqual(state['board'][2][3], BLACK)
        self.assertEqual(state['board'][5][6], WHITE)
        self.assertEqual({event['color'] for event in state['events']}, {'black', 'white'})

    def test_uncertain_gray_stone_is_not_forced_to_white(self):
        self.feed(self.stone(self.empty, 4, 4, 140), 25)
        self.assertTrue(self.tracker.occupied[4, 4])
        self.assertEqual(self.tracker.board[4, 4], UNKNOWN)

    def test_color_replacement_and_removal_event(self):
        self.feed(self.stone(self.empty, 4, 4, 35), 25)
        self.assertEqual(self.tracker.board[4, 4], BLACK)
        self.feed(self.stone(self.empty, 4, 4, 225), 25)
        self.assertEqual(self.tracker.board[4, 4], WHITE)
        self.assertEqual(self.tracker.events[-1]['action'], 'color_changed')
        self.assertEqual(self.tracker.events[-1]['previous_color'], 'black')
        self.feed(self.empty, 25)
        self.assertEqual(self.tracker.board[4, 4], 0)
        self.assertEqual(self.tracker.events[-1]['action'], 'removed')
        self.assertEqual(self.tracker.events[-1]['color'], 'white')

    def test_color_uses_local_reference_under_dim_light(self):
        self.tracker.start_capture()
        dim = np.clip(self.empty.astype(int) - 60, 0, 255).astype(np.uint8)
        self.feed(dim, 30)
        stones = self.stone(self.stone(dim, 2, 3, 30), 5, 6, 120)
        # Give the same-brightness white stone a faint rim.
        center = tuple(np.rint(self.grid[5, 6]).astype(int))
        cv2.circle(stones, center, 23, (100, 100, 100), 2, cv2.LINE_AA)
        self.feed(stones, 25)
        self.assertEqual(self.tracker.board[2, 3], BLACK)
        self.assertEqual(self.tracker.board[5, 6], WHITE)

    def test_white_stone_same_brightness_with_faint_rim(self):
        stone = self.empty.copy()
        center = tuple(np.rint(self.grid[3, 3]).astype(int))
        cv2.circle(stone, center, 23, (180, 180, 180), -1, cv2.LINE_AA)
        cv2.circle(stone, center, 23, (160, 160, 160), 2, cv2.LINE_AA)
        self.feed(stone, 25)
        self.assertLess(self.tracker.fractions[3, 3], 0.26)
        self.assertTrue(self.tracker.circle_evidence[3, 3])
        self.assertTrue(self.tracker.occupied[3, 3])
        self.assertEqual(int(self.tracker.occupied.sum()), 1)
        self.feed(self.empty, 25)
        self.assertFalse(self.tracker.occupied.any())
        self.assertFalse(self.tracker.board.any())

    def test_circular_mark_already_on_empty_board_is_not_a_stone(self):
        marked = self.empty.copy()
        center = tuple(np.rint(self.grid[3, 3]).astype(int))
        cv2.circle(marked, center, 23, (160, 160, 160), 2, cv2.LINE_AA)
        self.tracker.start_capture()
        self.feed(marked, 30)
        self.feed(marked, 25)
        self.assertFalse(self.tracker.occupied.any())

    def test_large_occlusion_pauses(self):
        frame = self.empty.copy()
        cv2.rectangle(frame, (250, 100), (580, 600), (80, 110, 160), -1)
        self.feed(frame, 30)
        self.assertTrue(self.tracker.paused)
        self.assertFalse(self.tracker.occupied.any())
        self.assertFalse(self.tracker.events)

    def test_baseline_reset_clears_state(self):
        self.feed(self.stone(self.empty, 0, 8, 25), 25)
        self.assertTrue(self.tracker.occupied[0, 8])
        self.tracker.start_capture()
        self.assertIsNone(self.tracker.reference)
        self.assertFalse(self.tracker.events)
        self.assertFalse(self.tracker.occupied.any())


if __name__ == "__main__":
    unittest.main()
