"""Baseline-based occupancy tracking with conservative black/white classification."""

from collections import deque

import cv2
import numpy as np

EMPTY, BLACK, WHITE, UNKNOWN = 0, 1, 2, 3
COLOR_NAMES = {EMPTY: "empty", BLACK: "black", WHITE: "white", UNKNOWN: "unknown"}
COLOR_CODES = {EMPTY: "-", BLACK: "B", WHITE: "W", UNKNOWN: "?"}


class StoneTracker:
    def __init__(self, grid, image_side=800, settle_seconds=0.6):
        self.side = 400
        self.grid = np.asarray(grid, dtype=np.float32) * (self.side / image_side)
        self.size = len(grid)
        self.radius = max(4, round(float(self.grid[0, 1, 0] - self.grid[0, 0, 0]) * 0.36))
        self.masks = []
        self.core_masks = []
        yy, xx = np.ogrid[:self.side, :self.side]
        for x, y in self.grid.reshape(-1, 2):
            self.masks.append((xx - x) ** 2 + (yy - y) ** 2 <= self.radius ** 2)
            self.core_masks.append((xx - x) ** 2 + (yy - y) ** 2 <= max(3, self.radius * 0.48) ** 2)
        self.union = np.logical_or.reduce(self.masks)
        lo = np.floor(self.grid[0, 0]).astype(int)
        hi = np.ceil(self.grid[-1, -1]).astype(int)
        self.board_mask = np.zeros((self.side, self.side), dtype=bool)
        self.board_mask[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = True
        self.background_mask = self.board_mask & ~self.union
        self.reference = None
        self.previous = None
        self.stable_since = None
        self.pending_since = None
        self.pending = None
        self.capture = False
        self.samples = []
        self.settle_seconds = settle_seconds
        self.occupied = np.zeros((self.size, self.size), dtype=bool)
        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        self.core_brightness = np.zeros_like(self.board, dtype=float)
        self.brightness_ratio = np.zeros_like(self.board, dtype=float)
        self.events = deque(maxlen=100)
        self.message = "Remove ALL stones, then press B to capture empty board."
        self.paused = True
        self.fractions = np.zeros_like(self.occupied, dtype=float)
        self.circle_evidence = np.zeros_like(self.occupied)
        self.background_change = 0.0

    def start_capture(self):
        self.capture = True
        self.reference = None
        self.previous = None
        self.stable_since = None
        self.pending_since = None
        self.pending = None
        self.samples.clear()
        self.events.clear()
        self.occupied[:] = False
        self.board[:] = EMPTY
        self.fractions[:] = 0
        self.circle_evidence[:] = False
        self.core_brightness[:] = 0
        self.brightness_ratio[:] = 0
        self.background_change = 0.0
        self.message = "Capturing EMPTY board: keep hands out and camera still..."
        self.paused = True

    def update(self, image, now):
        raw = cv2.cvtColor(cv2.resize(image, (self.side, self.side)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(raw, (5, 5), 0).astype(np.float32)
        moving = False
        if self.previous is not None:
            temporal = gray - self.previous
            temporal -= np.median(temporal[self.background_mask])
            moving = np.mean(abs(temporal[self.board_mask]) > 10) > 0.015
        self.previous = gray
        if moving or self.stable_since is None:
            self.stable_since = now
        if moving:
            self.samples.clear()
            self.pending = None
            self.pending_since = None
            self.paused = True
            self.message = "Motion: waiting for the board to settle."
            return
        if now - self.stable_since < self.settle_seconds:
            self.paused = True
            self.message = "Waiting for a stable image..."
            return
        if self.capture:
            self.samples.append(gray.copy())
            self.message = f"Capturing EMPTY board: {len(self.samples)}/12 frames"
            if len(self.samples) >= 12:
                self.reference = np.median(np.stack(self.samples), axis=0)
                self.samples.clear()
                self.capture = False
                self.message = "Empty board captured. Place one stone, then remove your hand."
            return
        if self.reference is None:
            self.message = "Remove ALL stones, then press B to capture empty board."
            return
        delta = gray - self.reference
        exposure_offset = np.median(delta[self.background_mask])
        delta -= exposure_offset
        changed = abs(delta) > 9
        # Large changes outside intersection patches usually mean a hand,
        # moving board, shadows or a lighting change. Do not commit that frame.
        self.background_change = float(np.mean(changed[self.background_mask]))
        if self.background_change > 0.055:
            self.paused = True
            self.pending = None
            self.pending_since = None
            self.message = "Large scene change: remove hands; if board moved, A then B."
            return
        fractions = np.array([np.mean(changed[mask]) for mask in self.masks]).reshape(self.size, self.size)
        self.fractions = fractions
        self.circle_evidence = self._new_circles(raw, delta)
        # Hysteresis avoids flickering at the occupancy threshold.
        occupancy = np.where(self.occupied, fractions > 0.16, fractions > 0.26) | self.circle_evidence
        candidate = self._classify(gray - exposure_offset, occupancy)
        self.paused = False
        if np.array_equal(candidate, self.board):
            self.pending = None
            self.pending_since = None
            self.message = (f"Live: Black {int(np.sum(self.board == BLACK))} | "
                            f"White {int(np.sum(self.board == WHITE))} | "
                            f"Unknown {int(np.sum(self.board == UNKNOWN))}")
            return
        if self.pending is None or not np.array_equal(candidate, self.pending):
            self.pending = candidate.copy()
            self.pending_since = now
        if now - self.pending_since < self.settle_seconds:
            self.message = "Confirming changed intersections..."
            return
        additions = np.argwhere((candidate != EMPTY) & ~self.occupied)
        removals = np.argwhere(self.occupied & (candidate == EMPTY))
        recolored = np.argwhere(self.occupied & (candidate != EMPTY) & (candidate != self.board))
        for kind, changes in (("added", additions), ("removed", removals), ("color_changed", recolored)):
            for row, col in changes:
                coordinate = f"{chr(65 + col)}{row + 1}"
                value = self.board[row, col] if kind == "removed" else candidate[row, col]
                event = {"time": float(now), "action": kind, "coordinate": coordinate,
                         "color": COLOR_NAMES[int(value)]}
                if kind == "color_changed":
                    event["previous_color"] = COLOR_NAMES[int(self.board[row, col])]
                self.events.append(event)
        self.board[:] = candidate
        self.occupied[:] = candidate != EMPTY
        self.pending = None
        self.pending_since = None
        parts = [f"+{chr(65 + col)}{row + 1}:{COLOR_NAMES[int(candidate[row, col])]}" for row, col in additions]
        parts += [f"-{chr(65 + col)}{row + 1}" for row, col in removals]
        parts += [f"{chr(65 + col)}{row + 1}:{COLOR_NAMES[int(candidate[row, col])]}" for row, col in recolored]
        self.message = "Confirmed: " + ", ".join(parts)

    def _classify(self, gray, occupied):
        """Use the stone core to avoid its shadow and small specular highlights.

        Brightness is exposure-corrected to the empty reference. Ratios leave a
        deliberate uncertain band rather than labeling every object as white.
        """
        result = np.zeros_like(self.board)
        self.core_brightness[:] = 0
        self.brightness_ratio[:] = 0
        for row, col in np.argwhere(occupied):
            mask = self.core_masks[row * self.size + col]
            current = float(np.median(gray[mask]))
            baseline = float(np.median(self.reference[mask]))
            ratio = current / max(baseline, 1)
            self.core_brightness[row, col] = current
            self.brightness_ratio[row, col] = ratio
            if ratio < 0.65 and baseline - current > 28:
                result[row, col] = BLACK
            elif ratio > 0.82 and current > 70:
                result[row, col] = WHITE
            else:
                result[row, col] = UNKNOWN
        return result

    def descriptions(self):
        return [f"{chr(65 + col)}{row + 1}:{COLOR_CODES[int(self.board[row, col])]}"
                for row, col in np.argwhere(self.occupied)]

    def _new_circles(self, gray, delta):
        """A faint white stone can add a rim without changing its interior."""
        evidence = np.zeros_like(self.occupied)
        spacing = float(self.grid[0, 1, 0] - self.grid[0, 0, 0])
        enhanced = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 1)
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, 1, spacing * 0.65,
            param1=65, param2=max(12, spacing * 0.4),
            minRadius=max(3, round(spacing * 0.21)),
            maxRadius=max(5, round(spacing * 0.43)))
        if circles is None:
            return evidence
        yy, xx = np.ogrid[:self.side, :self.side]
        # Accept both OpenCV 4's (1, N, 3) and OpenCV 5's (N, 3).
        for x, y, radius in np.asarray(circles).reshape(-1, 3):
            distances = np.linalg.norm(self.grid - [x, y], axis=2)
            row, col = np.unravel_index(distances.argmin(), distances.shape)
            if distances[row, col] > spacing * 0.28:
                continue
            rim = abs(np.sqrt((xx - x) ** 2 + (yy - y) ** 2) - radius) < 2
            # Require a NEW rim relative to the empty board. Fixed circular
            # texture, lighting offsets and existing marks are not sufficient.
            difference = abs(delta[rim])
            if difference.size and np.mean(difference > 3) > 0.32 and difference.mean() > 3:
                evidence[row, col] = True
        return evidence

    def draw(self, image, grid):
        for row, col in np.argwhere(self.occupied):
            value = int(self.board[row, col])
            color = {BLACK: (70, 230, 100), WHITE: (255, 230, 100), UNKNOWN: (0, 180, 255)}[value]
            if self.paused:
                color = (0, 160, 255)
            description = f"{chr(65 + col)}{row + 1} {COLOR_CODES[value]}"
            x, y = np.rint(grid[row, col]).astype(int)
            neighbor = grid[row, col + 1] if col + 1 < self.size else grid[row, col - 1]
            radius = max(5, round(float(np.linalg.norm(grid[row, col] - neighbor)) * 0.36))
            cv2.circle(image, (x, y), radius, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)
            origin = (x + radius + 3, y - 5)
            cv2.putText(image, description, origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, description, origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return image

    def state(self):
        return {"size": self.size, "baseline_ready": self.reference is not None,
                "paused": bool(self.paused), "occupancy": self.occupied.astype(int).tolist(),
                "board": self.board.tolist(), "board_values": COLOR_NAMES,
                "core_brightness": self.core_brightness.tolist(),
                "brightness_ratio": self.brightness_ratio.tolist(),
                "message": self.message, "background_change": self.background_change,
                "changed_fraction": self.fractions.tolist(),
                "new_circle": self.circle_evidence.astype(int).tolist(),
                "events": list(self.events)}
