"""Detect a mostly top-down board, including faint embossed white grids."""

import cv2
import numpy as np


def _families(gray, thresholds=(45, 130)):
    h, w = gray.shape
    edges = cv2.Canny(gray, *thresholds)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 720, 65,
                              minLineLength=min(h, w) * 0.23,
                              maxLineGap=18)
    groups = [[], []]
    if segments is None:
        return groups
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4).astype(float):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1]) / length
        if abs(dy) < abs(dx) * 0.6:
            if line[1] < 0:
                line *= -1
            pos = -(line[0] * w / 2 + line[2]) / line[1]
            family = 0
        elif abs(dx) < abs(dy) * 0.6:
            if line[0] < 0:
                line *= -1
            pos = -(line[1] * h / 2 + line[2]) / line[0]
            family = 1
        else:
            continue
        groups[family].append((pos, line, length))
    merged = []
    for group in groups:
        clusters = []
        for item in sorted(group, key=lambda item: item[0]):
            if not clusters or item[0] - clusters[-1][0][0] > 5:
                clusters.append([item])
            else:
                clusters[-1].append(item)
        lines = []
        for cluster in clusters:
            weights = [item[2] for item in cluster]
            lines.append((np.average([item[0] for item in cluster], weights=weights),
                          np.average([item[1] for item in cluster], axis=0, weights=weights)))
        merged.append(lines)
    return merged


def _bounds(lines, size):
    if len(lines) < size - 1:
        return []
    positions = np.array([item[0] for item in lines])
    candidates = []
    for start in range(len(lines)):
        for end in range(start + size - 2, min(len(lines), start + size + 2)):
            spacing = (positions[end] - positions[start]) / (size - 1)
            if spacing < 12:
                continue
            expected = np.linspace(positions[start], positions[end], size)
            errors = np.min(abs(expected[:, None] - positions[None, :]), axis=1) / spacing
            if np.count_nonzero(errors < 0.42) < size - 1:
                continue
            # A 15-line crop of a 19-line grid is not a complete board.
            outside = np.r_[positions[:start], positions[end + 1:]]
            if len(outside) and min(np.min(abs(outside - (expected[0] - spacing))),
                                    np.min(abs(outside - (expected[-1] + spacing)))) < spacing * 0.3:
                continue
            candidates.append((float(np.mean(errors)), lines[start][1], lines[end][1]))
    return sorted(candidates, key=lambda item: item[0])[:6]


def _intersection(a, b):
    point = np.cross(a, b)
    if abs(point[2]) < 1e-6:
        return None
    return point[:2] / point[2]


def _grid_score(gray, corners, size):
    # Verify actual dark lines at every expected row AND column after warping.
    margin, step = 16, 24
    end = margin + (size - 1) * step
    target = np.float32([[margin, margin], [end, margin], [end, end], [margin, end]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    warped = cv2.warpPerspective(gray, matrix, (end + margin + 1, end + margin + 1))
    dark = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 19, 7) / 255.0
    scores = []
    off_scores = []
    for index in range(size):
        p = margin + index * step
        scores.extend((np.max(dark[p - 3:p + 4, margin:end + 1], axis=0).mean(),
                       np.max(dark[margin:end + 1, p - 3:p + 4], axis=1).mean()))
        if index < size - 1:
            mid = p + step // 2
            off_scores.extend((dark[mid, margin:end + 1].mean(),
                               dark[margin:end + 1, mid].mean()))
    if min(scores) < 0.48 or np.mean(scores) < 0.70:
        return 0.0
    return float(np.mean(scores) - np.mean(off_scores))


def _detect_gray(gray, sizes, thresholds=(45, 130)):
    """Return (TL/TR/BR/BL corners, line count, score), or None if uncertain."""
    horizontal, vertical = _families(gray, thresholds)
    h, w = gray.shape
    best = None
    for size in sizes:
        for _, top, bottom in _bounds(horizontal, size):
            for _, left, right in _bounds(vertical, size):
                points = [_intersection(a, b) for a, b in
                          ((top, left), (top, right), (bottom, right), (bottom, left))]
                if any(point is None for point in points):
                    continue
                corners = np.float32(points)
                if (not np.isfinite(corners).all() or not cv2.isContourConvex(corners)
                        or cv2.contourArea(corners, oriented=True) < h * w * 0.08
                        or np.any(corners < 0) or np.any(corners[:, 0] >= w)
                        or np.any(corners[:, 1] >= h)):
                    continue
                score = _grid_score(gray, corners, size)
                if score > 0.48 and (best is None or score > best[2]):
                    best = corners.tolist(), size, score
    return best


def detect_board(frame, sizes=(9, 15, 19)):
    """Return outer intersections and INTERSECTION count (8 cells = 9 lines)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = _detect_gray(gray, sizes)
    if result is not None:
        return result
    # White embossed lines can be invisible to fixed Canny thresholds. Local
    # contrast enhancement preserves geometry while revealing their shadows.
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    result = _detect_gray(enhanced, sizes)
    if result is not None:
        return result
    result = _detect_gray(enhanced, sizes, thresholds=(20, 60))
    if result is not None:
        return result
    enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    return _detect_gray(enhanced, sizes)
