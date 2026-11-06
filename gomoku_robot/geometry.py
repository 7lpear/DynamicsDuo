"""Geometry in millimetres; no serial connection or motor commands."""

from dataclasses import dataclass
import math
import re

import numpy as np


def board_size(value):
    if type(value) is not int or not 3 <= value <= 19:
        raise ValueError("board_size must be an integer from 3 to 19")
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def xyz(value, name="xyz_mm"):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3
            or any(isinstance(v, (bool, str)) or v is None for v in value)):
        raise ValueError(f"{name} must contain three measured numbers in mm")
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers in mm")
    return result


def cell(coordinate, size):
    """Return zero-based (column, row), matching gomoku_vision (A1 = top left)."""
    board_size(size)
    if not isinstance(coordinate, str) or not re.fullmatch(r"[A-Sa-s][1-9][0-9]?", coordinate):
        raise ValueError(f"Invalid coordinate: {coordinate!r}")
    col, row = ord(coordinate[0].upper()) - ord("A"), int(coordinate[1:]) - 1
    if not 0 <= col < size or not 0 <= row < size:
        raise ValueError(f"Coordinate {coordinate} is outside the {size}x{size} board")
    return col, row


def template(size=9):
    board_size(size)
    last = chr(64 + size)
    middle = size // 2 + 1
    return {
        "schema_version": 1,
        "setup_id": None,
        "board_size": size,
        "units": "mm",
        "frame": "robot_base",
        "coordinate_convention": "A1=vision_top_left; columns_right; rows_down",
        "measurement_note": "All points are board-surface intersections in ONE robot-base frame; not pixels or joint angles.",
        "max_error_mm": 2.0,
        "fit_points": [{"coordinate": p, "xyz_mm": None}
                       for p in ("A1", f"{last}1", f"{last}{size}", f"A{size}")],
        "validation_points": [{"coordinate": f"{chr(64 + middle)}{middle}", "xyz_mm": None}],
    }


def samples(values, size, name):
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    coordinates, positions = [], []
    for point in values:
        if not isinstance(point, dict):
            raise ValueError(f"Each {name} entry must be an object")
        coordinates.append(cell(point.get("coordinate"), size))
        positions.append(xyz(point.get("xyz_mm"), f"{name}/{point.get('coordinate')}/xyz_mm"))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(f"Duplicate coordinates in {name}")
    return coordinates, np.asarray(positions)


@dataclass
class BoardFrame:
    size: int
    setup_id: str
    basis: np.ndarray  # rows: A1 position, column step, row step
    normal: np.ndarray  # board normal pointing toward base +Z
    report: dict

    def surface(self, coordinate):
        col, row = cell(coordinate, self.size)
        return np.array([1, col, row]) @ self.basis

    def target(self, coordinate, hover_mm=30):
        hover_mm = positive(hover_mm, "hover_mm")
        surface = self.surface(coordinate)
        return {
            "mode": "offline_cartesian_preview",
            "setup_id": self.setup_id,
            "frame": "robot_base", "units": "mm",
            "coordinate": coordinate.upper(),
            "surface_xyz_mm": surface.tolist(),
            "hover_xyz_mm": (surface + self.normal * hover_mm).tolist(),
            "hover_mm": hover_mm,
            "normal": self.normal.tolist(),
        }

    def hover_plan(self, coordinate, start_xyz_mm, hover_mm=30):
        """Raise, traverse, then approach the hover point; never descend to surface."""
        result = self.target(coordinate, hover_mm)
        start = xyz(start_xyz_mm, "start_xyz_mm")
        height = float((start - self.basis[0]) @ self.normal)
        if height <= 0:
            raise ValueError("Start TCP must be above the calibrated board plane")
        travel_height = max(height, result["hover_mm"])
        raised = start + (travel_height - height) * self.normal
        across = np.asarray(result["surface_xyz_mm"]) + travel_height * self.normal
        result["waypoints"] = [
            {"phase": "start", "xyz_mm": start.tolist()},
            {"phase": "raise", "xyz_mm": raised.tolist()},
            {"phase": "traverse", "xyz_mm": across.tolist()},
            {"phase": "hover", "xyz_mm": result["hover_xyz_mm"]},
        ]
        result["checks_not_performed"] = [
            "inverse_kinematics", "joint_limits", "reachability", "collisions",
            "tool_orientation", "payload_and_motor_capacity", "live_board_state",
        ]
        return result


def fit_session(session):
    """Fit four board corners and check independent points before exposing targets."""
    if (not isinstance(session, dict) or type(session.get("schema_version")) is not int
            or session["schema_version"] != 1):
        raise ValueError("Unsupported session schema_version")
    if session.get("units") != "mm" or session.get("frame") != "robot_base":
        raise ValueError("Measurements must use units=mm and frame=robot_base")
    if session.get("coordinate_convention") != "A1=vision_top_left; columns_right; rows_down":
        raise ValueError("Coordinate convention must match gomoku_vision")
    setup_id = session.get("setup_id")
    if not isinstance(setup_id, str) or not setup_id.strip():
        raise ValueError("Set setup_id to identify the fixed robot/board/camera arrangement")
    size = board_size(session.get("board_size"))
    threshold = positive(session.get("max_error_mm"), "max_error_mm")
    fit_cells, fit_xyz = samples(session.get("fit_points"), size, "fit_points")
    check_cells, check_xyz = samples(session.get("validation_points"), size, "validation_points")
    corners = {(0, 0), (size-1, 0), (size-1, size-1), (0, size-1)}
    if not corners.issubset(set(fit_cells)):
        raise ValueError("fit_points must include all four outermost board intersections")
    if not check_cells or set(fit_cells) & set(check_cells):
        raise ValueError("Provide independent validation_points not used in fit_points")
    design = np.array([[1, col, row] for col, row in fit_cells], dtype=float)
    basis, _, rank, _ = np.linalg.lstsq(design, fit_xyz, rcond=None)
    if rank != 3:
        raise ValueError("Degenerate calibration point layout")
    col_length, row_length = np.linalg.norm(basis[1:], axis=1)
    if min(col_length, row_length) < 1:
        raise ValueError("Estimated grid spacing is below 1 mm; check units and measurements")
    # Square-grid checks catch swapped labels and poorly measured corners.
    cosine = float(np.dot(basis[1], basis[2]) / (col_length * row_length))
    angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
    if abs(angle - 90) > 5 or abs(col_length-row_length)/max(col_length, row_length) > .05:
        raise ValueError("Grid is not square within 5 degrees / 5%; check corner labels and measurements")
    normal = np.cross(basis[1], basis[2])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    if normal[2] < math.cos(math.radians(20)):
        raise ValueError("Board must be within 20 degrees of the robot-base XY plane (+Z is up)")
    errors_fit = np.linalg.norm(design @ basis - fit_xyz, axis=1)
    check_design = np.array([[1, col, row] for col, row in check_cells])
    errors_check = np.linalg.norm(check_design @ basis - check_xyz, axis=1)
    max_fit, max_check = float(errors_fit.max()), float(errors_check.max())
    if max_fit > threshold or max_check > threshold:
        raise ValueError(f"Calibration rejected: max fit error={max_fit:.3f} mm, "
                         f"validation error={max_check:.3f} mm (limit={threshold:.3f} mm)")
    # A generous threshold must not make a geometrically collapsed fit acceptable.
    if threshold >= min(col_length, row_length) / 4:
        raise ValueError("max_error_mm must be less than one quarter of the grid spacing")
    report = {
        "setup_id": setup_id, "board_size": size, "units": "mm", "frame": "robot_base",
        "origin_xyz_mm": basis[0].tolist(),
        "column_step_xyz_mm": basis[1].tolist(), "row_step_xyz_mm": basis[2].tolist(),
        "normal": normal.tolist(), "spacing_mm": [float(col_length), float(row_length)],
        "axis_angle_degrees": angle, "max_fit_error_mm": max_fit,
        "max_validation_error_mm": max_check, "max_error_mm": threshold,
        "fit_errors_mm": dict(zip((p["coordinate"] for p in session["fit_points"]), errors_fit.tolist())),
        "validation_errors_mm": dict(zip((p["coordinate"] for p in session["validation_points"]), errors_check.tolist())),
    }
    return BoardFrame(size, setup_id, basis, normal, report)
