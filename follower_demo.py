"""Coordinated SO-101 demo, relative to the starting pose."""
import math
import time

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Relative angles in degrees; gripper displacement is percent of calibrated travel.
POSES = [
    ("Look left / open", [25, -10, 18, -12, 25, 20]),
    ("Look right / close", [-25, 10, -18, 12, -25, -20]),
    ("Reach / turn wrist", [0, -15, 25, -20, 45, 25]),
    ("Fold / turn back", [0, 15, -25, 20, -45, -25]),
    ("Wave left", [20, -8, 12, 25, 35, 15]),
    ("Wave right", [-20, -8, 12, -25, -35, -15]),
    ("Return to start", [0, 0, 0, 0, 0, 0]),
]


def make_wide_plan(start, calibration, range_fraction=0.7):
    """Sweep the central fraction of calibrated travel, with endpoint clearance."""
    if not math.isfinite(range_fraction) or not 0.2 <= range_fraction <= 0.85:
        raise ValueError("Range fraction must be 0.2..0.85")
    centers, widths, factors = {}, {}, {}
    for motor in NAMES:
        cal = calibration[motor]
        span = cal.range_max - cal.range_min
        if span <= 0 or not cal.range_min <= start[motor] <= cal.range_max:
            raise RuntimeError(f"Invalid calibration or starting position: {motor}")
        margin = max(5, round(span * 0.03)) if motor == "gripper" else 114
        low, high = cal.range_min + margin, cal.range_max - margin
        if low >= high:
            raise RuntimeError(f"Insufficient calibrated travel: {motor}")
        centers[motor] = (low + high) / 2
        widths[motor] = (high - low) * range_fraction / 2
        factors[motor] = span / 100 if motor == "gripper" else 4095 / 360
    peaks = [max(abs(offset[i]) for _, offset in POSES) for i in range(len(NAMES))]
    plan = [("Move to range center", {m: round(centers[m]) for m in NAMES})]
    for label, offsets in POSES[:-1]:
        target = {m: round(centers[m] + offsets[i] / peaks[i] * widths[m]) for i, m in enumerate(NAMES)}
        plan.append(("Wide: " + label, target))
    plan.append(("Return to start", dict(start)))
    return plan, factors


def segment_duration(previous, target, factors, speed):
    return max(1.0, 1.875 * max(abs(target[m] - previous[m]) / factors[m] for m in NAMES) / speed)


def make_load_plan(start, calibration):
    """Keep shoulder/elbow/wrist pitch fixed; filter gravity change along the path."""
    from follower_load import GravityModel, normalized
    model = GravityModel()
    baseline = model.torque(normalized(start, calibration))
    # Screening assumption, NOT a manufacturer continuous-torque rating:
    # 20% of the user's 30 kg.cm stall label, with 1.5x model uncertainty.
    screening_limit = 30 * 0.0980665 * 0.20
    uncertainty = 1.5
    exceeded = {m: t for m, t in baseline.items() if t * uncertainty > screening_limit}
    if exceeded:
        details = '; '.join(f'{m}: estimated={t:.4f} Nm, x{uncertainty:g}={t*uncertainty:.4f} Nm > {screening_limit:.4f} Nm'
                            for m,t in exceeded.items())
        print('MODEL CAUTION (not measured overload): '+details)
        print('The old 20% heuristic is advisory. Running only a small feedback-checked verification; continuous capacity is unverified.')
    labeled_max = 30 * 0.0980665
    if any(t * uncertainty >= labeled_max for t in baseline.values()):
        raise RuntimeError('Static model estimate reaches labeled maximum torque; verification refused.')
    factors = {m: (calibration[m].range_max-calibration[m].range_min)/100 if m=='gripper' else 4095/360 for m in NAMES}
    # Visible base/wrist rotation, no commanded extension of the load-bearing links.
    offsets = [('Verify left / open', {'shoulder_pan':8,'wrist_roll':10,'gripper':8}),
               ('Verify right / close', {'shoulder_pan':-8,'wrist_roll':-10,'gripper':0}),
               ('Small wrist wave', {'shoulder_pan':4,'wrist_roll':6,'gripper':5}),
               ('Return to start', {})]
    previous = start
    plan = []
    peaks = baseline.copy()
    for label, delta in offsets:
        accepted = None
        for scale in (1.0, 0.5, 0.25, 0.0):
            target = start.copy()
            for motor, value in delta.items():
                cal = calibration[motor]
                margin = max(5, round((cal.range_max-cal.range_min)*.03)) if motor=='gripper' else 80
                low, high = cal.range_min+margin, cal.range_max-margin
                if low>=high:
                    raise RuntimeError(f'Insufficient calibrated travel: {motor}')
                candidate = round(start[motor]+value*factors[motor]*scale)
                # A limited joint remains at start if the requested direction has no room.
                target[motor] = candidate if low <= candidate <= high else start[motor]
            estimates = []
            for step in range(21):
                raw = {m: previous[m]+(target[m]-previous[m])*step/20 for m in NAMES}
                estimates.append(model.torque(normalized(raw, calibration)))
            # This relative budget does NOT imply that baseline torque is within motor capacity.
            if all(t[m] <= baseline[m]+0.02 and t[m]*uncertainty < labeled_max
                   for t in estimates for m in NAMES):
                accepted = target
                for m in NAMES:
                    peaks[m] = max(peaks[m],*(t[m] for t in estimates))
                break
        if accepted is None:
            raise RuntimeError('No path fits the relative gravity budget')
        plan.append((label, accepted))
        previous = accepted
    print('URDF estimated peak static gravity torque (Nm), no payload or dynamic terms:')
    print(f'Old {screening_limit:.3f} Nm heuristic is advisory; static estimates do not establish continuous motor capacity.')
    for m in NAMES:
        print(f'  {m}: start={baseline[m]:.4f}, peak={peaks[m]:.4f}')
    return plan, factors


def check_servo_health(bus, reference=None):
    """Compare feedback with factory limits; never alter protection registers."""
    readings = {}
    for motor in NAMES:
        temperature = bus.read('Present_Temperature', motor, normalize=False, num_retry=2)
        max_temperature = bus.read('Max_Temperature_Limit', motor, normalize=False, num_retry=2)
        voltage = bus.read('Present_Voltage', motor, normalize=False, num_retry=2)
        min_voltage = bus.read('Min_Voltage_Limit', motor, normalize=False, num_retry=2)
        max_voltage = bus.read('Max_Voltage_Limit', motor, normalize=False, num_retry=2)
        status = bus.read('Status', motor, normalize=False, num_retry=2)
        readings[motor] = voltage
        if max_temperature <= 10 or min_voltage >= max_voltage:
            raise RuntimeError(f'Invalid protection limit feedback: {motor}')
        if temperature >= max_temperature-10 or not min_voltage < voltage < max_voltage:
            raise RuntimeError(f'Servo health limit: {motor}; temperature={temperature}/{max_temperature}, '
                               f'voltage_raw={voltage}, allowed=({min_voltage},{max_voltage})')
        if status != 0:
            raise RuntimeError(f'Servo reported status: {motor}; raw Status={status}')
        if reference and voltage < reference[motor]*0.9:
            raise RuntimeError(f'Voltage sag: {motor}; raw={voltage}, pre-enable={reference[motor]}')
    return readings


def check_tracking(actual, goal, factors, context):
    for motor in NAMES:
        threshold = max(23, round(factors[motor]*5)) if motor=='gripper' else 34
        if abs(actual[motor]-goal[motor]) > threshold:
            raise RuntimeError(f'Tracking error: {motor}; {context}; goal={goal[motor]}, actual={actual[motor]}, '
                               f'error={actual[motor]-goal[motor]} ticks; stopping demo')


def make_plan(start, calibration, amplitude=1.0):
    if not math.isfinite(amplitude) or not 0.2 <= amplitude <= 1.5:
        raise ValueError("Amplitude must be between 0.2 and 1.5")
    factors = {}
    scales = {}
    directions = {}
    for index, motor in enumerate(NAMES):
        cal = calibration[motor]
        span = cal.range_max - cal.range_min
        if span <= 0:
            raise RuntimeError(f"Invalid calibration range: {motor}")
        if not cal.range_min <= start[motor] <= cal.range_max:
            raise RuntimeError(f"{motor} is outside its calibrated range; check calibration and starting pose.")
        factor = span / 100 if motor == "gripper" else 4095 / 360
        factors[motor] = factor
        margin = max(5, round(span * 0.03)) if motor == "gripper" else 80
        low, high = cal.range_min + margin, cal.range_max - margin
        if low >= high:
            raise RuntimeError(f"Insufficient calibrated travel: {motor}")
        left, right = start[motor] - low, high - start[motor]
        clearance = min(left, right)
        maximum = max(abs(offset[index]) for _, offset in POSES) * factor * amplitude
        # Near an endpoint, put the whole sweep on the available side instead
        # of asking the user to center every joint (including a closed gripper).
        directions[motor] = 0 if clearance >= maximum * 0.5 else (1 if right >= left else -1)
        available = clearance if directions[motor] == 0 else max(left, right)
        scales[motor] = min(1.0, available / maximum)
        if scales[motor] < 0.2:
            raise RuntimeError(f"Insufficient usable travel: {motor}; check calibration and starting pose.")
    plan = []
    for pose_index, (label, offsets) in enumerate(POSES):
        target = {}
        for index, motor in enumerate(NAMES):
            offset = offsets[index]
            if directions[motor]:
                peak = max(abs(values[index]) for _, values in POSES)
                offset = directions[motor] * (offset + peak) / 2
            if pose_index == len(POSES) - 1:
                offset = 0  # Always return exactly to the measured starting pose.
            target[motor] = round(start[motor] + offset * factors[motor] * amplitude * scales[motor])
        plan.append((label, target))
    return plan, factors, scales


def run_demo(bus, calibration, viewer=None, automatic=False, amplitude=1.0, speed=12.0, cycles=1,
             wide=False, range_fraction=0.7, load_aware=False):
    if not math.isfinite(speed) or not 3 <= speed <= 40 or not 1 <= cycles <= 5:
        raise ValueError("Speed must be 3..40 degrees/sec and cycles 1..5")
    print("DEMO: all six joints will hold position and move together.")
    print("Clear space around the entire arm. Support it before the final torque release.")
    if wide:
        raise RuntimeError('Wide motion disabled after tracking fault. Use --analyze-load first, then --load-aware for conservative verification.')
    if not automatic and input("Type DEMO to start, or Enter to cancel: ").strip().upper() != "DEMO":
        return
    for motor in NAMES:
        if bus.read("Operating_Mode", motor, num_retry=2) != 0:
            raise RuntimeError(f"{motor} is not in position mode")
        if bus.read("Torque_Enable", motor, num_retry=2) != 0:
            raise RuntimeError(f"{motor} is already powered")
    start = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    if load_aware:
        plan, factors = make_load_plan(start, calibration)
        voltage_reference = check_servo_health(bus)
    else:
        plan, factors, _ = make_plan(start, calibration, amplitude)
    for motor in NAMES:
        displacements = [(target[motor] - start[motor]) / factors[motor] for _, target in plan]
        print(f"  {motor}: {min(displacements):+.1f} .. {max(displacements):+.1f} "
              f"{'percent' if motor == 'gripper' else 'degrees'} relative to start")
    if viewer:
        viewer.sample(bus, "Demo: ready")
    try:
        # Hold the measured pose before enabling torque; no initial pose jump.
        bus.sync_write("Goal_Position", start, normalize=False, num_retry=2)
        bus.enable_torque(NAMES, num_retry=2)
        if load_aware:
            print('Verifying current-pose hold before any movement...')
            for _ in range(10):
                time.sleep(0.05)
                actual = bus.sync_read('Present_Position', normalize=False, num_retry=2)
                check_tracking(actual, start, factors, 'initial pose hold')
                check_servo_health(bus, voltage_reference)
                if viewer:
                    viewer.sample(bus, 'Verifying current-pose hold')
            print('Pose hold check passed. Starting small verification motion.')
        previous = start
        for cycle in range(cycles):
            for label, target in plan:
                # Quintic easing has peak slope 1.875; enforce peak speed, not just average speed.
                duration = segment_duration(previous, target, factors, speed)
                if load_aware:
                    # Quintic peak acceleration coefficient 10/sqrt(3); keep acceleration low.
                    duration = max(duration, 5.0,
                                   max(1.875*abs(target[m]-previous[m])/factors[m]/min(speed,8) for m in NAMES),
                                   max(math.sqrt(5.774*abs(target[m]-previous[m])/factors[m]/10) for m in NAMES))
                steps = math.ceil(duration / 0.05)
                print(f"[{cycle + 1}/{cycles}] {label} ({duration:.1f}s)")
                for step in range(1, steps + 1):
                    t = step / steps
                    blend = 10*t**3 - 15*t**4 + 6*t**5
                    goal = {motor: round(previous[motor] + (target[motor] - previous[motor]) * blend) for motor in NAMES}
                    bus.sync_write("Goal_Position", goal, normalize=False, num_retry=2)
                    time.sleep(duration / steps)
                    actual = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                    check_tracking(actual, goal, factors, label)
                    if viewer:
                        viewer.sample(bus, f"Demo {cycle + 1}/{cycles}: {label}")
                    if load_aware and step % 10 == 0:
                        check_servo_health(bus, voltage_reference)
                time.sleep(0.4)
                actual = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                for motor in NAMES:
                    threshold = max(11, round(factors[motor] * 3)) if motor == "gripper" else 23
                    if abs(actual[motor] - target[motor]) > threshold:
                        raise RuntimeError(f"Target not reached: {motor}")
                previous = target
        if viewer:
            viewer.sample(bus, "Demo complete: holding start pose; support arm before release")
        # --auto skips only the start prompt, never silently releases a loaded arm at completion.
        input("Returned to start. SUPPORT THE ARM, then press Enter to release all torque: ")
    finally:
        try:
            bus.disable_torque(NAMES, num_retry=2)
            print("All joints: torque OFF.")
        except Exception as exc:
            raise RuntimeError("Cannot release torque. Disconnect motor power now.") from exc
