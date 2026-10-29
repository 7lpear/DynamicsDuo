"""URDF gravity estimates, not a motor capacity or collision certificate."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

MAPPING = dict(zip(
    ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'],
    ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']))


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = [float(v) for v in element.get('xyz', '0 0 0').split()]
        r, p, y = [float(v) for v in element.get('rpy', '0 0 0').split()]
        result[:3, :3] = rotation([0,0,1], y) @ rotation([0,1,0], p) @ rotation([1,0,0], r)
    return result


class GravityModel:
    def __init__(self, path=None):
        root = ET.parse(path or Path(__file__).parent/'viewer/model/urdf/so101_new_calib.urdf').getroot()
        self.links = {}
        self.joints = []
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is None or inertial.find('mass') is None:
                raise ValueError('Missing inertial mass: '+link.get('name'))
            self.links[link.get('name')] = (float(inertial.find('mass').get('value')), origin(inertial.find('origin')))
        for joint in root.findall('joint'):
            self.joints.append((joint.get('name'), joint.find('parent').get('link'),
                                joint.find('child').get('link'), origin(joint.find('origin')),
                                [float(v) for v in joint.find('axis').get('xyz').split()]))
        children = {j[2] for j in self.joints}
        self.root = next(name for name in self.links if name not in children)
        self.mass = sum(mass for mass, _ in self.links.values())

    def potential(self, angles):
        frames = {self.root: np.eye(4)}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                name, parent, child, fixed, axis = joint
                if parent not in frames:
                    continue
                motion = np.eye(4)
                motion[:3,:3] = rotation(axis, angles.get(name, 0))
                frames[child] = frames[parent] @ fixed @ motion
                remaining.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('Invalid URDF joint tree')
        return sum(mass*9.80665*(frames[name]@com)[2,3] for name,(mass,com) in self.links.items())

    def torque(self, positions):
        angles = {joint: math.radians(float(positions[motor])) for motor,joint in MAPPING.items()}
        result = {}
        for motor,joint in MAPPING.items():
            plus, minus = angles.copy(), angles.copy()
            plus[joint] += 1e-5
            minus[joint] -= 1e-5
            result[motor] = abs((self.potential(plus)-self.potential(minus))/2e-5)
        return result


def normalized(raw, calibration):
    result = {}
    for motor, value in raw.items():
        cal = calibration[motor]
        result[motor] = ((value-cal.range_min)*100/(cal.range_max-cal.range_min) if motor == 'gripper'
                         else (value-(cal.range_min+cal.range_max)/2)*360/4095)
    return result


def report_load(bus):
    positions = bus.sync_read('Present_Position', num_retry=2)
    print('Measured positions (body degrees; gripper percent):')
    for motor, value in positions.items():
        print(f'  {motor}: {value:.2f}')
    model = GravityModel()
    print(f'URDF total mass: {model.mass:.3f} kg; stock model only, no payload/cables.')
    print('Estimated STATIC gravity torque (Nm); excludes friction, acceleration, contact, gripping force.')
    for motor, torque in model.torque(positions).items():
        print(f'  {motor}: {torque:.4f} Nm')
    print('Raw servo diagnostic registers (units/scaling depend on servo model):')
    for motor in MAPPING:
        values = {}
        for register in ['Torque_Enable','Present_Voltage','Present_Temperature','Present_Load','Present_Current','Status']:
            try:
                values[register] = bus.read(register, motor, normalize=False, num_retry=2)
            except Exception as exc:
                values[register] = str(exc)
        print(motor, values)
    print('READ ONLY: no motion or torque-enable commands were sent.')
