"""Read-only SO-101 motor diagnostics; no force or payload estimates."""
import math

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')

REGISTERS = {
    'voltage_v': ('Present_Voltage', .1),
    'temperature_c': ('Present_Temperature', 1),
    'load_raw': ('Present_Load', 1),
    'current_raw': ('Present_Current', 1),
    'status': ('Status', 1),
}


def read_diagnostics(bus):
    """Called by the serial owner thread; never opens USB or writes registers."""
    result = {}
    for field, (register, scale) in REGISTERS.items():
        values = bus.sync_read(register, normalize=False, num_retry=2)
        if set(values) != set(JOINTS):
            raise RuntimeError('Incomplete motor diagnostics: '+register)
        for motor, raw in values.items():
            if not math.isfinite(float(raw)):
                raise RuntimeError('Nonfinite motor diagnostic: '+register)
            result.setdefault(motor, {})[field] = raw * scale
    return result
