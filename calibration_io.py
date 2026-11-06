"""Verify calibration register writes even when a servo write ACK is lost."""
from contextlib import contextmanager
import time


REGISTERS = frozenset({'Homing_Offset', 'Min_Position_Limit', 'Max_Position_Limit',
                       'Operating_Mode', 'Torque_Enable', 'Lock'})


@contextmanager
def verified_calibration_writes(bus, attempts=3):
    """Instance-local wrapper; never modify LeRobot's installed package or enable torque."""
    original_write = bus.write

    def write(data_name, motor, value, *, normalize=True, num_retry=0):
        if data_name == 'Goal_Position' or (data_name == 'Torque_Enable' and int(value) != 0):
            raise RuntimeError('Manual calibration cannot enable torque or command a position')
        if data_name not in REGISTERS:
            return original_write(data_name, motor, value, normalize=normalize, num_retry=num_retry)
        expected = int(value)
        # Do not repeatedly write an EEPROM register that already has the desired value.
        if bus.read(data_name, motor, normalize=False, num_retry=2) == expected:
            return
        observed = None
        last_error = None
        for attempt in range(attempts):
            ack_error = None
            try:
                original_write(data_name, motor, value, normalize=False, num_retry=0)
            except ConnectionError as error:
                # Missing ACK does NOT prove the write failed; read it before writing again.
                ack_error = error
                last_error = error
            # Device-reported errors (RuntimeError) propagate immediately rather than being hidden.
            time.sleep(.02)
            try:
                observed = bus.read(data_name, motor, normalize=False, num_retry=2)
            except ConnectionError as error:
                observed = None
                last_error = error
            if observed == expected:
                if ack_error is not None:
                    print(f'READBACK VERIFIED: {motor}/{data_name}={expected}; write ACK missing but stored value matches.')
                return
            if attempt+1 < attempts:
                time.sleep(.05)
        raise RuntimeError(f'Cannot verify calibration write: {motor}/{data_name}; '
                           f'expected={expected}, readback={observed}, attempts={attempts}. '
                           'Calibration is incomplete; do not teleoperate.') from last_error

    bus.write = write
    try:
        yield
    finally:
        bus.write = original_write
