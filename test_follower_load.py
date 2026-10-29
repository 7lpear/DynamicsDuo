import unittest
from types import SimpleNamespace
from unittest.mock import patch
from follower_demo import NAMES, make_load_plan, run_demo, check_servo_health
from follower_load import GravityModel, normalized
from test_follower_demo import Bus


class Tests(unittest.TestCase):
    calibration = {m: SimpleNamespace(range_min=0, range_max=4095) for m in NAMES}

    def starting_pose(self):
        values = [-24.66,-104.40,41.71,75.52,-65.89,6.50]
        return {m: round(values[i]*4095/100 if m=='gripper' else 2047.5+values[i]*4095/360)
                for i,m in enumerate(NAMES)}

    def test_gravity_base_axis_and_mass(self):
        model = GravityModel()
        self.assertAlmostEqual(model.mass, .632006, places=5)
        torques = model.torque(dict.fromkeys(NAMES,0))
        self.assertLess(torques['shoulder_pan'], 1e-5)
        self.assertGreater(torques['elbow_flex'], .01)
        self.assertTrue(all(0 <= t < 3 for t in torques.values()))

    def test_plan_holds_load_bearing_joints_and_bounds_gravity(self):
        start = self.starting_pose()
        start['gripper'] = 0
        plan,_ = make_load_plan(start,self.calibration)
        model = GravityModel()
        baseline = model.torque(normalized(start,self.calibration))
        previous = start
        for _,target in plan:
            for m in ('shoulder_lift','elbow_flex','wrist_flex'):
                self.assertEqual(target[m],start[m])
            for step in range(21):
                raw = {m:previous[m]+(target[m]-previous[m])*step/20 for m in NAMES}
                t = model.torque(normalized(raw,self.calibration))
                self.assertTrue(all(t[m] <= baseline[m]+.020001 for m in NAMES))
            previous = target
        self.assertEqual(plan[-1][1],start)

    @patch('follower_demo.time.sleep')
    @patch('builtins.input',return_value='')
    def test_load_aware_run_returns_and_releases(self,_input,_sleep):
        bus = Bus()
        bus.positions = self.starting_pose()
        initial = bus.positions.copy()
        original_read = bus.read
        def healthy_read(register,motor,**kwargs):
            values = {'Present_Temperature':25,'Max_Temperature_Limit':70,
                      'Present_Voltage':120,'Min_Voltage_Limit':60,'Max_Voltage_Limit':140,'Status':0}
            return values[register] if register in values else original_read(register,motor,**kwargs)
        bus.read = healthy_read
        run_demo(bus,self.calibration,automatic=True,load_aware=True,speed=6)
        self.assertFalse(bus.powered)
        self.assertEqual(bus.positions,initial)
        for cmd in bus.commands:
            for m in ('shoulder_lift','elbow_flex','wrist_flex'):
                self.assertEqual(cmd[m],initial[m])

    def test_old_heuristic_no_longer_blocks_valid_plan(self):
        bus=Bus()
        plan,_=make_load_plan(bus.positions,self.calibration)
        self.assertEqual(plan[-1][1],bus.positions)
        self.assertFalse(bus.powered)
        self.assertEqual(bus.commands,[])

    def test_health_limits_reject_hot_and_undervoltage(self):
        bus=Bus()
        values={'Present_Temperature':65,'Max_Temperature_Limit':70,
                'Present_Voltage':120,'Min_Voltage_Limit':60,'Max_Voltage_Limit':140,'Status':0}
        bus.read=lambda register,motor,**kwargs:values[register]
        with self.assertRaisesRegex(RuntimeError,'Servo health limit'):
            check_servo_health(bus)

    def test_startup_tracking_failure_releases_before_motion(self):
        bus=Bus(fault=True)
        bus.positions=self.starting_pose()
        with patch('follower_demo.check_servo_health',return_value=dict.fromkeys(NAMES,120)), patch('follower_demo.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'initial pose hold'):
                run_demo(bus,self.calibration,automatic=True,load_aware=True)
        self.assertFalse(bus.powered)
        self.assertEqual(len(bus.commands),1)

    def test_voltage_sag_rejected_even_within_factory_range(self):
        bus=Bus()
        values={'Present_Temperature':25,'Max_Temperature_Limit':70,'Present_Voltage':107,
                'Min_Voltage_Limit':60,'Max_Voltage_Limit':140,'Status':0}
        bus.read=lambda register,motor,**kwargs:values[register]
        with self.assertRaisesRegex(RuntimeError,'Voltage sag'):
            check_servo_health(bus,dict.fromkeys(NAMES,120))
        values['Present_Temperature']=25
        values['Present_Voltage']=50
        with self.assertRaisesRegex(RuntimeError,'Servo health limit'):
            check_servo_health(bus)


if __name__=='__main__':
    unittest.main()
