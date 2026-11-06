"""Read-only HTTP telemetry and bundled 3D assets; no robot required."""
from contextlib import redirect_stdout
import io
import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from follower_viewer import FollowerViewer


class Tests(unittest.TestCase):
    def setUp(self):
        with redirect_stdout(io.StringIO()):
            self.viewer = FollowerViewer(port=0,open_browser=False)
        self.addCleanup(self.viewer.close)

    def test_measured_positions_targets_and_diagnostics_are_served(self):
        self.viewer.publish({'elbow_flex':2.},command_positions={'elbow_flex':4.},
                            diagnostics={'elbow_flex':{'voltage_v':12.}},diagnostics_timestamp=123.)
        with urlopen(self.viewer.url+'/state') as response:
            state = json.load(response)
            self.assertEqual(response.headers['Cache-Control'],'no-store')
        self.assertEqual(state['positions'],{'elbow_flex':2.})
        self.assertEqual(state['feedback']['tracking_error'],{'elbow_flex':-2.})
        self.assertEqual(state['feedback']['diagnostics_timestamp'],123.)
        self.assertNotIn('load',state)

    def test_assets_are_local_and_removed_payload_endpoint_is_not_served(self):
        for path in ('/','/app.js','/model/urdf/so101_new_calib.urdf','/vendor/three.module.js'):
            with urlopen(self.viewer.url+path) as response:
                self.assertEqual(response.status,200)
        with self.assertRaises(HTTPError) as error:
            urlopen(self.viewer.url+'/payload')
        self.assertEqual(error.exception.code,404)

    def test_nonfinite_positions_do_not_replace_last_valid_frame(self):
        self.viewer.publish({'elbow_flex':2.})
        with self.assertRaises(ValueError):
            self.viewer.publish({'elbow_flex':float('nan')})
        self.assertEqual(self.viewer.state['positions']['elbow_flex'],2.)


if __name__ == '__main__':
    unittest.main()
