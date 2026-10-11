import unittest
from ecg_tool.interaction import zoom_limits


class WheelZoomTests(unittest.TestCase):
    def test_pointer_stays_at_same_relative_position(self):
        for limits, anchor in (((0,10),2), ((300,100),250)):
            a,b=zoom_limits(limits,anchor,1)
            self.assertAlmostEqual((anchor-a)/(b-a),(anchor-limits[0])/(limits[1]-limits[0]))
            self.assertLess(abs(b-a),abs(limits[1]-limits[0]))
            c,d=zoom_limits((a,b),anchor,-1)
            self.assertAlmostEqual(c,limits[0])
            self.assertAlmostEqual(d,limits[1])
