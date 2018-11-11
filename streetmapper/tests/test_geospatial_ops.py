import unittest
from shapely.geometry import LineString, Point
import numpy as np

import sys; sys.path.append("../")
import streetmapper


class TestCut(unittest.TestCase):
    def testSimple(self):
        input = LineString(((0, 0), (1, 0)))
        expected = LineString(((0, 0), (0.5, 0)))
        result = garbageman.pipeline.cut(input, 0.5)
        assert expected.equals(result)

    def testZero(self):
        input = LineString(((0, 0), (1, 0)))
        expected = LineString()
        result = garbageman.pipeline.cut(input, 0.0)
        assert expected.equals(result)

    def testOne(self):
        input = LineString(((0, 0), (1, 0)))
        expected = LineString(input.coords)
        result = garbageman.pipeline.cut(input, 1.0)
        assert expected.equals(result)

    def testIllegal(self):
        input = LineString(((0, 0), (1, 0)))
        with self.assertRaises(ValueError):
            garbageman.pipeline.cut(input, -0.01)
        with self.assertRaises(ValueError):
            garbageman.pipeline.cut(input,  1.01)


class TestReverse(unittest.TestCase):
    def test(self):
        line = LineString(((0, 0), (1, 0)))
        line_r = garbageman.pipeline.reverse(line)
        assert line.coords.xy[0] == line_r.coords.xy[0][::-1]
        assert line.coords.xy[1] == line_r.coords.xy[1][::-1]
        assert garbageman.pipeline.reverse(garbageman.pipeline.reverse(line)).equals(line)


class TestChopLineSegmentsUsingOffsets(unittest.TestCase):
    def test(self):
        f = garbageman.pipeline.chop_line_segment_using_offsets
        line = LineString(((0, 0), (1, 0)))
        offsets = {
            ('0.0', '0.33'): 1,
            ('0.33', '0.43'): 2,
            ('0.43', '0.52'): 3,
            ('0.52', '0.63'): 4,
            ('0.63', '0.71'): 5,
            ('0.71', '0.81'): 6,
            ('0.81', '1.00'): 7
        }
        offset_keys = list(offsets.keys())
        expected = f(line, offsets)

        for i in range(len(expected)):
            exp_boundary = np.round(expected[i].coords[0][0], decimals=2)
            key_boundary = float(offset_keys[i][0])
            assert exp_boundary == key_boundary


class TestNearestDistance(unittest.TestCase):
    def testMidpoint(self):
        f = garbageman.pipeline.distance
        line = LineString(((0, 0), (2, 0)))
        point = Point(1, 1)
        assert f(line, point) == 1

    def testTouching(self):
        f = garbageman.pipeline.distance
        line = LineString(((0, 0), (2, 0)))

        point = Point(1, 0)
        assert f(line, point) == 0

        point = Point(2, 0)
        assert f(line, point) == 0
