import unittest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

import streetmapper


class TestJoinBldgsBlocks(unittest.TestCase):
    def setUp(self):
        self.blocks = gpd.GeoDataFrame(
            {'block_uid': [1, 2]},
            geometry=[
                Polygon(((0, -1), (0, 1), (1, 1), (1, -1))),  # top side
                Polygon(((0, -1), (0, 1), (-1, 1), (-1, -1)))  # bottom side
            ]
        )

    def testNoBldgs(self):
        bldgs = gpd.GeoDataFrame()
        blocks = self.blocks

        matches, multimatches, nonmatches =\
            streetmapper.pipeline.join_bldgs_blocks(bldgs, blocks, 'bldg_uid', 'block_uid')

        self.assertEqual(len(matches), 0)
        self.assertEqual(len(multimatches), 0)
        self.assertEqual(len(nonmatches), 0)

    def testFullyUnivariateMatch(self):
        bldgs = gpd.GeoDataFrame(
            {'bldg_uid': [1, 2, 3, 4]},
            geometry=[
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))).buffer(-0.01),  # top right
                Polygon(((0, 0), (0, -1), (1, -1), (1, 0))).buffer(-0.01),  # top left
                Polygon(((0, 0), (0, 1), (-1, 1), (-1, 0))).buffer(-0.01),  # bottom right
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))).buffer(-0.01)  # bottom left
            ]
        )
        blocks = self.blocks

        matches, multimatches, nonmatches =\
            streetmapper.pipeline.join_bldgs_blocks(bldgs, blocks, 'bldg_uid', 'block_uid')

        self.assertEqual(len(matches), 4)
        self.assertEqual(len(multimatches), 0)
        self.assertEqual(len(nonmatches), 0)

    def testAllKindsOfMatches(self):
        bldgs = gpd.GeoDataFrame(
            {'bldg_uid': [1, 2, 3]},
            geometry=[
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))).buffer(-0.01),  # top right, interior
                Polygon(((-1, 0), (1, 0), (1, -1), (-1, -1))).buffer(-0.01),  # bottom, spanning
                Polygon(((10, 10), (10, 11), (11, 11), (11, 10)))  # exterior
            ]
        )
        blocks = self.blocks

        matches, multimatches, nonmatches =\
            streetmapper.pipeline.join_bldgs_blocks(bldgs, blocks, 'bldg_uid', 'block_uid')

        self.assertEqual(len(matches), 1)
        self.assertEqual(len(multimatches), 2)
        self.assertEqual(len(nonmatches), 1)


class TestBldgsOnBlock(unittest.TestCase):
    def setUp(self):
        self.block = gpd.GeoSeries(Polygon(((0, 0), (0, 2), (2, 2), (2, 0))))

    def testSimple(self):
        bldgs = gpd.GeoDataFrame(geometry=[
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),  # in
            Polygon(((10, 10), (10, 11), (11, 11), (11, 10)))  # out
        ])
        result = streetmapper.pipeline.bldgs_on_block(bldgs, self.block)
        assert len(result) == 1

    def testMulitmatchOff(self):
        bldgs = gpd.GeoDataFrame(geometry=[
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),  # in
            Polygon(((1, 1), (5, 1), (5, 5), (1, 5)))  # through
        ])
        result = streetmapper.pipeline.bldgs_on_block(bldgs, self.block, include_multimatches=False)
        assert len(result) == 1

    # def testMulitmatchOn(self):
    #     bldgs = gpd.GeoDataFrame(geometry=[
    #         Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),  # in
    #         Polygon(((1, 1), (5, 1), (5, 5), (1, 5)))  # through
    #     ])
    #     result = streetmapper.pipeline.bldgs_on_block(bldgs, self.block, include_multimatches=True)
    #     assert len(result) == 2
