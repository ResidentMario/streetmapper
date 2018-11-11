import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, mapping
from tqdm import tqdm
import rtree


def bldgs_on_block(bldgs, block, include_multimatches=False):
    """
    Returns the buildings that correspond with a given block.

    Parameters
    ----------
    bldgs: gpd.GeoDataFrame
        A tabular `GeoDataFrame` whose `geometry` consists of building footprints in the area of interest.

    block: gpd.GeoSeries
        A `GeoSeries` whose `geometry` is the block of interest.

    include_multimatches: boolean
        Whether or not to include buildings that are not fully contained with in the given block, e.g. those which span
        multiple blocks. This is usually only possible due to errors in the dataset. Defaults to `True`.

        Note that the contains operation, which is used when `False`, is slower than the intersects operation, which is
        used when `True`.
    """
    if include_multimatches:
        # TODO: this doesn't work?
        # return bldgs[block.intersects(bldgs)]
        raise NotImplementedError
    else:
        return bldgs[block.contains(bldgs)]


def join_bldgs_blocks(bldgs, blocks, bldg_uid_col='bldg_uid', block_uid_col='block_uid'):
    """
    Performs a geo-spatial join on buildings and blocks. Each of the `buildings` searches for `blocks` that it
    intersects with. In a good case, the building is found to be located within a particular block. In a bad case, the
    building is found to match with no blocks (if the space it is located on seemingly isn't included in `blocks`) or
    with many blocks (if its footprint intersects with more than one block).

    This function therefore returns a tuple of three items: `matches` for buildings uniquely joined to blocks,
    `multimatches` for buildings joined to multiple blocks, a `nonmatches` for buildings joined to no blocks.

    These bad cases are inevitable, and will generally occur when city blocks are rewritten. This will tend to mean
    around construction zones. The greater the difference in the timestamp between the buildings dataset and the blocks
    dataset, the greater the risk of match degradation.

    > Warning: buildings which touch the boundary of their block will likely be declared a multi-match due to the
    semantics of the intersection operation.

    Parameters
    ----------
    bldgs: gpd.GeoDataFrame
        A tabular `GeoDataFrame` whose `geometry` consists of building footprints in the area of interest.

    blocks: gpd.GeoDataFrame
        A tabular `GeoDataFrame` whose `geometry` corresponds with all block footprints in the area of interest.

    bldg_uid_col: str
        The unique ID column for the buildings. This field must be present in the `buildings` dataset, and it must
        be uniquely keyed.

    block_uid_col: str
        The unique ID column for the blocks. This field must be present in the `vlocks` dataset, and it must be
        uniquely keyed.

    Returns
    -------
    (matches, multimatches, nonmatches) : tuple
        A tuple of three `GeoDataFrame`. The first element is of buildings-block pairs that are unique, the second
        element is buildings that span multiple blocks, and the third is buildings that span no blocks (at least
        according to the data given).
    """
    if len(bldgs) == 0 or len(blocks) == 0:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()

    all_matches = (gpd.sjoin(bldgs, blocks, how="left", op='intersects'))

    nonmatches = all_matches[pd.isnull(all_matches['index_right'])]

    all_matches = all_matches.drop(columns=['index_right'])
    matches = all_matches.groupby(bldg_uid_col).filter(lambda df: sum(df[block_uid_col].notnull()) == 1).reset_index()
    multimatches = all_matches.groupby(bldg_uid_col).filter(lambda df: len(df) > 1).reset_index()

    return matches, multimatches, nonmatches


def _simplify(shp, tol=0.05):
    """
    Generate a simplified shape for shp, within a specified tolerance.

    Used for blockface alignment.
    """
    simp = None
    for thresh in [0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
        simp = shp.simplify(thresh)
        if shp.difference(simp).area / shp.area < tol:
            break

    return simp


# TODO: continue implementing tests and refactoring the API surface from here!
def blockfaces_for_block(block, tol=0.05):
    """
    Given a `block` footprint as a geometry, returns a breakdown of that block into individual blockface segments.

    We pass a simplification algorithm over the block geometry and examine the points which survive in order to
    determine which points are geometrically important to the block geometry. These points form the boundaries of the
    blockfaces we partition the block into. Points not included in the simplified geometry become intermediate points
    along the blockface boundary.

    Parameters
    ----------
    block: gpd.GeoSeries
        Data for a single block.
    tol: float
        The maximum simplification threshold. Blockfaces will be determined using a simplified representation of the
        block with at most this much inaccuracy with respect to the "real thing". Higher tolerances result in fewer but
        more complex blockfaces. Value is a float ratio out of 1.

    Returns
    -------
    out: gpd.GeoDataFrame
        Data and geometries corresponding with each blockface.
    """
    orig = block.geometry.buffer(0)  # MultiPolygon -> Polygon
    simp = _simplify(orig, tol=tol)

    orig_coords = mapping(orig)['coordinates'][0]
    simp_coords = mapping(simp)['coordinates'][0]

    simp_out = []
    orig_out = []

    orig_coords_idx = 0

    # generate coordinate strides
    for idx in range(1, len(simp_coords)):
        simp_blockface_start_coord, simp_blockface_end_coord = simp_coords[idx - 1], simp_coords[idx]

        orig_blockface_start_coord_idx = orig_coords_idx
        orig_coords_idx += 1
        while orig_coords[orig_coords_idx] != simp_blockface_end_coord:
            orig_coords_idx += 1
        orig_blockface_end_coord_idx = orig_coords_idx + 1

        simp_out.append((simp_blockface_start_coord, simp_blockface_end_coord))
        orig_out.append(orig_coords[orig_blockface_start_coord_idx:orig_blockface_end_coord_idx])

    # id will be a mutation of the block id
    block_id = block.geoid10

    out = []
    from shapely.geometry import LineString

    # frame id, block id, original geometry, and simplified geometry
    # TODO: allow arbitrary ID fields.
    for n, (simp_blockface_coord_seq, orig_blockface_coord_seq) in enumerate(zip(simp_out, orig_out)):
        blockface_num = n + 1
        out.append({
            "geoid10_n": f"{block_id}_{blockface_num}",
            "geoid10": block_id,
            "simplified_geometry": LineString(simp_blockface_coord_seq),
            "geometry": LineString(orig_blockface_coord_seq)
        })

    out = gpd.GeoDataFrame(out)

    return out


def _drop_noncontiguous_blocks(blocks):
    """
    Removes geometries from a `GeoDataFrame` input that are not polygonal, e.g. consist of multiple shapes. This is
    used to exclude discontiguous blocks from the data, e.g. island formations.
    """
    return blocks[blocks.geometry.map(lambda g: isinstance(g.buffer(0), Polygon))]


def blockfaces_for_blocks(blocks, tol=0.05):
    """
    Given `blocks` footprint data, returns a breakdown of those block into individual blockface segments.

    For implementation details see `blockfaces_for_block`, which this method calls.

    Parameters
    ----------
    blocks: gpd.GeoDataFrame
        Data for
    tol: float
        The maximum simplification threshold. Higher values result in fewer but more complex blockfaces. C.f.
        `blockfaces_for_block`.

    Returns
    -------
    out: gpd.GeoDataFrame
        Data and geometries for each blockface of each block.
    """
    contiguous_blocks = _drop_noncontiguous_blocks(blocks)
    blockfaces = pd.concat(contiguous_blocks.apply(lambda b: blockfaces_for_block(b, tol=tol), axis='columns').values)
    blockfaces = blockfaces.drop(columns=['simplified_geometry'])  # write compatibility
    return blockfaces


def create_index(srs):
    """
    Create a geospatial index of streets using the `rtree` package. Helper to `find_matching_street`, which uses this
    index to perform the actual join.
    """
    import rtree
    index = rtree.index.Index()

    for idx, feature in srs.iterrows():
        index.insert(idx, feature.geometry.bounds)

    return index


def find_matching_street(blockface, streets, index):
    """
    Finds the street that matches a blockface. Outputs zero to many hits.

    Most street segments contain subsegments which are close matches to blockface segments, but there is floating point
    arithmetic error and measurement error that keeps the match from being one hundred percent correct. To find matches
    we first use a spatial index to narrow down to near hits. Then we buffer the street segment geometrically (creating
    a polygon) and check if the buffered segment contains the blockface (or vice versa). Matches occur wherever this
    hueristic evaluates to True.

    Parameters
    ----------
    blockface: gpd.GeoSeries
        Data for a single blockface.
    streets: gpd.GeoDataFrame
        Data for streets in the area of interest.
    index: rtree.index.Index
        A geospatial index for streets, as created with `create_index`.

    Returns
    -------
    None or gpd.GeoDataFrame
        None if no matches are found. Otherwise, data for each street matching the given blockface.
    """
    def n_nearest_streets(block, mult=2):
        """
        Returns a frame of streets nearest the given block. mult controls how many times more streets will be looked up
        than the block has sides; because we need to return a lot of results, just to make sure we get every street
        fronting the block.
        """
        x, y = block.geometry.envelope.exterior.coords.xy
        n = (len(x) - 1) * 2
        idxs = index.nearest((min(x), min(y), max(x), max(y)), n)
        return streets.iloc[list(idxs)]

    streets_matched = n_nearest_streets(blockface)
    sub_matches = []
    for idx, street in streets_matched.iterrows():
        if street.geometry.buffer(0.00005).contains(blockface.geometry):
            return gpd.GeoDataFrame([street], geometry=[street.geometry])
        elif blockface.geometry.buffer(0.00005).contains(street.geometry):
            sub_matches.append(street)

    if len(sub_matches) > 0:
        return gpd.GeoDataFrame(sub_matches)


def merge_street_segments_blockfaces_blocks(blockfaces, streets, index):
    """
    Matches street segments to blockfaces, and returns a concatenated data representation thereof.

    See `find_matching_street` for implementation details.

    Parameters
    ----------
    blockface: gpd.GeoSeries
        Data for blockfaces of interest.
    streets: gpd.GeoDataFrame
        Data for streets in the area of interest.
    index: rtree.index.Index
        A geospatial index for streets, as created with `create_index`.

    Returns
    -------
    gpd.GeoDataFrame
        Data on each match found.
    """
    # TODO: parameterize the join key
    matches = []

    for idx, blockface in tqdm(blockfaces.iterrows()):
        matches.append(find_matching_street(blockface, streets, index))

    matches_merge = []

    for idx, (_, dat) in tqdm(enumerate(blockfaces.iterrows())):
        corresponding_street_segments = matches[idx]
        geoid10 = np.nan if corresponding_street_segments is None else dat.geoid10

        if pd.notnull(geoid10):
            corresponding_street_segments = corresponding_street_segments.assign(
                geoid10=geoid10, geoid10_n=dat.geoid10_n
            )

        matches_merge.append(corresponding_street_segments)

    street_segments = pd.concat(matches_merge)
    del matches_merge
    return street_segments


def filter_on_block_id(block_id, block_id_key="geoid10"):
    """
    Helper factory function. Returns a function which may be applied to a chunk of data to select on an ID. Used by
    `get_block_data`.
    """
    def select(df):
        return (df.set_index(block_id_key)
                .filter(like=block_id, axis='rows')
                .reset_index()
                )

    return select


def get_block_data(block_id, street_segments, blockfaces, buildings):
    """
    Given a block ID and a set of data inputs, returns a tuple of filtered data objects containing only matches on that
    ID.

    Parameters
    ----------
    block_id: str
        A unique ID for a block, which is expected to appear in all of the other inputs (this will only be true if you
        have already performed all of the requisite geospatial joins).
    street_segments, gpd.GeoDataFrame
        Street data.
    blockfaces, gpd.GeoDataFrame
        Blockface data.
    buildings, gpd.GeoDataFrame
        Building data.

    Returns
    -------
    tuple
        The corresponding filtered set of data.
    """
    ss = street_segments.pipe(filter_on_block_id(block_id))
    bf = blockfaces.pipe(filter_on_block_id(block_id))
    bldgs = buildings.pipe(filter_on_block_id(block_id))
    return ss, bf, bldgs


def collect_strides(point_observations):
    """
    Given a sequence of observations of which building is nearest to points on a shape (expressed as a percentage
    length out of 1), collects those observations into contiguous strides, thereby assigning buildings to chunks of the
    shape.

    Internal method to `frontages_for_blockfaces`.
    """
    point_obs_keys = list(point_observations.keys())
    curr_obs_start_offset = point_obs_keys[0]
    curr_obs_start_bldg = point_observations[point_obs_keys[0]]
    strides = dict()

    for point_obs in point_obs_keys[1:]:
        bldg_observed = point_observations[point_obs]
        if bldg_observed != curr_obs_start_bldg:
            strides[(curr_obs_start_offset, point_obs)] = curr_obs_start_bldg
            curr_obs_start_offset = point_obs
            curr_obs_start_bldg = bldg_observed
        else:
            continue

    strides[(curr_obs_start_offset, '1.00')] = bldg_observed
    return strides


def get_stride_boundaries(strides, step_size=0.02):
    """
    Given a sequence of strides as returned by `collect_strides`, determines boundary areas of inaccuracies.

    Internal method to `frontages_for_blockfaces`.
    """
    boundaries = list()

    keys = list(strides.keys())
    for idx, key in enumerate(keys[1:]):
        curr = strides[key]
        boundaries.append((key[0], str(float(key[0]) + step_size)))

    return boundaries


def cut(line, distance):
    """
    Cuts a line in two at a distance from its starting point. Helper function.

    Modified version of algorithm found at http://toblerity.org/shapely/manual.html#object.project.
    """
    if distance == 0.0:
        return LineString()
    elif distance == 1.0:
        return LineString(line)
    elif distance < 0.0 or distance > 1.0:
        raise ValueError("Cannot cut a line using a ratio outside the range [0, 1]")

    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p), normalized=True)
        if pd == distance:
            return LineString(coords[:i + 1])
        if pd > distance:
            cp = line.interpolate(distance, normalized=True)
            return LineString(coords[:i] + [(cp.x, cp.y)])


def reverse(l):
    """
    Reverses LineString coordinates. Also known as changing the "winding direction" of the geometry. Helper function.
    """
    l_x, l_y = l.coords.xy
    l_x, l_y = l_x[::-1], l_y[::-1]
    return LineString(zip(l_x, l_y))


def chop_line_segment_using_offsets(line, offsets):
    """
    Cuts a line into offset segments. Helper function.
    """
    offset_keys = list(offsets.keys())
    out = []

    for off_start, off_end in offset_keys:
        out_line = LineString(line.coords)
        orig_length = out_line.length

        # Reverse to cut off the start.
        out_line = reverse(out_line)
        out_line = cut(out_line, 1 - float(off_start))
        out_line = reverse(out_line)

        # Calculate the new cutoff end point, and apply it to the line.
        l_1_2 = (float(off_end) - float(off_start)) * orig_length
        l_1_3 = (1 - float(off_start)) * orig_length
        new_off_end = l_1_2 / l_1_3

        # Perform the cut.
        out_line = cut(out_line, new_off_end)

        out.append(np.nan if out_line is None else out_line)

    return out


def frontages_for_blockface(bldgs, blockface, step_size=0.01):
    """
    Given a single blockface and a set of buildings of interest, assigns each chunk of the blockface to the building
    out of the buildings of interest nearest to that blockface segment. The step size controls the size of the
    segments; specifying a lower value increases the accuracy of the algorithm, but also increases how long it takes
    to execute.

    The frontages are generated by performing an iterative search along the length of the blockface, asking at each
    point which building is nearest to that given point. The distance between points is controlled by the `step_size`.

    Parameters
    ----------
    bldgs: gpd.GeoDataFrame
        Data on buildings.
    blockface: gpd.GeoSeries
        A single blockface.
    step_size: float
        The step size, which controls the accuracy of the assessment (at the cost of speed).

    Returns
    -------
    gpd.GeoDataFrame
        Frontage match data.
    """
    # TODO: use a smarter search strategy than simple iterative search
    index = rtree.Rtree()

    if len(bldgs) == 0:
        return gpd.GeoDataFrame()

    for idx, bldg in bldgs.iterrows():
        index.insert(idx, bldg.geometry.bounds)

    bldg_frontage_points = dict()

    search_space = np.arange(0, 1, step_size)
    next_search_space = []
    while len(search_space) > 0:
        for offset in search_space:
            search_point = blockface.geometry.interpolate(offset, normalized=True)
            nearest_bldg = list(index.nearest(search_point.bounds, 1))[0]
            bldg_frontage_points[str(offset)[:6]] = nearest_bldg

        strides = collect_strides(bldg_frontage_points)
        search_space = next_search_space

    # convert the list of strides to a proper GeoDataFrame
    out = []
    for sk in strides.keys():
        srs = bldgs.loc[strides[sk], ['geoid10', 'sf16_BldgID']]
        srs['geoid10_n'] = blockface['geoid10_n']
        srs['geom_offset_start'] = sk[0]
        srs['geom_offset_end'] = sk[1]
        out.append(srs)

    out = gpd.GeoDataFrame(out)

    geoms = chop_line_segment_using_offsets(blockface.geometry, strides)
    out['geometry'] = geoms
    return out


def calculate_frontages(blocks, streets, blockfaces, buildings):
    """
    Given the set of available data, calculates building frontages for the given blocks.

    To only perform a more selective search, e.g. to focus on only a single area or even a single block, limit the
    `blocks` passed to the function.

    For implementation see `frontages_for_blockface`.

    Parameters
    ----------
    blocks: gpd.GeoDataFrame
        Data for blocks of interest.
    blockfaces: gpd.GeoSeries
        Data for blockfaces.
    streets: gpd.GeoDataFrame
        Data for streets.
    buildings: gpd.GeoDataFrame
        Data for buildings.

    Returns
    -------
    gpd.GeoDataFrame
        Data on building frontages.
    """
    # TODO: undo hard-code on the ID.
    frontages = []

    for block_idx, block in tqdm(list(blocks.iterrows())):
        _, blockface_targets, building_targets = get_block_data(block.geoid10, streets, blockfaces, buildings)
        for _, blockface_target in blockface_targets.iterrows():
            result = frontages_for_blockface(building_targets, blockface_target)
            frontages.append(result)

    frontages = gpd.GeoDataFrame(pd.concat([f for f in frontages if len(f) > 0]))
    if len(frontages) == 0:
        return frontages
    else:
        return frontages.groupby('sf16_BldgID').apply(
            lambda df: df.assign(sf16_BldgID_n=[f'{df.iloc[0].sf16_BldgID}_{n}' for n in range(len(df))])
        ).reset_index(drop=True)


def select_area_of_interest(blocks, poly):
    """Select all blocks that touch the given boundaries. Utility function."""
    return blocks[blocks.geometry.map(lambda block: poly.contains(block))]


def distance(line, point):
    """Returns the distance between a line and a point."""
    proj_pos = line.project(point, normalized=True)
    proj_point = line.interpolate(proj_pos, normalized=True)
    dist = proj_point.distance(point)
    return dist


def assign_points_to_frontages(points, frontages, index):
    """
    Given a sequence of `points` and of `frontages` and a geospatial `index` on `frontages`, assigns points to
    frontages in the dataset.
    """
    idxs = points.geometry.map(lambda g: list(index.nearest(g.bounds, 5)))
    frontage_groups = idxs.map(lambda idx_group: frontages.iloc[idx_group]).values

    out = []
    for i in range(len(points)):
        point = points.iloc[i]
        frontage_group = frontage_groups[i]

        distances = [distance(frontage.geometry, point.geometry) for _, frontage in frontage_group.iterrows()]
        frontage_idx = np.argmin(distances)
        out.append(frontages.iloc[frontage_group.index[frontage_idx]])

    return pd.concat(out, axis='columns').T.sf16_BldgID_n
