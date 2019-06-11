import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, mapping
from tqdm import tqdm
import rtree


def bldgs_on_blocks(bldgs, blocks, buildings_uid_col='building_id'):
    """
    Matches buildings to blocks and returns the result.

    This function returns a three-element tuple: `matches` for buildings uniquely joined to
    blocks, `multimatches` for buildings joined to multiple blocks, and `nonmatches` for
    buildings joined to no blocks.

    Warning: buildings which touch the boundary of their block will likely be declared a
    multi-match due to the semantics of the intersection operation.

    Parameters
    ----------
    bldgs: gpd.GeoDataFrame
        Building footprints.

    blocks: gpd.GeoDataFrame
        Block footprints.

    buildings_uid_col: str
        The unique ID column name for the buildings data.

    Returns
    -------
    (matches, multimatches, nonmatches) : tuple
        A tuple of three `GeoDataFrame`. The first element is of buildings-block pairs that are
        unique, the second element is buildings that span multiple blocks, and the third is
        buildings that span no blocks (at least according to the data given).
    """
    if len(bldgs) == 0 or len(blocks) == 0:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()

    all_matches = (gpd.sjoin(bldgs, blocks, how='left', op='intersects'))

    nonmatches = (
        all_matches[pd.isnull(all_matches['index_right'])]
        .drop(columns=['index_right'])
    )

    # TODO: speed this up
    multimatches = (
        all_matches.groupby(buildings_uid_col)
        .filter(lambda df: len(df) > 1)
        .drop(columns=['index_right'])
    )

    matches = (
        all_matches[
            (~all_matches.index.isin(nonmatches.index)) & 
            (~all_matches.index.isin(multimatches.index))
        ]
        .drop(columns=['index_right'])
    )

    return matches, multimatches, nonmatches


def _simplify(shp, tol=0.05):
    """
    Generate a simplified shape, within a specified tolerance.
    """
    simp = None
    for thresh in [0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
        simp = shp.simplify(thresh)
        if shp.difference(simp).area / shp.area < tol:
            break

    return simp


def _blockfaces_on_block(block, block_uid, tol=0.05, blocks_uid_col='block_uid'):
    """
    Breaks a block into blockfaces.

    We pass a simplification algorithm over the block geometry and examine the points which
    survive in order to determine which points are geometrically important to the block
    geometry. These points form the boundaries of the blockfaces we partition the block into.
    Points not included in the simplified geometry become intermediate points along the
    blockface boundary.

    Parameters
    ----------
    block: Polygon
        Data for a single block.

    block_uid: str
        The unique ID identifying this block.

    tol: float
        The maximum simplification threshold. Blockfaces will be determined using a simplified
        representation of the block with at most this much inaccuracy with respect to the "real
        thing". Higher tolerances result in fewer but more complex blockfaces. Value is a float
        ratio out of 1.

    blocks_uid_col: str
        The unique ID column for the blocks. This field must be present in the `block`, and it
        must be uniquely keyed.

    Returns
    -------
    out: gpd.GeoDataFrame
        Data and geometries corresponding with each blockface.
    """
    orig = block.buffer(0)  # MultiPolygon -> Polygon
    simp = _simplify(orig, tol=tol)

    orig_coords = mapping(orig)['coordinates'][0]
    simp_coords = mapping(simp)['coordinates'][0]

    simp_out = []
    orig_out = []

    orig_coords_idx = 0

    # generate coordinate strides
    for idx in range(1, len(simp_coords)):
        simp_blockface_start_coord = simp_coords[idx - 1]
        simp_blockface_end_coord = simp_coords[idx]

        orig_blockface_start_coord_idx = orig_coords_idx
        orig_coords_idx += 1
        while orig_coords[orig_coords_idx] != simp_blockface_end_coord:
            orig_coords_idx += 1
        orig_blockface_end_coord_idx = orig_coords_idx + 1

        simp_out.append((simp_blockface_start_coord, simp_blockface_end_coord))
        orig_out.append(orig_coords[orig_blockface_start_coord_idx:orig_blockface_end_coord_idx])

    out = []
    for n, (simp_bf_coord_seq, orig_bf_coord_seq) in enumerate(zip(simp_out, orig_out)):
        blockface_num = n + 1
        out.append({
            blocks_uid_col: block_uid,
            'blockface_id': f"{block_uid}_{blockface_num}",
            "simplified_geometry": LineString(simp_bf_coord_seq),
            "geometry": LineString(orig_bf_coord_seq)
        })
    out = gpd.GeoDataFrame(out)
    return out


def _drop_noncontiguous_blocks(blocks):
    """
    Removes geometries from a `GeoDataFrame` input that are non-polygonal. This is used to exclude
    discontiguous blocks from the data, e.g. island formations.
    """
    return blocks[blocks.geometry.map(lambda g: isinstance(g.buffer(0), Polygon))]


def blockfaces_on_blocks(blocks, tol=0.05, blocks_uid_col='block_uid'):
    """
    Breaks `blocks` into blockfaces.

    Parameters
    ----------
    blocks: gpd.GeoDataFrame
        Blocks.
    tol: float
        The simplification threshold. Higher values result in fewer but more complex blockfaces.

    Returns
    -------
    out: gpd.GeoDataFrame
    """
    contiguous_blocks = _drop_noncontiguous_blocks(blocks)
    blockfaces = pd.concat(contiguous_blocks.apply(
        lambda b: _blockfaces_on_block(
            b.geometry, b[blocks_uid_col], tol=tol, blocks_uid_col=blocks_uid_col
        ), axis='columns').values
    )
    blockfaces = blockfaces.drop(columns=['simplified_geometry'])  # write compatibility
    return blockfaces


def _create_index(srs):
    """
    Helper function. Create a geospatial index of streets using the `rtree` package.
    """
    index = rtree.index.Index()

    for idx, feature in srs.iterrows():
        index.insert(idx, feature.geometry.bounds)

    return index


def _filter_on_block_id(block_id, block_id_key='geoid10'):
    """
    Helper function, returns a selector func.
    """
    def select(df):
        return df.set_index(block_id_key).filter(like=block_id, axis='rows').reset_index()
    return select


def _get_block_data(
    block_id, blockfaces, buildings, 
    blockfaces_block_uid_col='blockface_id', buildings_block_uid_col='building_id'
):
    """
    Helper function. Get blockfaces and buildings that match on `block_id`.

    Parameters
    ----------
    block_id: str
        A unique ID for a block, which is expected to appear in all of the other inputs.
    blockfaces, gpd.GeoDataFrame
        Blockface data.
    buildings, gpd.GeoDataFrame
        Building data.

    Returns
    -------
    tuple
        The corresponding filtered set of data.
    """
    bf = blockfaces.pipe(_filter_on_block_id(block_id, blockfaces_block_uid_col))
    bldgs = buildings.pipe(_filter_on_block_id(block_id, buildings_block_uid_col))
    return bf, bldgs


def _collect_strides(point_observations):
    """
    Given a sequence of observations of which building is nearest to points on a shape
    (expressed as a percentage length out of 1), collects those observations into contiguous
    strides, thereby assigning buildings to chunks of the shape.
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


def _cut(line, distance):
    """
    Cuts a line in two at a distance from its starting point. Helper function.

    Modified version of algorithm found at 
    http://toblerity.org/shapely/manual.html#object.project.
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


def _reverse(l):
    """
    Reverses LineString coordinates. Also known as changing the "winding direction" of the
    geometry. Helper function.
    """
    l_x, l_y = l.coords.xy
    l_x, l_y = l_x[::-1], l_y[::-1]
    return LineString(zip(l_x, l_y))


def _chop_line_segment_using_offsets(line, offsets):
    """
    Cuts a line into offset segments. Helper function.
    """
    offset_keys = list(offsets.keys())
    out = []

    for off_start, off_end in offset_keys:
        out_line = LineString(line.coords)
        orig_length = out_line.length

        # Reverse to cut off the start.
        out_line = _reverse(out_line)
        out_line = _cut(out_line, 1 - float(off_start))
        out_line = _reverse(out_line)

        # Calculate the new cutoff end point, and apply it to the line.
        l_1_2 = (float(off_end) - float(off_start)) * orig_length
        l_1_3 = (1 - float(off_start)) * orig_length
        new_off_end = l_1_2 / l_1_3

        # Perform the cut.
        out_line = _cut(out_line, new_off_end)

        out.append(np.nan if out_line is None else out_line)

    return out


def _frontages_on_blockface(
    bldgs, blockface, buildings_uid_col='building_id', blocks_uid_col='block_id',
    blockfaces_uid_col='blockface_id', step_size=0.01
):
    """
    Assign `bldgs` to a `blockface`.

    The step size controls the size of the segments; specifying a lower value increases the
    accuracy of the algorithm, but also increases how long it take to execute.

    The frontages are generated by performing an iterative search along the length of the
    blockface, asking at each point which building is nearest to that given point. The distance
    between points is controlled by the `step_size`.

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

        strides = _collect_strides(bldg_frontage_points)
        search_space = next_search_space

    # convert the list of strides to a proper GeoDataFrame
    out = []
    for sk in strides.keys():
        srs = bldgs.loc[strides[sk], [blocks_uid_col, buildings_uid_col]]
        srs[blockfaces_uid_col] = blockface[blockfaces_uid_col]
        srs['geom_offset_start'] = sk[0]
        srs['geom_offset_end'] = sk[1]
        out.append(srs)

    out = gpd.GeoDataFrame(out)

    geoms = _chop_line_segment_using_offsets(blockface.geometry, strides)
    out['geometry'] = geoms
    return out


def frontages_on_blockfaces(
    blocks, blockfaces, buildings, buildings_uid_col='building_id', blocks_uid_col='block_id',
    buildings_block_uid_col='block_id', blockfaces_block_uid_col='block_id',
    blockfaces_uid_col='blockface_id'
):
    """
    Given the set of available data, calculates building frontages for the given blocks.

    To only perform a more selective search, e.g. to focus on only a single area or even a single
    block, limit the `blocks` passed to the function.

    Parameters
    ----------
    blocks: gpd.GeoDataFrame
        Data for blocks of interest.
    blockfaces: gpd.GeoSeries
        Data for blockfaces.
    buildings: gpd.GeoDataFrame
        Data for buildings.

    Returns
    -------
    gpd.GeoDataFrame
        Data on building frontages.
    """
    frontages = []

    for _, block in tqdm(list(blocks.iterrows())):
        blockface_targets, building_targets = _get_block_data(
            block[blocks_uid_col], blockfaces, buildings,
            blockfaces_block_uid_col='block_id', buildings_block_uid_col='block_id'
        )
        for _, blockface_target in blockface_targets.iterrows():
            result = _frontages_on_blockface(
                building_targets, blockface_target, buildings_uid_col=buildings_uid_col,
                blocks_uid_col=blocks_uid_col, blockfaces_uid_col=blockfaces_uid_col
            )
            frontages.append(result)

    frontages = gpd.GeoDataFrame(pd.concat([f for f in frontages if len(f) > 0]))
    if len(frontages) == 0:
        return frontages
    else:
        return frontages.groupby(buildings_uid_col).apply(
            lambda df: df.assign(
                frontage_id=[f'{df.iloc[0].loc[buildings_uid_col]}_{n}' for n in range(len(df))]
            )
        ).reset_index(drop=True)


def select_area_of_interest(blocks, poly):
    """Select all blocks that are within the boundaries given. Utility function."""
    return blocks[blocks.geometry.map(lambda block: poly.contains(block))]


def _distance(line, point):
    """Returns the distance between a line and a point."""
    proj_pos = line.project(point, normalized=True)
    proj_point = line.interpolate(proj_pos, normalized=True)
    dist = proj_point.distance(point)
    return dist


def points_on_frontages(points, frontages):
    """
    Given a sequence of `points` and `frontages`, assigns points to frontages in the dataset.
    """
    # import spaghetti as spgh
    # net = spgh.Network(in_data=frontages)
    # net.snapobservations(points, 'points')
    # snapped_gdf = spgh.element_as_gdf(net, pp_name='points', snapped=True).geometry
    # return snapped_gdf

    index = _create_index(frontages)
    idxs = points.geometry.map(lambda g: list(index.nearest(g.bounds, 5)))
    frontage_groups = idxs.map(lambda idx_group: frontages.iloc[idx_group]).values

    out = []
    for i in range(len(points)):
        point = points.iloc[i]
        frontage_group = frontage_groups[i]

        distances = [_distance(frontage.geometry, point.geometry) for _, frontage 
                     in frontage_group.iterrows()]
        frontage_idx = np.argmin(distances)
        out.append(frontages.iloc[frontage_group.index[frontage_idx]])

    # TODO: cleaner merge op
    return pd.concat(out, axis='columns').T
