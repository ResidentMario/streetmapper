import numpy as np
from .pipeline import get_block_data


def plot_building_block_multimatch(multimatches, blocks, building_id_key="sf16_BldgID", id_value=None):
    """
    Plots a single building matching multiple blocks.

    Parameters
    ----------
    multimatches: gpd.GeoDataFrame
        A `GeoDataFrame` of multimatches, as would be returned by `join_bldgs_blocks`.

    blocks
        A `GeoDataFrame` of blocks in the area of interest.

    building_id_key: str, default "sf16_BldgID"
        The key corresponding with the building ID.

    id_value: str, default None
        If non-null, that specific building will be plotted, otherwise, a random building will be plotted.

    Returns
    -------
    A matplotlib.pyplot.Axes instance which plots the requested data.
    """
    choice = np.random.choice(multimatches.loc[:, building_id_key]) if not id_value else id_value

    ax = (multimatches
          .query(f'{building_id_key} == "{choice}"')
          .pipe(lambda df: df.assign(
              block_geometry=df.apply(lambda srs: blocks.iloc[int(srs['index_right'])].geometry, axis='columns'))
          )
          .set_geometry("block_geometry")
          .plot(linewidth=0.5, edgecolor='white', color='lightgray'))

    (multimatches
        .query(f'{building_id_key} == "{choice}"')
        .head(1)
        .plot(ax=ax, color='black'))

    return ax


def plot_building_block_multimatches(multimatches, blocks, xlim=None, ylim=None):
    """
    Plots all buildings matching multiple blocks.

    Parameters
    ----------
    multimatches: gpd.GeoDataFrame
        A `GeoDataFrame` of multimatches, as would be returned by `join_bldgs_blocks`.

    blocks
        A `GeoDataFrame` of blocks in the area of interest.

    xlim: None or tuple
        A two-element tuple with lower and upper bounds on the x-axis for the resultant plot.

    ylim: None or tuple
        A two-element tuple with lower and upper bounds on the y-axis for the resultant plot.

    Returns
    -------
    A matplotlib.pyplot.Axes instance which plots the requested data.
    """
    if not xlim:
        xlim = [-122.525, -122.35]
    if not ylim:
        ylim = [37.7, 37.85]

    ax = blocks.plot(color='None', linewidth=1, edgecolor='lightgray', figsize=(12, 12))
    multimatches.plot(figsize=(12, 12), color='red', ax=ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


# TODO
def plot_building_block_nonmatch():
    pass


# TODO
def plot_building_block_nonmatches():
    pass


# TODO
def plot_unmatched_blockface():
    pass


# TODO
def plot_unmatched_blockfaces():
    pass


# TODO: docstring etcetera
def plot_block(block_id, street_segments, blockfaces, buildings):
    street_segments, blockfaces, buildings = get_block_data(block_id, street_segments, blockfaces, buildings)

    ax = street_segments.plot(color='red', linewidth=1)
    blockfaces.plot(color='black', ax=ax, linewidth=1)
    buildings.plot(ax=ax, color='lightsteelblue', linewidth=1, edgecolor='steelblue')
    return ax
