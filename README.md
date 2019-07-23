# streetmapper

`streetmapper` is a utility library for working with common forms of geospatial building and block level data. In particular, it is designed for performing joins between the following common types of geospatial data:

* building footprints
* census blocks
* street networks
* point features

`streetmapper` is built mostly using `geopandas` and `shapely` under the hood. It is in the same style as `osmnx`, which provides structured utilities for working with street networks.

This module is currently still a single-purpose library, designed in support of the [`trash-talk`](https://github.com/ResidentMario/trash-talk) project. In the future I would like to perhaps merge the most valuable ideas in this module into the [`pysal` project](https://pysal-spaghetti.readthedocs.io/en/latest/).

## Installation

I recommend installing the underlying geospatial packages with `conda`, followed by installing `streetmapper` with `pip`:

```
conda install -c conda-forge geopandas rtree
pip install git+https://github.com/ResidentMario/streetmapper.git@master
```