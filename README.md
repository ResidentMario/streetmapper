# streetmapper

`streetmapper` is a utility library for working with common forms of geospatial building and block level data. In particular, it is designed to make working with (and integrating) the following types of data easy:

* building footprints
* census blocks
* street networks
* point features

`streetmapper` is built mostly using `geopandas` and `shapely` under the hood. It is in the same style as the ([more mature](https://github.com/gboeing/osmnx)) `osmnx` package, which provides structured utilities for working with street networks.

Currently under active development. Check back soon!

## Installation

I recommend installing the underlying geospatial packages with `conda`, followed by installing `streetmapper` with `pip`:

```
conda install -c conda-forge geopandas rtree
pip install git+https://github.com/ResidentMario/streetmapper.git@master
```