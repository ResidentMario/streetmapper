from setuptools import setup
setup(
    name='streetmapper',
    packages=['streetmapper'],
    install_requires=['pandas', 'geopandas', 'rtree', 'tqdm'],
    py_modules=['pipeline', 'utils', 'ops'],
    version='0.0.1',
    description='Data processing and visualization utilities for working with trash data',
    author='Aleksey Bilogur',
    author_email='aleksey.bilogur@gmail.com',
    url='https://github.com/ResidentMario/streetmapper',
    keywords=['data', 'data visualization', 'data analysis', 'geospatial analysis', 'gis'],
    classifiers=[]
)
