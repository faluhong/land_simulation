
import numpy as np
from osgeo import gdal, gdal_array, gdalconst
import os
from os.path import join, exists
import sys
import matplotlib.pyplot as plt
import glob
import click
from pandas import options
from scipy import ndimage
from skimage.morphology import erosion, dilation
from scipy.stats import entropy

# pwd = os.getcwd()
# rootpath = os.path.abspath(os.path.join(pwd, '..'))
# the rootpath in the scratch is /scratch/zhz18039/fah20002/LCM_diversity/
rootpath = r'/scratch/zhz18039/fah20002/LCM_diversity/'

def get_basic_info():
    """get the basic information, projection

    Returns:
        img_landmask: img_landmask 
        nrows: rows of the study area, 10000
        ncols: columns of the study area, 25000
        get_trans
        proj
    """

    obj_land_mask = gdal.Open(join(rootpath, 'data', 'shapefile', 'landmask', 'countryid_hispaniola.tif'))

    ncols = obj_land_mask.RasterXSize
    nrows = obj_land_mask.RasterYSize
    geo_trans = obj_land_mask.GetGeoTransform()
    proj = obj_land_mask.GetProjection()

    img_countrymask = obj_land_mask.ReadAsArray()

    return img_countrymask, nrows, ncols, geo_trans, proj


def land_cover_map_read_hispaniola(year, outputpath_version_flag):
    """read the land cover map in Hispaniola

    Args:
        year (_type_): land cover map year
        outputpath_version_flag (_type_): land cover version 

    Returns:
        _type_: _description_
    """

    filename = join(rootpath, 'results', '{}_landcover_classification'.format(outputpath_version_flag),
                    'mosaic', '{}_{}_landcover.tif'.format(outputpath_version_flag, year))
    img = gdal_array.LoadFile(filename)
    img = img.astype(float)
    
    return img


def process_results_output(filename_output, ncols, nrows, geo_trans, proj, img_output, gdal_type=gdalconst.GDT_Float32):
    """output the process results 

    Args:
        filename_output (_type_): the output file name 
        ncols (_type_): number of columns
        nrows (_type_): number of rows
        geo_trans (_type_): GeoTransform
        proj (_type_): Projection
        img_output (_type_): the image for output
    """

    ds_output = gdal.GetDriverByName('GTiff').Create(filename_output, ncols, nrows, 1, gdal_type, options=['COMPRESS=LZW'])
    ds_output.SetGeoTransform(geo_trans)
    ds_output.SetProjection(proj)

    Band = ds_output.GetRasterBand(1)
    Band.WriteArray(img_output)

    ds_output = None
