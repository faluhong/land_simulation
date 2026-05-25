"""
    utility function to read the land cover map and projection information of Hispaniola
"""

import numpy as np
import os
from os.path import join
import sys
from osgeo import gdal_array, gdal
import pandas as pd

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)


def land_cover_map_read_hispaniola(year, country_flag='hispaniola'):
    """
        load the land cover map

        :param year: the year of the land cover map, e.g., 1984, 1990, 1995, 2000, 2005, 2010, 2015, 2018
        :param country_flag: 'hispaniola' for the whole island, 'haiti' for Haiti, 'dr' for the Dominican Republic
        :return:
    """

    filename_country_id = join(rootpath_project, 'data', 'countryid_hispaniola.tif')
    img_country_id = gdal_array.LoadFile(filename_country_id)

    # the land cover map of Hispaniola can be downloaded from https://doi.org/10.6084/m9.figshare.28100408
    filename_lc = join(rootpath_project, 'data', 'hispaniola', f'hispaniola_lc_{year}.tif')

    img = gdal_array.LoadFile(filename_lc)
    img = img.astype(float)

    if country_flag == 'hispaniola':
        img[img_country_id == 0] = np.nan
    elif country_flag == 'haiti':
        img[img_country_id != 1] = np.nan
    elif country_flag == 'dr':
        img[img_country_id != 2] = np.nan
    else:
        raise ValueError('country_flag should be "hispaniola", "haiti", or "dr"')

    return img


def get_basic_info():
    """get the basic projection information and country mask of the study area

    Returns:
        img_landmask: img_landmask
        nrows: rows of the study area, 10000
        ncols: columns of the study area, 25000
        get_trans
        proj
    """

    obj_land_mask = gdal.Open(join(rootpath_project, 'data', 'countryid_hispaniola.tif'))

    ncols = obj_land_mask.RasterXSize
    nrows = obj_land_mask.RasterYSize
    geo_trans = obj_land_mask.GetGeoTransform()
    proj = obj_land_mask.GetProjection()

    img_countrymask = obj_land_mask.ReadAsArray()

    return img_countrymask, nrows, ncols, geo_trans, proj


def read_obs_pct_file(list_observe_year, country_flag='hispaniola'):
    """
    read the land cover statistical file

    Args:
        list_observe_year (_type_): the observation year, e.g., np.arange(1984, 2023), np.arange(1996, 2023)
        country_flag: the country flag, e.g., 'hispaniola', 'dr', 'haiti'

    Returns:
        sheet_hispaniola: dataframe containing each land cover percentage
    """
    filename_percentile = join(rootpath_project, 'data',  'hispaniola_landcover_analysis.xlsx')

    if country_flag == 'hispaniola':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Hispaniola')
    elif country_flag == 'dr':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Dominican')
    elif country_flag == 'haiti':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Haiti')

    sheet_hispaniola = sheet_hispaniola.loc[sheet_hispaniola['Year'].isin(list_observe_year)]

    return sheet_hispaniola


if __name__=='__main__':

    country_flag = 'haiti'
    year = 2018

    img_lc = land_cover_map_read_hispaniola(year, country_flag)

    # show the land cover map
    import matplotlib.pyplot as plt

    figure = plt.imshow(img_lc, interpolation='nearest')

    plt.title(f'land cover of {country_flag} in {year}', fontsize=14)
    plt.show()





