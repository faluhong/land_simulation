"""
prepare the predictors, which include
(1) distance to the target land cover, the morphology flag is set defautly as 1
(2) land cover percentage in the window size, the default is 26
"""

import numpy as np
from osgeo import gdal, gdal_array, gdalconst
import os
from os.path import join, exists
import sys
import matplotlib.pyplot as plt
import glob
import click
from scipy import ndimage
from skimage.morphology import erosion, dilation
from scipy.stats import entropy

# pwd = os.getcwd()
# rootpath = os.path.abspath(os.path.join(pwd, '..'))
rootpath = r'/scratch/zhz18039/fah20002/LCM_diversity/'

from change_probability_map.utils_model import get_basic_info, land_cover_map_read_hispaniola, process_results_output


def dist_to_target_value(landcover_version: str, year: int, target_value: int, morphology_flag=1):
    """calculate the distance (unit: meter) to the target land cover value

    Args:
        landcover_version (str): the land cover version
        year (int): the land cover year
        target_value (int): the target land cover value
        morphology_flag (int, optional): the morpohology. Defaults to 1. 0: No morphology; 1: Morphology with circle 1; 2: circle 2
        country_flag(str, optional): the country flag. Defaults to 'hispaniola'.
    Returns:
        _type_: _description_
    """

    img_lc = land_cover_map_read_hispaniola(year, landcover_version)

    img_lc[img_lc != target_value] = 9999
    img_lc[img_lc == target_value] = 0
    img_lc[img_lc == 9999] = 1

    if morphology_flag == 0:
        dist_to_target_value = ndimage.distance_transform_edt(img_lc, return_indices=False)
    elif morphology_flag == 1:
        cross1 = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0], ])
        img_lc_dilation1 = dilation(img_lc, cross1)
        img_lc_erosion1 = erosion(img_lc_dilation1, cross1)
        # FP(img_lc_erosion2, title='lc circle 1')
        dist_to_target_value = ndimage.distance_transform_edt(img_lc_erosion1, return_indices=False)

    elif morphology_flag == 2:
        cross2 = np.array([[0, 0, 1, 0, 0],
                           [0, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 0, 0],
                           ])
        img_lc_dilation2 = dilation(img_lc, cross2)
        img_lc_erosion2 = erosion(img_lc_dilation2, cross2)
        # FP(img_lc_erosion2, title='lc circle 2')
        dist_to_target_value = ndimage.distance_transform_edt(img_lc_erosion2, return_indices=False)

    dist_to_target_value = dist_to_target_value * 30
    # FP(dist_to_target_value, title='morphology flag {}'.format(morphology_flag))

    return dist_to_target_value


def determine_chip_location(i_row, nrows, window_radius_size=26):
    """
    get the boundary of the chip location, can be used for both the row and column

    Args:
        i_row (_type_): row/col id of the central pixel
        nrows (_type_): total row/col number
        window_radius_size (int, optional): the radius of the window size. Defaults to 26, i.e., the window size is 51 by 51

    Returns:
        row_start:
        row_end:
    """

    if i_row in np.arange(0, window_radius_size - 1):
        row_start = 0
        row_end = i_row + window_radius_size
    elif i_row in np.arange(nrows - window_radius_size, nrows):
        row_start = i_row - window_radius_size + 1
        row_end = nrows
    else:
        row_start = i_row - window_radius_size + 1
        row_end = i_row + window_radius_size

    return row_start, row_end


def calculate_pct(img_lc, landcover_id, window_radius_size, img_countrymask, nrows, ncols, type_flag='lc_pct'):
    """calculate the land cover percentage or entropy to represent the heterogeneity for the assigned land cover id

    Args:
        img_lc (_type_): land cover map
        landcover_id (_type_): the assigned land cover id
        window_radius_size (_type_): _description_
        img_countrymask (_type_): _description_
        nrows (_type_): _description_
        ncols (_type_): _description_
        type_flag: the value you want to get, Default of 'lc_pct', i.e., the land cover percentage, otherwise, the entropy

    Returns:
        _type_: _description_
    """
    img_lc_pct = np.zeros((nrows, ncols), dtype=float)

    for i_row in range(0, nrows):
        for i_col in range(0, ncols):

            if img_countrymask[i_row, i_col] == 0:
                pass
            else:
                # get the boundary of the chip location
                row_start, row_end = determine_chip_location(i_row, nrows, window_radius_size=window_radius_size)
                col_start, col_end = determine_chip_location(i_col, ncols, window_radius_size=window_radius_size)

                img_chip = img_lc[row_start: row_end, col_start: col_end].copy()
                # print(i_row, i_col, row_start, row_end, col_start, col_end, img_chip.shape)

                if type_flag == 'lc_pct':
                    lc_pct = np.count_nonzero(img_chip == landcover_id) / img_chip.size
                    img_lc_pct[i_row, i_col] = lc_pct
                else:
                    # use entropy to define the heterogeneity, not recommended after testing
                    list_unique = np.unique(img_chip.ravel())
                    array_pct = np.zeros(len(list_unique), dtype=float)

                    for i_element, element in enumerate(list_unique):
                        array_pct[i_element] = np.count_nonzero(img_chip.ravel() == element) / img_chip.size

                    img_lc_pct[i_row, i_col] = entropy(array_pct)

    return img_lc_pct


def lc_pct_output(year, img_lc, lc_target_value, window_radius_size, path_output,
                  img_countrymask, nrows, ncols, geo_trans, proj):
    """
        output the land cover percentage for the assigned land cover id
    """

    img_lc_pct = calculate_pct(img_lc, lc_target_value, window_radius_size, img_countrymask, nrows, ncols)

    filename_lc_pct_output = join(path_output, '{}_pct_{}_{}.tif'.format(year, lc_target_value, window_radius_size))
    process_results_output(filename_lc_pct_output, ncols, nrows, geo_trans, proj, img_lc_pct)

    return img_lc_pct


# def main():
if __name__ == '__main__':
    landcover_version = 'degrade_v2_refine_3_3'

    path_output = join(rootpath, 'results', 'land_change_modelling', landcover_version, 'predictor_variables')
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    img_countrymask, nrows, ncols, geo_trans, proj = get_basic_info()

    # for year in [1996, 2022]:  # can be used for other years, usually 2022 for forecasting
    for year in [1996]:
        
        print(year)
        
        for target_value in range(1, 10):
            
            morphology_flag = 1
            print(year, target_value, morphology_flag)
            
            dist_to_lc = dist_to_target_value(landcover_version, year, target_value, morphology_flag)
    
            if morphology_flag == 0:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}.tif'.format(year, target_value))
            elif morphology_flag == 1:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}_mcircle_1.tif'.format(year, target_value))
            elif morphology_flag == 2:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}_mcircle_2.tif'.format(year, target_value))
            
            process_results_output(filename_recent_year_output, ncols, nrows, geo_trans, proj, dist_to_lc)
    
        window_radius_size = 26

        img_lc = land_cover_map_read_hispaniola(year, landcover_version)
        
        img_lc_pct_1 = lc_pct_output(year, img_lc, 1, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_2 = lc_pct_output(year, img_lc, 2, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_3 = lc_pct_output(year, img_lc, 3, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_4 = lc_pct_output(year, img_lc, 4, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_5 = lc_pct_output(year, img_lc, 5, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_6 = lc_pct_output(year, img_lc, 6, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_7 = lc_pct_output(year, img_lc, 7, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_8 = lc_pct_output(year, img_lc, 8, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_9 = lc_pct_output(year, img_lc, 9, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)

    