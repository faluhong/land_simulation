"""
    prepare the predictor variables for the land change simulation model

    This code is for the multi-land cover types, i.e., 8 land cover types
"""

import numpy as np
import os
from os.path import join
import time
import sys
from osgeo import gdal_array
import click
from osgeo import gdal, gdal_array, gdalconst
import os
from os.path import join, exists
import sys
import matplotlib.pyplot as plt
import glob
import click
from scipy import ndimage
from skimage.morphology import erosion, dilation
from scipy.ndimage import uniform_filter

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from change_matrix.util_read_hispaniola_lc import land_cover_map_read_hispaniola, get_basic_info


def dist_to_target_value(img_lc_input, target_value: int, morphology_flag=1):
    """calculate the distance (unit: meter) to the target land cover value

        Morphological dilation and erosion are applied to reduce the salt-and-pepper noise in the land cover map

    Args:
        img_lc_input: the land cover map
        target_value (int): the target land cover value
        morphology_flag (int, optional): the morphology. Defaults to 1. 0: No morphology; 1: Morphology with circle 1; 2: circle 2
    """

    img_lc = img_lc_input.copy()

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

    return dist_to_target_value


def calculate_pct(img_lc, landcover_id, window_radius_size, img_countrymask):
    """calculate the land cover percentage or entropy to represent the heterogeneity for the assigned land cover id

    Args:
        img_lc (_type_): land cover map
        landcover_id (_type_): the assigned land cover id
        window_radius_size (_type_): _description_
        img_countrymask (_type_): _description_
    Returns:
        _type_: _description_
    """

    window_size = 2 * window_radius_size - 1
    window_area = window_size ** 2

    # make a copy of the land cover map
    img_lc_test = img_lc.copy()
    img_lc_test = img_lc_test.astype(float)

    # convert the land cover map to the binary map based on the assigned land cover id
    img_lc_test[img_lc_test != landcover_id] = 9999
    img_lc_test[img_lc_test == landcover_id] = 1
    img_lc_test[img_lc_test == 9999] = 0

    # uniform_filter computes the mean, so multiply by the window area to get the sum
    img_sum_array = uniform_filter(img_lc_test.astype(float), size=window_size) * window_area

    # Calculate the percentage of 1s in each window
    img_lc_pct = img_sum_array / window_area
    img_lc_pct[img_countrymask == 0] = np.nan

    return img_lc_pct


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


def lc_pct_output(year, img_lc, lc_target_value, window_radius_size, path_output,
                  img_countrymask, nrows, ncols, geo_trans, proj):
    """
        output the land cover percentage for the assigned land cover id
    """

    img_lc_pct = calculate_pct(img_lc, lc_target_value, window_radius_size, img_countrymask)

    filename_lc_pct_output = join(path_output, '{}_pct_{}_{}.tif'.format(year, lc_target_value, window_radius_size))
    process_results_output(filename_lc_pct_output, ncols, nrows, geo_trans, proj, img_lc_pct)

    return img_lc_pct



@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
# if __name__ == '__main__':

    landcover_version = 'publish_v1'

    path_output = join(rootpath_project, 'results', 'land_change_modelling', landcover_version, 'predictor_variables')
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    img_countrymask, nrows, ncols, geo_trans, proj = get_basic_info()

    list_year = np.arange(1996, 2022)

    each_core_task = int(np.ceil(len(list_year) / n_cores))
    for i in range(0, each_core_task):
        new_rank = rank - 1 + i * n_cores
        print(new_rank)
        if new_rank > len(list_year) - 1:  # means that all folder has been processed
            print('this is the last running task')
            break

        year = list_year[new_rank]
        print(year)

        img_lc = land_cover_map_read_hispaniola(year, country_flag='hispaniola')

        # get the distance to the target land cover value
        for target_value in range(1, 9):

            morphology_flag = 1
            print(year, target_value, morphology_flag)

            dist_to_lc = dist_to_target_value(img_lc, target_value, morphology_flag)

            if morphology_flag == 0:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}.tif'.format(year, target_value))
            elif morphology_flag == 1:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}_mcircle_1.tif'.format(year, target_value))
            elif morphology_flag == 2:
                filename_recent_year_output = join(path_output, '{}_dist_to_{}_mcircle_2.tif'.format(year, target_value))

            process_results_output(filename_recent_year_output, ncols, nrows, geo_trans, proj, dist_to_lc)

        # output the land cover percentage for each land cover type
        window_radius_size = 26

        img_lc_pct_1 = lc_pct_output(year, img_lc, 1, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_2 = lc_pct_output(year, img_lc, 2, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_3 = lc_pct_output(year, img_lc, 3, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_4 = lc_pct_output(year, img_lc, 4, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_5 = lc_pct_output(year, img_lc, 5, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_6 = lc_pct_output(year, img_lc, 6, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_7 = lc_pct_output(year, img_lc, 7, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)
        img_lc_pct_8 = lc_pct_output(year, img_lc, 8, window_radius_size, path_output, img_countrymask, nrows, ncols, geo_trans, proj)


if __name__ == '__main__':
    main()