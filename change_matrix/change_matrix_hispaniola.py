"""
calculate the transition matrix in the Hispaniola island
"""

import numpy as np
import os
from os.path import join
import time
import sys
from osgeo import gdal_array
import click

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

# landcover_system = {'1': 'Developed',
#                     '2': 'Barren',
#                     '3': 'PrimaryForest',
#                     '4': 'SecondaryForest',
#                     '5': 'ShrubGrass',
#                     '6': 'Cropland',
#                     '7': 'Water',
#                     '8': 'Wetland'}


def change_matrix_generate(array_lc_former, array_lc_latter, landcover_types=9):
    """
        generate the change matrix, compared with confusion_matrix, can ensure 7 types are calculated
    """
    change_matrix = np.zeros((landcover_types, landcover_types), dtype=int)
    for i in range(0, landcover_types):
        for j in range(0, landcover_types):
            mask_tmp = (array_lc_former == i+1) & (array_lc_latter == j+1)
            change_matrix[i,j] = np.count_nonzero(mask_tmp)
    return change_matrix


def matrix_normalized(input_matrix):
    """
    normalize the row of the matrix to make each row sums up to 1
    (1) each row value minus the minimum one avoid the negative values
    (2) after step(1), each row was normalized to make each row sums up to 1
    """

    output_matrix = np.zeros((input_matrix.shape), dtype=float)
    for i_row in range(0, np.shape(input_matrix)[0]):
        if np.sum(input_matrix[i_row, :]) == 0:
            # if one land cover type does not exist in the former date, 1 will be assigned to the diagonal value
            # e.g., develop does not exist in 1984, the change_matrix for develop (type 1) will be [0, 0, 0, 0, 0, 0, 0]
            # after the normalization, the transition matrix will be [1, 0, 0, 0, 0, 0, 0]
            # output_matrix[i_row, :] = 0
            output_matrix[i_row, i_row] = 1
        else:
            tmp = input_matrix[i_row, :].copy()
            output_matrix[i_row, :] = tmp / np.sum(tmp)

    return output_matrix


def land_cover_map_read_hispaniola(year, outputpath_version_flag, folder='mosaic', country_flag='hispaniola'):

    filename_country_id = join(rootpath_project, 'data', 'shapefile', 'landmask', 'countryid_hispaniola.tif')
    img_country_id = gdal_array.LoadFile(filename_country_id)

    filename = join(rootpath_project, 'results', '{}_landcover_classification'.format(outputpath_version_flag),
                    folder, '{}_{}_landcover.tif'.format(outputpath_version_flag, year))
    img = gdal_array.LoadFile(filename)
    img = img.astype(float)
    
    if country_flag == 'hispaniola':
        img[img_country_id == 0] = np.nan
    elif country_flag == 'haiti':
        img[img_country_id != 1] = np.nan
    elif country_flag == 'dr':
        img[img_country_id != 2] = np.nan

    return img


def transition_prob_matrix_two_times(img_lc_t1, img_lc_t2, landcover_types=9):
    change_matrix_t1_t2 = change_matrix_generate(img_lc_t1, img_lc_t2, landcover_types=landcover_types)
    transition_prob_matrix_t1_t2 = matrix_normalized(change_matrix_t1_t2)

    class change_stats:
        def __init__(self, change_matrix_t1_t2, transition_prob_matrix_t1_t2):
            self.change_matrix = change_matrix_t1_t2
            self.transition_prob_matrix = transition_prob_matrix_t1_t2

        def print_change_matrix(self):
            print(self.change_matrix)
            print(self.transition_prob_matrix)

    output_change = change_stats(change_matrix_t1_t2, transition_prob_matrix_t1_t2)

    return output_change


@click.command()
@click.option('--outputpath_version_flag', type=str, default='irf_v13_0001_01_15000', help='output path (land cover) version flag')
@click.option('--folder', type=str, default='mosaic_mmu_3_3', help='the folder in the land cover classification folder indicating the MMU processing')
@click.option('--predict_flag', type=str, default='hindcast', help='the prediction flag, forecast or hindcast')
@click.option('--country_flag', type=str, default='hispaniola', help='the country flag, haiti or dr or hispaniola')
def main(outputpath_version_flag, predict_flag, folder, country_flag):

# if __name__ == '__main__':
#     outputpath_version_flag = 'irf_v42_5'
#     predict_flag = 'forecast'
#     country_flag = 'haiti'    

    start_time = time.perf_counter()
    np.set_printoptions(precision=4, suppress=True)
    
    # folder = 'mosaic_mmu_3_3'

    if predict_flag == 'forecast':
        list_year = np.arange(1996, 2023)
    else:
        list_year = np.arange(2022, 1995, -1)

    landcover_types = 9

    transition_prob_matrix_adjacent = np.zeros((len(list_year), landcover_types, landcover_types), dtype=float)
    transition_prob_matrix_adjacent[0, :, :] = np.identity(landcover_types)

    transition_prob_matrix_accumulate_indirect = transition_prob_matrix_adjacent.copy()
    transition_prob_matrix_accumulate_direct = transition_prob_matrix_adjacent.copy()

    transition_prob_matrix_tmp = np.identity(landcover_types)
    for i_year in range(0, len(list_year) - 1):
    # for i_year in range(1, 2):
        year = list_year[i_year]
        print(year)

        img_lc_t1 = land_cover_map_read_hispaniola(list_year[i_year], outputpath_version_flag, folder, country_flag)
        img_lc_t2 = land_cover_map_read_hispaniola(list_year[i_year + 1], outputpath_version_flag, folder, country_flag)

        change_stats_adjacent = transition_prob_matrix_two_times(img_lc_t1, img_lc_t2, landcover_types=landcover_types)
        transition_prob_matrix_adjacent[i_year + 1] = change_stats_adjacent.transition_prob_matrix

        transition_prob_matrix_tmp = transition_prob_matrix_tmp @ change_stats_adjacent.transition_prob_matrix
        transition_prob_matrix_accumulate_indirect[i_year + 1] = transition_prob_matrix_tmp

        img_lc_start = land_cover_map_read_hispaniola(list_year[0], outputpath_version_flag, folder, country_flag)
        transition_prob_matrix_accumulate_direct[i_year + 1] \
            = transition_prob_matrix_two_times(img_lc_start, img_lc_t2).transition_prob_matrix

    output_path = join(rootpath_project, 'results', 'land_change_modelling', outputpath_version_flag, 'change_matrix')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_adjacent_prob_matrix = join(output_path, '{}_{}_adjacent_matrix.npy'.format(predict_flag, country_flag))
    np.save(output_adjacent_prob_matrix, transition_prob_matrix_adjacent)

    output_accumulate_prob_matrix_indirect = join(output_path,
                                                    '{}_{}_accumulate_matrix_indirect.npy'.format(predict_flag, country_flag))
    np.save(output_accumulate_prob_matrix_indirect, transition_prob_matrix_accumulate_indirect)

    output_accumulate_prob_matrix_direct = join(output_path,
                                                '{}_{}_accumulate_matrix_direct.npy'.format(predict_flag, country_flag))
    np.save(output_accumulate_prob_matrix_direct, transition_prob_matrix_accumulate_direct)

    end_time = time.perf_counter()
    print('running time: {} second'.format(end_time - start_time))


if __name__ == '__main__':
    main()
