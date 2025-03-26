
import numpy as np
from os.path import join
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import sys
import click
import logging

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from Basic_tools.Figure_plot import FP

from land_change_modelling.change_matrix_hispaniola import land_cover_map_read_hispaniola
from land_change_modelling.hispaniola_prepare_predictors import get_basic_info
from Landcover_classification.landcover_function import land_cover_plot


MAX_LC = 8
MIN_LC = 1

landcover_system = {'1': 'Developed',
                    '2': 'Barren',
                    '3': 'PrimaryForest',
                    '4': 'SecondaryForest',
                    '5': 'ShrubGrass',
                    '6': 'Cropland',
                    '7': 'Water',
                    '8': 'Wetland'}


def pixel_count(img):
    """
        count the pixel number for each land cover type
        Args:
            img
        Returns:
    """
    array_pixel_count = np.zeros(MAX_LC - MIN_LC + 1, dtype=int)
    for p in range(MIN_LC, MAX_LC + 1):
        array_pixel_count[p - 1] = np.count_nonzero(img == p)

    return array_pixel_count


def get_change_row_col(img_change_prob, change_pixel_count, img_change_flag, mask_prediction):
    nrow, ncol = np.shape(img_change_prob)
    if nrow > ncol:
        tmp = nrow
    else:
        tmp = ncol

    valid_row_col = np.where(mask_prediction & (img_change_flag == 0))
    valid_pos = valid_row_col[0] * tmp + valid_row_col[1]

    array_valid_change_prob = img_change_prob[valid_row_col[0], valid_row_col[1]]

    mask_pos = np.argsort(array_valid_change_prob)[::-1][0:change_pixel_count]
    pos_select = valid_pos[mask_pos]

    row_select = pos_select // tmp
    col_select = pos_select - row_select * tmp

    return row_select, col_select


def read_change_prob_map(path_change_prob, landcover_id_from, nrows=10000, ncols=22500):
    
    img_change_prob = np.zeros((MAX_LC, nrows, ncols), dtype=float)
    
    for landcover_id_to in range(MIN_LC, MAX_LC + 1):
        
        change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                                    landcover_id_to, landcover_system[str(landcover_id_to)])
        
        filename_transition_prob = '{}_{}.tif'.format(path_change_prob, change_info)
    
        if os.path.exists(filename_transition_prob) is False:
            pass
            logging.info('transition potential layer for {} does not exist'.format(change_info))
        else:  
            logging.info('read transition potential layer {} '.format(filename_transition_prob))
            img_change_prob[landcover_id_to -1, :, :] = gdal_array.LoadFile(filename_transition_prob)
    
    return img_change_prob


def predict_land_cover_generate_parallel(img_lc_current, current_count, predict_matrix_employed, 
                                         predict_matrix_sum,
                                         path_change_prob):

    img_lc_predict = np.zeros(np.shape(img_lc_current), dtype=float)
    # img_lc_predict[img_landmask == 0] = np.nan

    img_change_flag = img_lc_predict.copy()
    
    # predict_matrix_employed_sum = np.nansum(predict_matrix_employed, axis=0)

    for landcover_id_from in range(MIN_LC, MAX_LC + 1):

        mask_prediction = img_lc_current == landcover_id_from

        # for landcover_id_to in range(MIN_LC, MAX_LC + 1):
        
        transition_pct = predict_matrix_sum[landcover_id_from - 1, :]
        list_landcover_id_to = np.argsort(transition_pct) + 1

        for i_landcover_id_to in range(0, len(list_landcover_id_to)):
            landcover_id_to = list_landcover_id_to[i_landcover_id_to]

            change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                                       landcover_id_to, landcover_system[str(landcover_id_to)])
            
            filename_transition_prob = '{}_{}.tif'.format(path_change_prob, change_info)
            logging.info(os.path.split(filename_transition_prob)[-1])

            if predict_matrix_employed[landcover_id_from - 1, landcover_id_to - 1] == 0:
                pass
                # print('land cover transition does not happen for {}'.format(change_info))
            elif not os.path.exists(filename_transition_prob):
                pass
                # print('transition potential layer for {} does not exist'.format(change_info))
            else:
                img_change_prob = gdal_array.LoadFile(filename_transition_prob)
                # img_change_prob[img_change_prob == 0] = np.nan

                change_pixel_count = int(np.ceil(
                    current_count[landcover_id_from - 1]
                    * predict_matrix_employed[landcover_id_from - 1, landcover_id_to - 1]))

                nrow, ncol = np.shape(img_change_prob)
                if nrow > ncol:
                    tmp = nrow
                else:
                    tmp = ncol

                valid_row_col = np.where(mask_prediction & (img_change_flag == 0))
                valid_pos = valid_row_col[0] * tmp + valid_row_col[1]

                array_valid_change_prob = img_change_prob[valid_row_col[0], valid_row_col[1]]

                mask_pos = np.argsort(array_valid_change_prob)[::-1][0:change_pixel_count]

                pos_select = valid_pos[mask_pos]

                row_select = pos_select // tmp
                col_select = pos_select - row_select * tmp

                img_change_flag[row_select, col_select] = 1

                img_lc_predict[row_select, col_select] = landcover_id_to

    return img_lc_predict


def output_predict_land_cover(img_lc_predict,
                              src_geotrans, src_proj,
                              path_output, output_basicname):

    # output_filename = join(rootpath_modelling, predict_flag, classifier_flag,
    #                        '{}_{}_{}.tif'.format(classifier_flag, predict_flag, predict_date))
    
    output_filename = join(path_output, '{}.tif'.format(output_basicname))

    abspath = os.path.abspath(join(output_filename, os.pardir))
    if not os.path.exists(abspath):
        os.makedirs(abspath, exist_ok=True)

    tif_out = gdal.GetDriverByName('GTiff').Create(output_filename, np.shape(img_lc_predict)[1],
                                                   np.shape(img_lc_predict)[0], 1, gdalconst.GDT_Byte)

    tif_out.SetGeoTransform(src_geotrans)
    tif_out.SetProjection(src_proj)

    band = tif_out.GetRasterBand(1)
    band.WriteArray(img_lc_predict)

    tif_out = None
    
    return output_filename


@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
def main(rank, n_cores):
    
# if __name__ == '__main__':
    landcover_version = 'irf_v52_0_5'
    predict_flag = 'hindcast'
    modelling_flag = 'alpha_0035_accumulate_matrix'
    # classifier_flag = 'rf'
    change_prob_flag = 'v2'
    modelling_folder = 'alpha_0035'

    np.set_printoptions(precision=4, suppress=True)

    rootpath_modelling = join(rootpath_project, 'results', 'land_change_modelling', landcover_version)
    path_output = join(rootpath_modelling, predict_flag, modelling_folder)
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    path_change_prob = join(rootpath_modelling, predict_flag, 'change_prob', change_prob_flag,'{}_{}'.format(predict_flag, change_prob_flag))
    
    logging.basicConfig(filename=join(path_output, '{}_{}_{}_predict.log'.format(rank, landcover_version, modelling_folder)),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.info('land cover output version: {}'.format(landcover_version))
    logging.info('prediction flag (forecast/hindcast): {}'.format(predict_flag))
    logging.info('modelling flag: {}'.format(modelling_flag))
    logging.info('change probability path: {}'.format(path_change_prob))
    logging.info('modelling folder: {}'.format(modelling_folder))

    transition_matrix = np.load(join(rootpath_modelling, 'change_matrix', '{}_hispaniola_accumulate_matrix_direct.npy'.format(predict_flag)))
    # predict_matrix = np.load(join(rootpath_modelling, predict_flag, 'prediction_matrix','matrix_{}.npy').format(modelling_flag))
    
    predict_matrix = np.load(join(rootpath_modelling, predict_flag, 'prediction_matrix', modelling_folder, '{}.npy').format(modelling_flag))
    
    img_landmask, nrows, ncols, src_geotrans, src_proj = get_basic_info()
    
    predict_matrix_sum = np.nansum(predict_matrix, axis=0)

    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_prediction_year = np.arange(2022, 2123, 1)
        img_lc_current = land_cover_map_read_hispaniola(2022, landcover_version)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        # list_prediction_year = np.arange(1996, 1895, -1)
        # list_prediction_year = np.arange(1996, 1499, -1)
        list_prediction_year = np.arange(1996, 1491, -1)

        img_lc_current = land_cover_map_read_hispaniola(1996, landcover_version)

    current_count = pixel_count(img_lc_current)

    logging.info('predict year is {}'.format(list_prediction_year))
    
    each_core_task = int(np.ceil(len(list_prediction_year) / n_cores))

    for i in range(0, each_core_task):
        new_rank = rank - 1 + i * n_cores
        print(new_rank)
        if new_rank > len(list_prediction_year) - 1:  # means that all folder has been processed
            print('this is the last running task')
            break
    
        predict_year = list_prediction_year[new_rank]
        logging.info('predict for {}'.format(predict_year))

        predict_matrix_employed = predict_matrix[list_prediction_year == predict_year][0, :, :]
        logging.info('the transition matrix from 2022 to {} is'.format(predict_year))
        logging.info(predict_matrix_employed)
        
        img_lc_predict = predict_land_cover_generate_parallel(img_lc_current, current_count, 
                                                              predict_matrix_employed, predict_matrix_sum,
                                                              path_change_prob)

        output_basicname = '{}_{}_{}_{}_{}'.format(landcover_version, modelling_folder, modelling_flag, change_prob_flag, predict_year)

        # land_cover_plot(img_lc_predict, title=output_basicname, outputflag=0, path_output=join(path_output, 'figure'))

        output_filename = output_predict_land_cover(img_lc_predict,
                                                    src_geotrans, src_proj,
                                                    path_output, output_basicname)

if __name__ == '__main__':
    main()

