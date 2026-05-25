"""
    predict the land cover map for each year based on the change probability layer and the predicted change matrix
"""

import numpy as np
from os.path import join
from osgeo import gdal, gdal_array, gdalconst
import matplotlib.pyplot as plt
import os
import sys
import logging

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)


from change_matrix.util_read_hispaniola_lc import land_cover_map_read_hispaniola, get_basic_info


def pixel_count(img, MIN_LC=1, MAX_LC=8):
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
    """
        get the row and column of the pixels to be changed based on the change probability layer

        :param img_change_prob:
        :param change_pixel_count:
        :param img_change_flag:
        :param mask_prediction:
        :return:
    """

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


def predict_land_cover(img_lc_current,
                       current_count,
                       predict_matrix_employed,
                       predict_matrix_sum,
                       file_prefix_change_prob,
                       img_country_mask,
                       country_id,
                       MIN_LC=1,
                       MAX_LC=8,
                       landcover_system=None, ):
    """
        predict the land cover map
    """

    # array to store the predicted land cover map, initialized as the current land cover map
    img_lc_predict = np.zeros(np.shape(img_lc_current), dtype=float)

    # array to store the change flag, meaning whether the pixel has been changed or not
    img_change_flag = img_lc_predict.copy()


    for landcover_id_from in range(MIN_LC, MAX_LC + 1):

        mask_prediction = img_lc_current == landcover_id_from

        # get transition percentage to determine the order to allocate the land cover
        transition_pct = predict_matrix_sum[landcover_id_from - 1, :]
        list_landcover_id_to = np.argsort(transition_pct) + 1

        # allocate the land cover type based on the sorted order
        for i_landcover_id_to in range(0, len(list_landcover_id_to)):
            landcover_id_to = list_landcover_id_to[i_landcover_id_to]

            change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                                       landcover_id_to, landcover_system[str(landcover_id_to)])

            filename_transition_prob = '{}_{}.tif'.format(file_prefix_change_prob, change_info)
            assert os.path.exists(filename_transition_prob), 'transition potential layer for {} does not exist'.format(change_info)

            if predict_matrix_employed[landcover_id_from - 1, landcover_id_to - 1] == 0:
                pass
                # print('land cover transition does not happen for {}'.format(change_info))
            elif not os.path.exists(filename_transition_prob):
                pass
                # print('transition potential layer for {} does not exist'.format(change_info))
            else:
                img_change_prob = gdal_array.LoadFile(filename_transition_prob)
                # img_change_prob[img_change_prob == 0] = np.nan

                change_pixel_count = int(np.ceil(current_count[landcover_id_from - 1] * predict_matrix_employed[landcover_id_from - 1, landcover_id_to - 1]))

                nrow, ncol = np.shape(img_change_prob)
                if nrow > ncol:
                    tmp = nrow
                else:
                    tmp = ncol

                valid_row_col = np.where(mask_prediction & (img_change_flag == 0) & (img_country_mask == country_id))
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
    """
        output the predicted land cover map as a tif file

        :param img_lc_predict:
        :param src_geotrans:
        :param src_proj:
        :param path_output:
        :param output_basicname:
        :return:
    """

    output_filename = join(path_output, '{}.tif'.format(output_basicname))

    abspath = os.path.abspath(join(output_filename, os.pardir))
    if not os.path.exists(abspath):
        os.makedirs(abspath, exist_ok=True)

    tif_out = gdal.GetDriverByName('GTiff').Create(output_filename, np.shape(img_lc_predict)[1],
                                                   np.shape(img_lc_predict)[0], 1,
                                                   gdalconst.GDT_Byte,
                                                   options=['COMPRESS=LZW'])

    tif_out.SetGeoTransform(src_geotrans)
    tif_out.SetProjection(src_proj)

    band = tif_out.GetRasterBand(1)
    band.WriteArray(img_lc_predict)

    tif_out = None

    return output_filename


def add_pyramids_color_in_lc_tif(filename_tif, list_overview=None):
    """
        add pyramids and color table in the tif file
    """

    if list_overview is None:
        list_overview = [2, 4, 8, 16, 32, 64]

    colors = np.array([np.array([255, 255, 255, 255]),    # Ocean
                       np.array([241, 1, 0, 255]),        # Developed
                       np.array([29, 101, 51, 255]),      # Primary wet forest
                       np.array([208, 209, 129, 255]),    # Primary dry forest
                       np.array([108, 169, 102, 255]),    # Secondary forest
                       np.array([174, 114, 41, 255]),     # Shrub/Grass
                       np.array([72, 109, 162, 255]),     # Water
                       np.array([200, 230, 248, 255]),    # Wetland
                       np.array([179, 175, 164, 255]),    # Other
                       ])

    dataset = gdal.Open(filename_tif, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews
    dataset.BuildOverviews(overviewlist=list_overview)

    # Get the first band of the image
    band = dataset.GetRasterBand(1)

    # Create a new color table
    color_table = gdal.ColorTable()

    # Set the color for each value in the color table
    for i in range(0, len(colors)):
        color = tuple(colors[i])
        color_table.SetColorEntry(i, color)

    # Assign the color table to the band
    band.SetRasterColorTable(color_table)

    # Save the changes and close the dataset
    dataset = None

    return None


def main():
# if __name__ == '__main__':

    landcover_version = 'publish_v1'
    predict_flag = 'forecast'
    change_prob_version = f'change_prob_{predict_flag}_v1'
    simulation_folder = 'simulation_v1'

    MAX_LC = 8
    MIN_LC = 1

    landcover_system = {'1': 'Developed',
                        '2': 'PrimaryWetForest',
                        '3': 'PrimaryDryForest',
                        '4': 'SecondaryForest',
                        '5': 'ShrubGrass',
                        '6': 'Water',
                        '7': 'Wetland',
                        '8': 'Other'}

    rootpath_modelling = join(rootpath_project, 'results', 'land_change_modelling', landcover_version)
    path_output = join(rootpath_modelling, predict_flag, simulation_folder)
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    #
    file_prefix_change_prob = join(rootpath_modelling, predict_flag, 'change_prob',
                                   change_prob_version, '{}_{}'.format(predict_flag, change_prob_version))

    logging.basicConfig(filename=join(path_output, '{}_{}_predict.log'.format(landcover_version, simulation_folder)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.info('land cover output version: {}'.format(landcover_version))
    logging.info('prediction flag (forecast/hindcast): {}'.format(predict_flag))
    logging.info('change probability path: {}'.format(file_prefix_change_prob))
    logging.info('modelling folder: {}'.format(simulation_folder))

    img_country_mask, nrows, ncols, src_geotrans, src_proj = get_basic_info()

    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_prediction_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_prediction_year = np.arange(1996, 1491, -1)

    logging.info('predict year is {}'.format(list_prediction_year))

    for i in range(0, len(list_prediction_year)):

        predict_year = list_prediction_year[i]

        img_lc_predict_merge = np.zeros((nrows, ncols), dtype=int)

        # simulate Haiti and DR separately
        for country_flag in ['haiti', 'dr']:

            logging.info(f'predict for {country_flag} in {predict_year}')

            if country_flag == 'haiti':
                country_id = 1
            else:
                country_id = 2

            predict_matrix = np.load(join(rootpath_modelling, predict_flag, 'prediction_matrix',
                                          f'{country_flag}_mk_chain_predict_matrix_modified_normalized.npy'))
            predict_matrix_sum = np.nansum(predict_matrix, axis=0)

            if predict_flag == 'forecast':
                img_lc_current = land_cover_map_read_hispaniola(2022, country_flag=country_flag)
            else:
                img_lc_current = land_cover_map_read_hispaniola(1996, country_flag=country_flag)

            current_count = pixel_count(img_lc_current, MIN_LC=MIN_LC, MAX_LC=MAX_LC)

            predict_matrix_employed = predict_matrix[list_prediction_year == predict_year][0, :, :]
            logging.info('the transition matrix from 2022 to {} is'.format(predict_year))
            logging.info(predict_matrix_employed)

            img_lc_predict = predict_land_cover(img_lc_current,
                                                current_count,
                                                predict_matrix_employed,
                                                predict_matrix_sum,
                                                file_prefix_change_prob,
                                                img_country_mask,
                                                country_id=country_id,
                                                MIN_LC=MIN_LC,
                                                MAX_LC=MAX_LC,
                                                landcover_system=landcover_system)

            img_lc_predict_merge[img_country_mask == country_id] = img_lc_predict[img_country_mask == country_id]

        output_basicname = f'{landcover_version}_{simulation_folder}_{predict_year}'

        output_filename = output_predict_land_cover(img_lc_predict_merge,
                                                    src_geotrans, src_proj,
                                                    path_output, output_basicname)

        add_pyramids_color_in_lc_tif(output_filename, list_overview=None)


if __name__ == '__main__':
    main()