"""
    generate the change probability map for the multi-land cover data
"""

import numpy as np
from os.path import join
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import logging

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from change_matrix.util_read_hispaniola_lc import land_cover_map_read_hispaniola, get_basic_info


def predictor_variables_read(landcover_version, predict_flag, usage_flag, read_year, morphology_flag=1, landcover_types=8,):
    """
        read the predictor variables
    """

    predictor_variable_dem_path = join(rootpath_project, 'data', 'dem', 'hispaniola_dem_info')

    img_dem = gdal_array.LoadFile(join(predictor_variable_dem_path, 'dem_mosaic.tif'))
    img_slope = gdal_array.LoadFile(join(predictor_variable_dem_path, 'slope_mosaic.tif'))

    predictor_variables = np.zeros((landcover_types + 2, np.shape(img_dem)[0], np.shape(img_dem)[1]), dtype=float)

    predictor_variable_path = join(rootpath_project, 'results', 'land_change_modelling', landcover_version, 'predictor_variables')

    if ((predict_flag == 'forecast') & (usage_flag == 'training')) | ((predict_flag == 'hindcast') & (usage_flag == 'predicting')):

        for i_landcover in range(0, landcover_types):
            filename_dist = join(predictor_variable_path, f'{read_year}_dist_to_{i_landcover + 1}_mcircle_{morphology_flag}.tif')
            predictor_variables[i_landcover, :, :] = gdal_array.LoadFile(filename_dist)
    else:

        for i_landcover in range(0, landcover_types):
            filename_dist = join(predictor_variable_path, f'{read_year}_dist_to_{i_landcover + 1}_mcircle_{morphology_flag}.tif')
            predictor_variables[i_landcover, :, :] = gdal_array.LoadFile(filename_dist)

    predictor_variables[landcover_types, :, :] = img_dem
    predictor_variables[landcover_types + 1, :, :] = img_slope

    return predictor_variables


def read_density_info(landcover_version, landcover_id, window_radius_size=26, read_year=2022):
    """
        read the density information for the land cover type

        :param landcover_version:
        :param landcover_id:
        :param window_radius_size:
        :param read_year:
        :return:
    """

    predictor_variable_path = join(rootpath_project, 'results', 'land_change_modelling', landcover_version, 'predictor_variables')

    filename_pct = join(predictor_variable_path, '{}_pct_{}_{}.tif').format(read_year, landcover_id, window_radius_size)
    pct_return = gdal_array.LoadFile(filename_pct)

    return pct_return


def generate_change_type_img(img_lc_former, img_lc_latter, img_landmask, landcover_types=9):
    """

    :param img_lc_former:
    :param img_lc_latter:
    :param img_landmask:
    :param landcover_types:
    :return:
    """

    img_change_type = np.zeros(np.shape(img_landmask), dtype=float)

    for land_cover_id_from in range(1, landcover_types + 1):
        for land_cover_id_to in range(1, landcover_types + 1):
            change_mask_from_to = (img_lc_former == land_cover_id_from) & (img_lc_latter == land_cover_id_to)
            img_change_type[change_mask_from_to] = (land_cover_id_from - 1) * landcover_types + land_cover_id_to

    img_change_type[img_change_type == 0] = np.nan
    img_change_type[img_landmask == 0] = np.nan

    return img_change_type


def pixel_number_for_each_lc_transition(landcover_id_from, img_change_type_1984_2022, total_number, min_prob=0.03, max_prob=0.4, landcover_types=8):
    """
        the selected pixel number for each land cover transition
        the maximum proportion is 40% and the minimum proportion is 3%

        first determine the minimum and maximum part, then the rest part is determined proportionally
    """

    array_change_pixel_count = np.zeros(landcover_types, dtype=float)

    for landcover_id_to in range(1, landcover_types + 1):
        change_id = (landcover_id_from - 1) * landcover_types + landcover_id_to
        array_change_pixel_count[landcover_id_to - 1] = np.count_nonzero(img_change_type_1984_2022 == change_id)

    array_change_pct = array_change_pixel_count / np.sum(array_change_pixel_count)
    array_select_num = np.zeros(len(array_change_pct), dtype=int)

    if np.count_nonzero(array_change_pct > max_prob) > 0:
        array_select_num[array_change_pct > max_prob] = int(total_number * max_prob)
    if np.count_nonzero((array_change_pct < min_prob) & (array_change_pct > 0)) > 0:
        array_select_num[(array_change_pct < min_prob) & (array_change_pct > 0)] = int(total_number * min_prob)

    mask_proportional = ~((array_change_pct > max_prob) | (array_change_pct < min_prob))
    array_proportional = array_change_pct[mask_proportional] / np.nansum(array_change_pct[mask_proportional])

    array_select_num[mask_proportional] = (total_number - array_select_num.sum()) * array_proportional
    array_select_num[array_select_num > int(total_number * max_prob)] = int(total_number * max_prob)

    return array_select_num


def training_sample_generate(landcover_id_from,
                             array_select_num,
                             predictor_variables_selected,
                             img_change_type_begin_to_end,
                             landcover_types=8):
    x_training_each_land_cover_type = []
    y_training_each_land_cover_type = []

    for landcover_id_to in range(1, landcover_types + 1):
        # change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
        #                                            landcover_id_to, landcover_system[str(landcover_id_to)])

        if array_select_num[landcover_id_to - 1] == 0:
            # print('no change type exists for {}'.format(change_info))
            pass
        else:
            change_id = (landcover_id_from - 1) * landcover_types + landcover_id_to
            # print(change_info, array_select_num[landcover_id_to - 1], 'available pixel {}'.format(np.count_nonzero(mask_change_id)))

            mask_change_id = img_change_type_begin_to_end == change_id

            random_id = np.random.permutation(np.count_nonzero(mask_change_id))[0: array_select_num[landcover_id_to - 1]]

            row_id_select = np.where(mask_change_id)[0][random_id]
            col_id_select = np.where(mask_change_id)[1][random_id]

            x_training_each_change_type = predictor_variables_selected[:, row_id_select, col_id_select].T

            y_training_each_change_type = img_change_type_begin_to_end[row_id_select, col_id_select]

            x_training_each_land_cover_type.append(x_training_each_change_type)
            y_training_each_land_cover_type.append(y_training_each_change_type)

    x_training_each_land_cover_type = np.concatenate(x_training_each_land_cover_type)
    y_training_each_land_cover_type = np.concatenate(y_training_each_land_cover_type)

    return x_training_each_land_cover_type, y_training_each_land_cover_type


def classifier_generate(x_training_each_land_cover_type, y_training_each_land_cover_type, classifier_flag,
                        solver='lbfgs'):
    """
    generate the classifier, random forest or support vector machine classifier
    random forest: training time is around 2 seconds, overall accuracy close to 1.0
    svm:  training time is around 80 seconds, overall accuracy is around 0.5 to 0.6
    logistic regression: training time is neglectable, overall accuracy is around 0.5 to 0.6
    decision tree: training time is neglectable, overall accuracy close to 1.0
    """

    if classifier_flag == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        classifier.fit(x_training_each_land_cover_type, y_training_each_land_cover_type)
    elif classifier_flag == 'svc':
        classifier = SVC(probability=True)
        classifier.fit(x_training_each_land_cover_type, y_training_each_land_cover_type)
    elif classifier_flag == 'logistic_reg':
        classifier = LogisticRegression(solver=solver)
        classifier.fit(x_training_each_land_cover_type, y_training_each_land_cover_type)
    elif classifier_flag == 'decision_tree':
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x_training_each_land_cover_type, y_training_each_land_cover_type)
    else:
        classifier = None

    assert classifier is not None

    y_predict = classifier.predict(x_training_each_land_cover_type)
    y_predict_prob = classifier.predict_proba(x_training_each_land_cover_type)

    return y_predict, y_predict_prob, classifier


def change_prob_predict(mask_predict, predictor_variables, classifier):
    """
        predict the change probability for each pixel using the trained classifier

        :param mask_predict:
        :param predictor_variables:
        :param classifier:
        :return:
    """

    x_predict_each_land_cover_type = predictor_variables[:, mask_predict].T

    y_predict = classifier.predict(x_predict_each_land_cover_type)
    y_predict_prob = classifier.predict_proba(x_predict_each_land_cover_type)

    return y_predict, y_predict_prob


def get_change_prob_map(array_predict_prob, mask_predict, nrows, ncols):
    """
        get the change probability map from the predicted probability

        Args:
            array_predict_prob: the predicted probability
            mask_predict: the mask for the prediction
            nrows: number of rows
            ncols: number of columns
    """

    img_change_prob = np.zeros((nrows, ncols), dtype=float)
    img_change_prob[mask_predict] = array_predict_prob

    img_change_prob[~mask_predict] = np.nan

    return img_change_prob


def get_change_info(change_type, landcover_types=8, landcover_system=None):
    """
        get the change information from the change type, e.g., from_1_Developed_to_2_PrimaryWetForest

        :param change_type:
        :param landcover_types:
        :param landcover_system:
        :return:
    """

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'PrimaryWetForest',
                            '3': 'PrimaryDryForest',
                            '4': 'SecondaryForest',
                            '5': 'ShrubGrass',
                            '6': 'Water',
                            '7': 'Wetland',
                            '8': 'Other'}

    if change_type % landcover_types == 0:
        landcover_id_to = landcover_types
        landcover_id_from = int(change_type // landcover_types)
    else:
        landcover_id_from = int(change_type // landcover_types + 1)
        landcover_id_to = int(change_type - (landcover_id_from - 1) * landcover_types)

    change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                               landcover_id_to, landcover_system[str(landcover_id_to)])

    return landcover_id_from, landcover_id_to, change_info


def output_change_prob(img_change_prob, path_output, output_basicname, src_geotrans, src_proj):
    """
    output the change probability map
    Args:
        img_change_prob:
        path_output:
        output_basicname:
        src_geotrans:
        src_proj:
    """

    output_filename = join(path_output, output_basicname)

    abspath = os.path.abspath(join(output_filename, os.pardir))
    if not os.path.exists(abspath):
        os.makedirs(abspath, exist_ok=True)

    tif_out = gdal.GetDriverByName('GTiff').Create(output_filename, np.shape(img_change_prob)[1],
                                                   np.shape(img_change_prob)[0], 1,
                                                   gdalconst.GDT_Float32,
                                                   options=['COMPRESS=LZW'])

    tif_out.SetGeoTransform(src_geotrans)
    tif_out.SetProjection(src_proj)

    band = tif_out.GetRasterBand(1)
    band.WriteArray(img_change_prob)

    tif_out = None


# def main():
if __name__ == '__main__':

    landcover_version = 'publish_v1'
    classifier_flag = 'rf'
    predict_flag = 'forecast'
    change_prob_version = f'change_prob_{predict_flag}_v1'

    np.set_printoptions(precision=4, suppress=True)

    morphology_flag = 1
    window_radius_size = 26

    begin_year = 1996
    end_year = 2022

    total_number = 20000  # total number of training samples
    MIN_LC = 1  # minimum land cover id
    MAX_LC = 8  # maximum land cover id
    landcover_types = 8  # number of land cover types

    landcover_system = {'1': 'Developed',
                        '2': 'PrimaryWetForest',
                        '3': 'PrimaryDryForest',
                        '4': 'SecondaryForest',
                        '5': 'ShrubGrass',
                        '6': 'Water',
                        '7': 'Wetland',
                        '8': 'Other'}

    # Path to output the change probability map
    path_output = join(rootpath_project, 'results', 'land_change_modelling',
                       landcover_version, predict_flag, 'change_prob', change_prob_version)

    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)
    print(path_output)

    logging.basicConfig(filename=join(path_output, '{}_{}.log'.format(landcover_version, change_prob_version)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.info('output the transition prob')
    logging.info('land cover output version: {}'.format(landcover_version))
    logging.info('prediction flag (forecast/hindcast): {}'.format(predict_flag))
    logging.info('classifier flag: {}'.format(classifier_flag))
    logging.info('change prob version: {}'.format(change_prob_version))
    logging.info('window radius size: {}'.format(window_radius_size))
    logging.info('land cover types: {}'.format(landcover_types))

    img_landmask, nrows, ncols, src_geotrans, src_proj = get_basic_info()

    training_variables = predictor_variables_read(landcover_version, predict_flag, 'training', end_year, morphology_flag, landcover_types=landcover_types)
    predicting_variables = predictor_variables_read(landcover_version, predict_flag, 'predicting', end_year, morphology_flag, landcover_types=landcover_types)

    img_lc_begin = land_cover_map_read_hispaniola(year=begin_year, country_flag='hispaniola')
    img_lc_end = land_cover_map_read_hispaniola(year=end_year, country_flag='hispaniola')

    img_lc_begin[img_landmask == 0] = np.nan
    img_lc_end[img_landmask == 0] = np.nan

    img_change_type_begin_to_end = generate_change_type_img(img_lc_begin, img_lc_end, img_landmask, landcover_types=landcover_types)

    for landcover_id_from in range(MIN_LC, MAX_LC + 1):
        # for landcover_id_from in range(5, 5 + 1):
        logging.info('land cover id from: {} {}'.format(landcover_id_from, landcover_system[str(landcover_id_from)]))

        # read the density information for the land cover type
        pct_weight_lc = read_density_info(landcover_version,
                                          landcover_id=landcover_id_from,
                                          window_radius_size=window_radius_size,
                                          read_year=end_year)

        mask_predict = img_lc_end == landcover_id_from

        training_variables_selected = np.vstack([training_variables[0: landcover_id_from - 1],
                                                 training_variables[landcover_id_from::]])
        predicting_variables_selected = np.vstack([predicting_variables[0: landcover_id_from - 1],
                                                   predicting_variables[landcover_id_from::]])

        print('selected training variable shape', np.shape(training_variables_selected))

        # get the change type list for the land cover type
        list_change_type = [(landcover_id_from - 1) * landcover_types + landcover_id_to
                            for landcover_id_to in range(MIN_LC, MAX_LC + 1)]

        # number of pixels for each land cover transition type, the maximum proportion is 40% and the minimum proportion is 3%
        array_select_num = pixel_number_for_each_lc_transition(landcover_id_from, img_change_type_begin_to_end, total_number,
                                                               landcover_types=landcover_types)
        logging.info('selected training sample number {}'.format(array_select_num))

        # get the training sample from observed land cover changes
        x_training_each_land_cover_type, y_training_each_land_cover_type = training_sample_generate(landcover_id_from,
                                                                                                    array_select_num,
                                                                                                    training_variables_selected,
                                                                                                    img_change_type_begin_to_end,
                                                                                                    landcover_types=landcover_types)

        logging.info(x_training_each_land_cover_type.shape)
        logging.info(y_training_each_land_cover_type.shape)

        if len(np.unique(y_training_each_land_cover_type)) == 1:
            logging.info('only one change type happen for {}'.format(landcover_id_from))

            y_predict = np.unique(y_training_each_land_cover_type)
            y_predict_prob = np.zeros((1, landcover_types))
            y_predict_prob[0, y_predict == list_change_type] = 1

            for i in range(0, len(list_change_type)):
                change_id = list_change_type[i]
                landcover_id_from, landcover_id_to, change_info = get_change_info(change_id,
                                                                                  landcover_types=landcover_types,
                                                                                  landcover_system=landcover_system)
                if change_id in np.unique(y_predict):
                    img_change_prob = get_change_prob_map(1, mask_predict, nrows, ncols)
                else:
                    img_change_prob = get_change_prob_map(0, mask_predict, nrows, ncols)

                output_basicname = '{}_{}_{}.tif'.format(predict_flag, change_prob_version, change_info)
                logging.info('output {}'.format(output_basicname))

                output_change_prob(img_change_prob, path_output, output_basicname, src_geotrans, src_proj)

        else:
            y_training_predict, y_training_predict_prob, classifier = classifier_generate(x_training_each_land_cover_type,
                                                                                          y_training_each_land_cover_type,
                                                                                          classifier_flag=classifier_flag,
                                                                                          solver='newton-cg')
            logging.info('features importance {}'.format(classifier.feature_importances_))

            y_predict, y_predict_prob = change_prob_predict(mask_predict, predicting_variables_selected, classifier)

            for i in range(0, len(list_change_type)):

                change_id = list_change_type[i]
                landcover_id_from, landcover_id_to, change_info = get_change_info(change_id,
                                                                                  landcover_types=landcover_types,
                                                                                  landcover_system=landcover_system)

                if change_id in np.unique(y_predict):
                    mask_id = change_id == np.unique(y_training_each_land_cover_type)

                    # get the random forest predicted change probability map
                    img_change_prob_rf = get_change_prob_map(y_predict_prob[:, mask_id][:, 0], mask_predict, nrows, ncols)

                    # combine the random forest predicted change probability map and the land cover density information to get the final change probability map
                    if landcover_id_from == landcover_id_to:
                        img_change_prob = img_change_prob_rf * pct_weight_lc
                    else:
                        img_change_prob = img_change_prob_rf * (1 - pct_weight_lc)

                    output_basicname = '{}_{}_{}.tif'.format(predict_flag, change_prob_version, change_info)
                    output_change_prob(img_change_prob, path_output, output_basicname, src_geotrans, src_proj)

                    # also output the random forest predicted change probability map for comparison
                    output_basicname = '{}_{}_{}_rf.tif'.format(predict_flag, change_prob_version, change_info)
                    output_change_prob(img_change_prob_rf, path_output, output_basicname, src_geotrans, src_proj)

                else:
                    img_change_prob = get_change_prob_map(0, mask_predict, nrows, ncols)

                    output_basicname = '{}_{}_{}.tif'.format(predict_flag, change_prob_version, change_info)
                    logging.info('output {}'.format(output_basicname))

                    output_change_prob(img_change_prob, path_output, output_basicname, src_geotrans, src_proj)


# if __name__ == '__main__':
#     main()