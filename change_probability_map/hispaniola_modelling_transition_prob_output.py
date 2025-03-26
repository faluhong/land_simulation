"""
output the transition prob
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import click
import logging

rootpath_project = r'/scratch/zhz18039/fah20002/LCM_diversity/'

from change_probability_map.utils_model import get_basic_info, land_cover_map_read_hispaniola


def predictor_variables_read(landcover_version, predict_flag, usage_flag, morphology_flag=1, landcover_types=8):

    predictor_variable_dem_path = join(rootpath_project, 'data', 'dem', 'hispaniola_dem_info')
    img_dem = gdal_array.LoadFile(join(predictor_variable_dem_path, 'dem_mosaic.tif'))
    img_slope = gdal_array.LoadFile(join(predictor_variable_dem_path, 'slope_mosaic.tif'))
    
    predictor_variables = np.zeros((landcover_types + 2, np.shape(img_dem)[0], np.shape(img_dem)[1]), dtype=float)
    
    predictor_variable_path = join(rootpath_project, 'results', 'land_change_modelling', landcover_version, 'predictor_variables')
    
    if ((predict_flag == 'forecast') & (usage_flag == 'training')) | ((predict_flag == 'hindcast') & (usage_flag == 'predicting')):
        
        for i_landcover in range(0, landcover_types):
            filename_dist = join(predictor_variable_path, '1996_dist_to_{}_mcircle_{}.tif'.format(i_landcover + 1, morphology_flag))
            predictor_variables[i_landcover, :, :] = gdal_array.LoadFile(filename_dist)
    else:
        
        for i_landcover in range(0, landcover_types):
            filename_dist = join(predictor_variable_path, '2022_dist_to_{}_mcircle_{}.tif'.format(i_landcover + 1, morphology_flag))
            predictor_variables[i_landcover, :, :] = gdal_array.LoadFile(filename_dist)
 
    # predictor_variables = np.array([img_dist_to_1, img_dist_to_2, img_dist_to_3, img_dist_to_4,
    #                                 img_dist_to_5, img_dist_to_6, img_dist_to_7, img_dist_to_8,
    #                                 img_dem, img_slope])

    predictor_variables[landcover_types, :, :] = img_dem
    predictor_variables[landcover_types + 1, :, :] = img_slope
    
    return predictor_variables


def read_density_info(landcover_version, predict_flag, window_radius_size=26, landcover_types=9):
    predictor_variable_path = join(rootpath_project, 'results', 'land_change_modelling', landcover_version,
                                   'predictor_variables')
    
    if predict_flag == 'forecast':
        read_year = 2022
    else:
        read_year = 1996
        
    img_pct_1 = gdal_array.LoadFile(join(predictor_variable_path, '{}_pct_1_{}.tif').format(read_year, window_radius_size))
    pct_return = np.zeros((landcover_types, np.shape(img_pct_1)[0], np.shape(img_pct_1)[1]), dtype=float)
    
    for i_landcover in range(0, landcover_types):
        pct_return[i_landcover, :, :] = gdal_array.LoadFile(join(predictor_variable_path, '{}_pct_{}_{}.tif').format(read_year, i_landcover + 1,window_radius_size))
    
    return pct_return


def change_type_generate(img_lc_former, img_lc_latter, img_landmask, landcover_types=9):
    img_change_type = np.zeros(np.shape(img_landmask), dtype=float)
    
    for land_cover_id_from in range(1, landcover_types + 1):
        for land_cover_id_to in range(1, landcover_types + 1):
            change_mask_from_to = (img_lc_former == land_cover_id_from) & (img_lc_latter == land_cover_id_to)
            img_change_type[change_mask_from_to] = (land_cover_id_from - 1) * landcover_types + land_cover_id_to

    img_change_type[img_change_type == 0] = np.nan
    img_change_type[img_landmask == 0] = np.nan

    return img_change_type


def pixel_number_for_each_lc_transition(landcover_id_from, img_change_type_1984_2022, total_number, 
                                        min_prob=0.03, max_prob=0.4, landcover_types=8):
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

    # array_confusion_matrix_rf = confusion_matrix(y_training_each_land_cover_type, y_predict)
    # logging.info('overall accuracy of {}: {}'.format(classifier_flag, accuracy_score(y_training_each_land_cover_type,
    #                                                                           y_predict)))

    return y_predict, y_predict_prob, classifier


def change_prob_predict(mask_predict, predictor_variables, classifier):

    x_predict_each_land_cover_type = predictor_variables[:, mask_predict].T

    y_predict = classifier.predict(x_predict_each_land_cover_type)
    y_predict_prob = classifier.predict_proba(x_predict_each_land_cover_type)

    return y_predict, y_predict_prob


def get_change_prob_map(array_predict_prob, mask_predict, nrows, ncols):
    img_change_prob = np.zeros((nrows, ncols), dtype=float)
    img_change_prob[mask_predict] = array_predict_prob

    img_change_prob[~mask_predict] = np.nan

    return img_change_prob


def get_change_info(change_type, landcover_types=8, landcover_system=None):

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryForest',
                            '4': 'SecondaryForest',
                            '5': 'ShrubGrass',
                            '6': 'Cropland',
                            '7': 'Water',
                            '8': 'Wetland'}

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
    ouptput the change probability map
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
                                                   gdalconst.GDT_Float32, options=['COMPRESS=LZW'])

    tif_out.SetGeoTransform(src_geotrans)
    tif_out.SetProjection(src_proj)

    band = tif_out.GetRasterBand(1)
    band.WriteArray(img_change_prob)

    tif_out = None


# def main():
if __name__ == '__main__':
    landcover_version = 'degrade_v2_refine_3_3'
    classifier_flag = 'rf'
    predict_flag = 'hindcast'
    # country_flag = 'hispaniola'
    change_prob_version = 'v1'
    # training_sample_flag = 3
    morphology_flag = 1
    window_radius_size = 26
    
    total_number = 20000
    MIN_LC = 1
    MAX_LC = 9
    landcover_types = 9
    
    landcover_system = {'1': 'Developed',
                        '2': 'Barren',
                        '3': 'PrimaryWetForest',
                        '4': 'PrimaryDryForest',
                        '5': 'SecondaryForest',
                        '6': 'ShrubGrass',
                        '7': 'Cropland',
                        '8': 'Water',
                        '9': 'Wetland'}

    np.set_printoptions(precision=4, suppress=True)

    # You may need to change the path_output to your own path
    path_output = join(rootpath_project, 'results', 'land_change_modelling', landcover_version,
                        predict_flag, 'change_prob', change_prob_version)

    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)
    print(path_output)

    logging.basicConfig(filename=join(path_output, '{}.log'.format(landcover_version)),
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

    training_variables = predictor_variables_read(landcover_version, predict_flag, 'training',  morphology_flag, landcover_types=landcover_types)
    predicting_variables = predictor_variables_read(landcover_version, predict_flag, 'predicting', morphology_flag, landcover_types=landcover_types)

    pct_weight = read_density_info(landcover_version, predict_flag, window_radius_size, landcover_types=landcover_types)
    
    if predict_flag =='forecast':
        img_lc_begin = land_cover_map_read_hispaniola(1996, landcover_version)
        img_lc_end = land_cover_map_read_hispaniola(2022, landcover_version)
    else:
        img_lc_begin = land_cover_map_read_hispaniola(2022, landcover_version)
        img_lc_end = land_cover_map_read_hispaniola(1996, landcover_version)
    img_lc_begin[img_landmask == 0] = np.nan
    img_lc_end[img_landmask == 0] = np.nan
    
    img_change_type_begin_to_end = change_type_generate(img_lc_begin, img_lc_end, img_landmask, landcover_types=landcover_types)
    # img_change_type_begin_to_end[img_change_type_begin_to_end == 0] = np.nan

    #%
    for landcover_id_from in range(MIN_LC, MAX_LC + 1):
    # for landcover_id_from in range(5, 5 + 1):
        logging.info('land cover id from: {} {}'.format(landcover_id_from, landcover_system[str(landcover_id_from)]))

        pct_weight_lc = pct_weight[landcover_id_from - 1, :, :].copy()
        
        mask_predict = img_lc_end == landcover_id_from
        
        training_variables_selected = np.vstack([training_variables[0 : landcover_id_from - 1],
                                                  training_variables[landcover_id_from::]])
        predicting_variables_selected = np.vstack([predicting_variables[0 : landcover_id_from - 1],
                                                  predicting_variables[landcover_id_from::]])
        
        print('selected training variable shape', np.shape(training_variables_selected))

        list_change_type = np.zeros(landcover_types)
        for landcover_id_to in range(MIN_LC, MAX_LC + 1):
            list_change_type[landcover_id_to - 1] = (landcover_id_from - 1) * landcover_types + landcover_id_to
    
        array_select_num = pixel_number_for_each_lc_transition(landcover_id_from, img_change_type_begin_to_end, total_number,
                                                               landcover_types=landcover_types)
        logging.info('selected training sample number {}'.format(array_select_num))
        
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
                landcover_id_from, landcover_id_to, change_info = get_change_info(change_id, landcover_types=landcover_types, landcover_system=landcover_system)
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

            #%
            for i in range(0, len(list_change_type)):
            # for i in range(2, 3):
                change_id = list_change_type[i]
                landcover_id_from, landcover_id_to, change_info = get_change_info(change_id, landcover_types=landcover_types, landcover_system=landcover_system)
                if change_id in np.unique(y_predict):
                    mask_id = change_id == np.unique(y_training_each_land_cover_type)
                    
                    img_change_prob = get_change_prob_map(y_predict_prob[:, mask_id][:, 0], mask_predict, nrows, ncols)
                    
                    if landcover_id_from == landcover_id_to:
                        img_change_prob = img_change_prob * pct_weight_lc
                    else:
                        img_change_prob = img_change_prob * (1 - pct_weight_lc) 
                    
                    # FP(img_change_prob, title=change_info)
                    
                else:
                    img_change_prob = get_change_prob_map(0, mask_predict, nrows, ncols)
                
                output_basicname = '{}_{}_{}.tif'.format(predict_flag, change_prob_version, change_info)
                logging.info('output {}'.format(output_basicname))
                
                output_change_prob(img_change_prob, path_output, output_basicname, src_geotrans, src_proj)


# if __name__ == '__main__':
#     main()


# %%
