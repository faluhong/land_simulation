"""
    code to use bootstrapping determine the variance of the forecast land cover percentage
"""

import time
import numpy as np
import os
from os.path import join
from scipy.stats import linregress
import pandas as pd
import json
import pickle

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '..'))

from change_matrix.read_change_matrix import read_prob_matrix
from change_matrix_prediction_extrapolation.change_matrix_direct_extrapolation import matrix_normalization_predict, read_obs_pct_file, predict_df_generate


def read_forecast_lc_pct_file(landcover_version, list_observe_year, country_flag, predict_flag='forecast', matrix_folder=None):
    """
        read the forecast land cover percentage file
    """

    df_observe = read_obs_pct_file(output_version_flag=landcover_version,
                                   list_observe_year=list_observe_year,
                                   country_flag=country_flag)

    if matrix_folder is None:
        output_rootpath = join(rootpath_project, 'results', 'land_change_modelling',
                               landcover_version, predict_flag, 'prediction_matrix')
    else:
        output_rootpath = join(rootpath_project, 'results', 'land_change_modelling',
                               landcover_version, predict_flag, 'prediction_matrix', matrix_folder)

    mk_chain_predict_matrix = np.load(join(output_rootpath, '{}_matrix_mk_chain.npy'.format(country_flag)))
    arima_predict_matrix_normalized = np.load(join(output_rootpath, '{}_matrix_arima.npy'.format(country_flag)))
    complete_reg_predict_matrix_normalized = np.load(join(output_rootpath, '{}_matrix_complete_reg.npy'.format(country_flag)))
    partial_reg_predict_matrix_normalized = np.load(join(output_rootpath, '{}_matrix_partial_reg.npy'.format(country_flag)))

    df_predict_mk_chain = pd.read_csv(join(output_rootpath, '{}_pixel_count_mk_chain.csv'.format(country_flag)))
    df_predict_arima = pd.read_csv(join(output_rootpath, '{}_pixel_count_arima.csv'.format(country_flag)))
    df_predict_complete_reg = pd.read_csv(join(output_rootpath, '{}_pixel_count_complete_reg.csv'.format(country_flag)))
    df_predict_partial_reg = pd.read_csv(join(output_rootpath, '{}_pixel_count_partial_reg.csv'.format(country_flag)))

    return (mk_chain_predict_matrix, arima_predict_matrix_normalized, complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized,
            df_predict_mk_chain, df_predict_arima, df_predict_complete_reg, df_predict_partial_reg, df_observe)


def get_linear_regression_matrix(list_predict_year, list_observe_year,
                                 bootstrap_sample_year,
                                 transition_prob_matrix_input,
                                 bootstrapping_years):
    """
        get the predicted transition matrix based on the linear regression
    """

    landcover_types = transition_prob_matrix_input.shape[1]

    # the transition matrix of the years selected in the bootstrap sample
    transition_prob_matrix_selected = np.zeros((len(bootstrapping_years), landcover_types, landcover_types), dtype=float)
    for j in range(0, len(bootstrapping_years)):
        transition_prob_matrix_selected[j] = transition_prob_matrix_input[list_observe_year == bootstrap_sample_year[j]]

    ar_predict_matrix = np.zeros((len(list_predict_year), landcover_types, landcover_types), dtype=float)

    for landcover_id_from in range(1, landcover_types + 1):
        for landcover_id_to in range(1, landcover_types + 1):

            x_predict = np.arange(len(list_observe_year) - 1, len(list_observe_year) + len(list_predict_year) - 1)

            transition_prob_unit = transition_prob_matrix_selected[:, landcover_id_from - 1, landcover_id_to - 1]
            x_fit = bootstrap_sample_year - list_observe_year[0]

            res = linregress(x_fit, transition_prob_unit)
            pred_ar = res.slope * x_predict + res.intercept

            pred_ar = pred_ar - (pred_ar[0] -  transition_prob_matrix_input[-1, landcover_id_from - 1, landcover_id_to - 1])  # make sure the prediction is continuous
            ar_predict_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = pred_ar

    ar_predict_matrix_normalized = np.zeros(np.shape(ar_predict_matrix), dtype=float)
    for i in range(0, len(list_predict_year)):
        ar_predict_matrix_normalized[i] = matrix_normalization_predict(ar_predict_matrix[i])

    return ar_predict_matrix_normalized, transition_prob_matrix_selected


def get_predict_land_cover_percentage_from_bootstrapping_sample(bootstrapping_predict_matrix_normalized,
                                                                df_observe,
                                                                list_observe_year,
                                                                list_predict_year,
                                                                landcover_types=8,
                                                                reference_flag='end'):

    # calculate the land cover percentage
    list_year = df_observe['Year'].values

    count_latest_year = df_observe.iloc[list_year == list_observe_year[-1], 2:landcover_types + 2].values
    count_oldest_year = df_observe.iloc[list_year == list_observe_year[0], 2:landcover_types + 2].values

    if reference_flag == 'end':
        count_initial = count_latest_year
    else:
        count_initial = count_oldest_year

    df_predict_sample = predict_df_generate(count_initial, bootstrapping_predict_matrix_normalized,
                                            list_predict_year, landcover_types=landcover_types)

    return df_predict_sample


# def main():
if __name__ == '__main__':

    predict_flag = 'forecast'

    # the prediction initial based year, if the observation year range is from 1984 to 2022, the reference year can
    # be 1984 ('start') or 2022 ('end')
    reference_flag = 'end'
    landcover_version = 'publish_v1'

    landcover_system = {'1': 'Developed',
                        '2': 'PrimaryWetForest',
                        '3': 'PrimaryDryForest',
                        '4': 'SecondaryForest',
                        '5': 'ShrubGrass',
                        '6': 'Water',
                        '7': 'Wetland',
                        '8': 'Other'}

    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_predict_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_predict_year = np.arange(1996, 1491, -1)


    # the years used for bootstrapping, recent 10 years
    partial_linear_offset = 10
    bootstrapping_years = list_observe_year[-partial_linear_offset::]

    block_size = 1      # the number of years in each block
    n_iterations = 1000    # the number of bootstrapping iterations, change to get more precision confidence intervals

    start_time = time.perf_counter()
    # Perform bootstrapping
    for i_iteration in range(0, n_iterations):

        for country_flag in ['haiti', 'dr']:

            print('Iteration:', i_iteration, 'Country:', country_flag, 'Block size:', block_size)

            rootpath_modelling = join(rootpath_project, 'results', 'land_change_modelling', landcover_version)

            (transition_prob_matrix_adjacent,
             transition_prob_matrix_accumulate_indirect,
             transition_prob_matrix_accumulate_direct,
             transition_prob_matrix_accumulate_direct_inverse) = read_prob_matrix(predict_flag=predict_flag,
                                                                                  country_flag=country_flag)

            # read the forecast land cover percentage and prediction matrix
            (mk_chain_predict_matrix, arima_predict_matrix_normalized,
             complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized,
             df_predict_mk_chain, df_predict_arima,
             df_predict_complete_reg, df_predict_partial_reg, df_observe) = read_forecast_lc_pct_file(landcover_version, list_observe_year, country_flag, predict_flag)

            landcover_types = np.shape(transition_prob_matrix_adjacent)[1]

            # size of bootstrapping sample based on the block size
            n_size = int(len(bootstrapping_years) / block_size)

            # generate a bootstrap sample with replacement based on the block size
            bootstrapping_sample_generate = np.random.choice(bootstrapping_years[0:len(bootstrapping_years) - block_size + 1], size=n_size, replace=True)

            # expand the bootstrapping sample to the same size as the original years based on the block size
            bootstrap_sample_year = np.array([np.arange(bootstrapping_sample_generate[i], bootstrapping_sample_generate[i] + block_size) for i in range(0, len(bootstrapping_sample_generate), 1)])
            bootstrap_sample_year = bootstrap_sample_year.ravel()

            # get the linear regression matrix based on the bootstrapping samples
            if reference_flag == 'end':
                bootstrapping_predict_matrix_normalized, transition_prob_matrix_selected = get_linear_regression_matrix(list_predict_year, list_observe_year,
                                                                                                                        bootstrap_sample_year,
                                                                                                                        transition_prob_matrix_accumulate_direct_inverse,
                                                                                                                        bootstrapping_years)
            else:
                bootstrapping_predict_matrix_normalized, transition_prob_matrix_selected = get_linear_regression_matrix(list_predict_year, list_observe_year,
                                                                                                                        bootstrap_sample_year,
                                                                                                                        transition_prob_matrix_accumulate_direct,
                                                                                                                        bootstrapping_years)

            # get the predicted land cover percentage based on the bootstrapping sample
            df_predict_sample = get_predict_land_cover_percentage_from_bootstrapping_sample(bootstrapping_predict_matrix_normalized,
                                                                                            df_observe,
                                                                                            list_observe_year,
                                                                                            list_predict_year,
                                                                                            landcover_types=landcover_types,
                                                                                            reference_flag=reference_flag)

            dict_bootstrap_sample = {'bootstrap_sample_year': bootstrap_sample_year,
                                     'bootstrap_matrix': bootstrapping_predict_matrix_normalized,
                                     'bootstrap_lc_pct': df_predict_sample}

            # save the bootstrap results
            rootpath_modelling = join(rootpath_project, 'results', 'land_change_modelling', landcover_version)
            output_rootpath = join(rootpath_modelling, predict_flag, f'bootstrap_partial_linear_10_block_{block_size}', country_flag)
            if not os.path.exists(output_rootpath):
                os.makedirs(output_rootpath, exist_ok=True)

            output_filename = join(output_rootpath, f'bootstrap_sample_{i_iteration + 1}.json')

            with open(output_filename, "wb") as file:
                pickle.dump(dict_bootstrap_sample, file)
            file.close()

    end_time = time.perf_counter()
    print(f'running time for {n_iterations} iterations: {end_time - start_time} seconds')