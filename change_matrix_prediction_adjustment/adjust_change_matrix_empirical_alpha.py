"""
customize predict matrix plan 3:
On the basis of iterative hindcast predict matrix
increase other land cover type -> primary forest by multiplying the exponential function
"""

import numpy as np
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress, theilslopes
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from osgeo import gdal, gdal_array, gdalconst
import pandas as pd
import matplotlib.ticker as plticker
import seaborn as sns
import random
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')

sns.set_theme()

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))


# from land_change_modelling.hispaniola_create_model_matrix import predict_df_generate, matrix_normalization_predict, get_initial_count
# from land_change_modelling.his_model_matrix_iterative import sns_hist_plot, probability_distribution_function_plot, \
#     get_accumulate_indirect_prediction_matrix, plot_predict_pct_with_confident_interval

from change_matrix_prediction_extrapolation.change_matrix_direct_extrapolation import get_initial_count, predict_df_generate
from change_matrix_prediction_extrapolation.change_matrix_iterative_extrapolation import (plot_predict_pct_with_confident_interval,
                                                                                          get_accumulate_indirect_prediction_matrix)
from change_matrix_prediction_extrapolation.plot_utils import plot_predict_pct, plot_change_matrix, ensemble_transition_prob_plot
from change_matrix.read_change_matrix import read_prob_matrix



def matrix_normalization_predict(input_matrix):
    """
    normalize the row of the matrix
    (1) the negative values were assigned to zero
    (2) after step(1), each row was normalized to make each row sums up to 1
    """

    output_matrix = np.zeros((input_matrix.shape), dtype=float)
    for i_row in range(0, np.shape(input_matrix)[0]):
        if (input_matrix[i_row, :] == 0).all():
            # if one land cover type does not exist in the former date, 1 will be assigned to the diagonal value
            # e.g., develop does not exist in 1984, the change_matrix for develop (type 1) will be [0, 0, 0, 0, 0, 0, 0]
            # after the normalization, the transition matrix will be [1, 0, 0, 0, 0, 0, 0]
            output_matrix[i_row, i_row] = 1
        elif (input_matrix[i_row, :] < 0).any():
            tmp = input_matrix[i_row, :].copy()
            tmp[tmp < 0] = 0
            output_matrix[i_row, :] = tmp / np.sum(tmp)
        else:
            tmp = input_matrix[i_row, :].copy()
            output_matrix[i_row, :] = tmp / np.sum(tmp)

    return output_matrix


def generate_adjust_iterative_predict_adjacent_matrix_alpha_based(alpha, transition_prob_matrix_adjacent, list_predict_year, landcover_types):
    predict_adjacent_matrix = np.zeros((len(list_predict_year) - 1, landcover_types, landcover_types), dtype=float)

    for landcover_id_from in range(1, landcover_types + 1):
        for landcover_id_to in range(1, landcover_types + 1):

            transition_adjacent_unit = transition_prob_matrix_adjacent[:, landcover_id_from - 1, landcover_id_to - 1]

            if np.nanstd(transition_adjacent_unit) == 0:
                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = transition_adjacent_unit[0]
            else:
                x_predict = np.linspace(transition_adjacent_unit.min(), transition_adjacent_unit.max(), 100)

                kde_sp = gaussian_kde(transition_adjacent_unit, bw_method='scott')  ### I MEAN HERE! ###
                fit_sp = kde_sp.pdf(transition_adjacent_unit)
                plot_sp = kde_sp.pdf(x_predict)

                predict_prob = np.random.choice(x_predict, size=len(list_predict_year) - 1, p= plot_sp / np.sum(plot_sp))

                # adjust the change rate of primary forest -> other land cover types to match the anchor points
                if (landcover_id_to == 3) | (landcover_id_to == 4):

                    # predict_prob = predict_prob + x_predict[np.argmax(plot_sp)] * 50    # add approach
                    # predict_prob = predict_prob * 20  # multiple approach

                    for t in range(0, len(predict_prob)):
                        predict_prob[t] = predict_prob[t] * np.exp(alpha * t)

                        # exp_part = np.exp(-alpha * (t - 45))
                        # print(exp_part / (1 + exp_part) / (1 + exp_part))
                        # predict_prob[t] = predict_prob[t] * alpha * exp_part / (1 + exp_part) / (1 + exp_part)

                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = predict_prob

    predict_adjacent_matrix[:, 1::, 0] = 0  # set other land cover types -> developed to zero

    # normalize the transition matrix to make sure each row sum up to 1
    for i in range(0, np.shape(predict_adjacent_matrix)[0]):
        predict_adjacent_matrix[i, :, :] = matrix_normalization_predict(predict_adjacent_matrix[i, :, :])

    return predict_adjacent_matrix


def sum_alpha_base_prediction_matrix_generate(iterative_rounds, alpha, transition_prob_matrix_adjacent,
                                              count_initial, list_predict_year, landcover_types):

    # iterative_rounds = 100
    # alpha = 0.015

    array_predict_pct = np.zeros((iterative_rounds, len(list_predict_year), 2 * landcover_types + 1), dtype=float)

    for i_round in range(0, iterative_rounds):
        # print(i_round)
        predict_adjacent_matrix = generate_adjust_iterative_predict_adjacent_matrix_alpha_based(alpha, transition_prob_matrix_adjacent, list_predict_year, landcover_types)
        predict_accumulate_matrix = get_accumulate_indirect_prediction_matrix(predict_adjacent_matrix, list_predict_year, landcover_types)
        df_predict = predict_df_generate(count_initial, predict_accumulate_matrix, list_predict_year, landcover_types=landcover_types)

        array_predict_pct[i_round, :, :] = df_predict.values

    array_average = np.nanmean(array_predict_pct, axis=0)
    df_average = df_predict.copy()
    df_average.iloc[:, :] = array_average

    df_95th = df_average.copy()
    df_5th = df_average.copy()

    df_95th.iloc[:, :] = np.nanpercentile(array_predict_pct, 95, axis=0)
    df_5th.iloc[:, :] = np.nanpercentile(array_predict_pct, 5, axis=0)

    return predict_adjacent_matrix, predict_accumulate_matrix, df_predict, df_average, df_95th, df_5th


# def main():
if __name__ == '__main__':

    predict_flag = 'hindcast'
    reference_flag = 'end'
    output_version_flag = 'degrade_v2_refine_3_3'
    country_flag = 'hispaniola'

    landcover_types = 9

    np.set_printoptions(precision=4, suppress=True)
    ##
    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_predict_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_predict_year = np.arange(1996, 1491, -1)

    # print('observe years:', list_observe_year)
    # print('prediction years', list_prediction_year)

    transition_prob_matrix_adjacent, transition_prob_matrix_accumulate_indirect, \
        transition_prob_matrix_accumulate_direct, transition_prob_matrix_accumulate_direct_inverse \
        = read_prob_matrix(predict_flag, country_flag=country_flag)

    ##
    df_observe, count_initial = get_initial_count(output_version_flag, list_observe_year, country_flag, predict_flag, reference_flag, landcover_types)

    iterative_rounds = 10   # iterative rounds
    alpha = 0.005           # parameter alpha, can be adjusted to control the change rate

    predict_adjacent_matrix, predict_accumulate_matrix, df_predict, df_average, df_95th, df_5th \
        = sum_alpha_base_prediction_matrix_generate(iterative_rounds, alpha, transition_prob_matrix_adjacent,
                                                   count_initial, list_predict_year, landcover_types)

    plot_predict_pct_with_confident_interval(list_observe_year,
                                             list_predict_year,
                                             df_observe,
                                             df_average,
                                             df_5th,
                                             df_95th,
                                             title='alpha {}'.format(alpha),
                                             output_flag=0,
                                             output_folder=None,
                                             axis_reverse_flag=False,
                                             x_axis_interval=50.0)

    ##
    # output the prediction change matrix
    output_rootpath = join(rootpath, 'results', 'land_change_modelling', output_version_flag, predict_flag,
                           'prediction_matrix', 'empirical_alpha_{:04d}'.format(int(alpha*1000)))
    print(output_rootpath)
    if not os.path.exists(output_rootpath):
        os.makedirs(output_rootpath, exist_ok=True)

    np.save(join(output_rootpath, 'alpha_{:04d}_adjacent_matrix.npy'.format(int(alpha*1000))), predict_adjacent_matrix)
    np.save(join(output_rootpath, 'alpha_{:04d}_accumulate_matrix.npy'.format(int(alpha*1000))), predict_accumulate_matrix)

    df_predict.to_csv(join(output_rootpath,'alpha_{:04d}_df_predict.csv'.format(int(alpha*1000))))

    df_average.to_csv(join(output_rootpath, 'alpha_{:04d}_df_average.csv'.format(int(alpha*1000))))
    df_95th.to_csv(join(output_rootpath, 'alpha_{:04d}_df_95th.csv'.format(int(alpha*1000))))
    df_5th.to_csv(join(output_rootpath, 'alpha_{:04d}_df_5th.csv'.format(int(alpha*1000))))

