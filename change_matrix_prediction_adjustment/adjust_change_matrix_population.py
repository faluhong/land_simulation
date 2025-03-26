"""
    customize predict matrix plan 4:
    On the basis of iterative hindcast predict matrix
    adjust other land cover type -> primary forest with the population information

    Prob_adjust = Prob_stochastic *(1 + pop_count * pop_change_rate * t * alpha)
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
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

# from land_change_modelling.plot_utils import line_plot_each_predictor, plot_predict_pct, plot_change_matrix, transition_prob_plot, ensemble_transition_prob_plot
# from land_change_modelling.hispaniola_create_model_matrix import read_prob_matrix, \
#     auto_regression_predict_matrix_generate, read_obs_pct_file, predict_df_generate, plot_predict_pct, matrix_normalization_predict, get_initial_count
# from land_change_modelling.change_matrix_hispaniola import land_cover_map_read_hispaniola
# from Basic_tools.Figure_plot import FP, HistPlot
# from land_change_modelling.his_model_matrix_iterative import sns_hist_plot, probability_distribution_function_plot, \
#     get_accumulate_indirect_prediction_matrix, plot_predict_pct_with_confident_interval


from change_matrix.read_change_matrix import read_prob_matrix
from change_matrix_prediction_extrapolation.change_matrix_iterative_extrapolation  import get_accumulate_indirect_prediction_matrix, plot_predict_pct_with_confident_interval
from change_matrix_prediction_extrapolation.change_matrix_direct_extrapolation import predict_df_generate, matrix_normalization_predict, get_initial_count


def read_population_info(list_predict_year):
    """
        read the population information
    """
    df_population = pd.read_excel(join(rootpath, 'data', 'HYDE', 'hispaniola_population_info.xlsx'))
    df_population = df_population[df_population['year'].isin(list_predict_year)]
    if list_predict_year[0] > list_predict_year[-1]:   # means hindcasting
        df_population = df_population[::-1]

    array_population = df_population['population_count'].values / 1000000
    array_population_change_rate = df_population['population_change_rate'].values * 100

    return array_population, array_population_change_rate


def generate_adjust_iterative_predict_adjacent_matrix_population_based(alpha, transition_prob_matrix_adjacent, list_predict_year, landcover_types):
    """
        generate the adjusted iterative predict adjacent matrix with the adjustment of population information
    """

    predict_adjacent_matrix = np.zeros((len(list_predict_year) - 1, landcover_types, landcover_types), dtype=float)

    for landcover_id_from in range(1, landcover_types + 1):
        for landcover_id_to in range(1, landcover_types + 1):

            transition_adjacent_unit = transition_prob_matrix_adjacent[:, landcover_id_from - 1, landcover_id_to - 1]
            # sns_hist_plot(transition_adjacent_unit, title='Fitted point distribution')

            if np.nanstd(transition_adjacent_unit) == 0:
                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = transition_adjacent_unit[0]
            else:
                x_predict = np.linspace(transition_adjacent_unit.min(), transition_adjacent_unit.max(), 100)

                kde_sp = gaussian_kde(transition_adjacent_unit, bw_method='scott')
                fit_sp = kde_sp.pdf(transition_adjacent_unit)
                plot_sp = kde_sp.pdf(x_predict)

                # if (landcover_id_from == 4) & (landcover_id_to == 3):
                #     probability_distribution_function_plot(transition_adjacent_unit, x_predict, fit_sp, plot_sp,
                #                                            title='{} Fitted PDF'.format(change_info),
                #                                            x_label='{} change prob'.format(change_info), y_label='probability')

                # get the prediction probability
                predict_prob = np.random.choice(x_predict, size=len(list_predict_year) - 1, p=plot_sp / np.sum(plot_sp))
                # sns_hist_plot(predict_prob, title='Generated point distribution')

                if (landcover_id_to == 3) | (landcover_id_to == 4):
                    array_population, array_population_change_rate = read_population_info(list_predict_year)

                    for t in range(0, len(predict_prob)):
                        # predict_prob[t] = predict_prob[t] * np.exp(alpha * t)
                        # predict_prob[t] = predict_prob[t] * (1 + alpha * t)
                        # predict_prob[t] = predict_prob[t] * (1 + alpha * t)

                        # adjust the probability of other land cover types -> primary forest with the population information
                        # assumption: (1) More population brings more primary forest loss
                        #             (2) Higher population growth rate brings more primary forest loss
                        #             (3) The available PF resource limits the PF loss. Abundant PF resources can cause more PF loss
                        predict_prob[t] = predict_prob[t] * (1 + alpha * t * array_population[t] * array_population_change_rate[t])
                        # predict_prob[t] = predict_prob[t] * (1 + array_population[t] * array_population_change_rate[t] * alpha / (1 + np.exp(-t)))

                        # exp_part = np.exp(-alpha * (t - 45))
                        # print(exp_part / (1 + exp_part) / (1 + exp_part))
                        # predict_prob[t] = predict_prob[t] * alpha * exp_part / (1 + exp_part) / (1 + exp_part)

                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = predict_prob

    predict_adjacent_matrix[:, 1::, 0] = 0  # set other land cover types -> developed to zero

    # normalize the transition matrix to make sure each row sum up to 1
    for i in range(0, np.shape(predict_adjacent_matrix)[0]):
        predict_adjacent_matrix[i, :, :] = matrix_normalization_predict(predict_adjacent_matrix[i, :, :])

    return predict_adjacent_matrix




def sum_alpha_base_prediction_matrix_generate_population_based(iterative_rounds, alpha, transition_prob_matrix_adjacent,
                                                               count_initial, list_predict_year, landcover_types):

    array_predict_pct = np.zeros((iterative_rounds, len(list_predict_year), 2*landcover_types+1), dtype=float)

    for i_round in range(0, iterative_rounds):
        # print(i_round)
        predict_adjacent_matrix = generate_adjust_iterative_predict_adjacent_matrix_population_based(alpha, transition_prob_matrix_adjacent, list_predict_year, landcover_types)
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


def plot_population_change_info(list_predict_year, array_population, array_population_change_rate):
    """
        plot the population change information, could be helpful to adjust the prediction matrix
    """

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(18, 8))
    legend_size = 12
    tick_label_size = 14
    axis_label_size = 16
    title_label_size = 18
    tick_length = 4

    t = np.arange(0, len(list_predict_year))

    # plt.plot(list_predict_year, array_population * array_population_change_rate * alpha / (1 + np.exp(-t)))
    # plt.plot(list_predict_year, array_population * array_population_change_rate * alpha * t)
    plt.plot(list_predict_year, array_population * array_population_change_rate)

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    axes.set_xlabel('predict year', size=axis_label_size)
    axes.set_ylabel('land cover percentage (%)', size=axis_label_size)

    # axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=5.0))

    plt.legend(loc='best', fontsize=legend_size, bbox_to_anchor=(1.04, 1))
    # plt.title(title, fontsize=title_label_size)
    plt.tight_layout()

    plt.show()


# def main():
if __name__ == '__main__':

    predict_flag = 'hindcast'
    reference_flag = 'end'
    output_version_flag = 'degrade_v2_refine_3_3'
    country_flag = 'hispaniola'

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
    ##
    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_predict_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_predict_year = np.arange(1996, 1491, -1)

    # print('observe years:', list_observe_year)
    # print('prediction years', list_prediction_year)

    (transition_prob_matrix_adjacent, transition_prob_matrix_accumulate_indirect,
     transition_prob_matrix_accumulate_direct, transition_prob_matrix_accumulate_direct_inverse) = read_prob_matrix(predict_flag, country_flag=country_flag)

    ##
    df_observe, count_initial = get_initial_count(output_version_flag, list_observe_year, country_flag, predict_flag, reference_flag, landcover_types)

    iterative_rounds = 10
    alpha = 0.20

    predict_adjacent_matrix, predict_accumulate_matrix, df_predict, df_average, df_95th, df_5th \
        = sum_alpha_base_prediction_matrix_generate_population_based(iterative_rounds, alpha, transition_prob_matrix_adjacent,
                                                                     count_initial, list_predict_year, landcover_types)

    ##
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
    array_population, array_population_change_rate = read_population_info(list_predict_year)
    plot_population_change_info(list_predict_year, array_population, array_population_change_rate)

    ##
    # output_rootpath = join(rootpath, 'results', 'land_change_modelling', output_version_flag, predict_flag,
    #                        'prediction_matrix', 'population_alpha_{:04d}'.format(int(alpha * 1000)))
    # print(output_rootpath)
    # if not os.path.exists(output_rootpath):
    #     os.makedirs(output_rootpath, exist_ok=True)
    #
    # np.save(join(output_rootpath, 'alpha_{:04d}_adjacent_matrix.npy'.format(int(alpha* 1000))), predict_adjacent_matrix)
    # np.save(join(output_rootpath, 'alpha_{:04d}_accumulate_matrix.npy'.format(int(alpha* 1000))), predict_accumulate_matrix)
    #
    # df_predict.to_csv(join(output_rootpath,'alpha_{:04d}_df_predict.csv'.format(int(alpha* 1000))))
    #
    # df_average.to_csv(join(output_rootpath, 'alpha_{:04d}_df_average.csv'.format(int(alpha* 1000))))
    # df_95th.to_csv(join(output_rootpath, 'alpha_{:04d}_df_95th.csv'.format(int(alpha* 1000))))
    # df_5th.to_csv(join(output_rootpath, 'alpha_{:04d}_df_5th.csv'.format(int(alpha* 1000))))
