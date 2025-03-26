"""
using the iterative was
"""
import time

import numpy as np
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.ar_model import ar_select_order
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
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


from change_matrix_prediction_extrapolation.plot_utils import plot_change_matrix, plot_predict_pct, ensemble_transition_prob_plot
from change_matrix_prediction_extrapolation.change_matrix_direct_extrapolation import (read_prob_matrix, predict_df_generate,
                                                                                       matrix_normalization_predict, get_initial_count)


def generate_iterative_predict_adjacent_matrix(transition_prob_matrix_adjacent, list_predict_year, landcover_types):
    """
        generate the prediction adjacent matrix using the iterative stochastic approach
        step 1: get the probability distribution function using the kernel density estimation approach in scipy
        step 2: generate the future change probability based on the PDF
        step 3: normalize the change matrix to make sure each row sum up to 1
    """

    predict_adjacent_matrix = np.zeros((len(list_predict_year) - 1, landcover_types, landcover_types), dtype=float)

    for landcover_id_from in range(1, landcover_types + 1):
        for landcover_id_to in range(1, landcover_types + 1):

            # change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
            #                                            landcover_id_to, landcover_system[str(landcover_id_to)])

            transition_adjacent_unit = transition_prob_matrix_adjacent[:, landcover_id_from - 1, landcover_id_to - 1]
            # sns_hist_plot(transition_adjacent_unit, title='Fitted point distribution')

            if np.nanstd(transition_adjacent_unit) == 0:
                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = transition_adjacent_unit[0]
            else:
                x_predict = np.linspace(transition_adjacent_unit.min(), transition_adjacent_unit.max(), 100)

                kde_sp = gaussian_kde(transition_adjacent_unit, bw_method='scott')
                fit_sp = kde_sp.pdf(transition_adjacent_unit)
                plot_sp = kde_sp.pdf(x_predict)

                # probability_distribution_function_plot(transition_adjacent_unit, x_predict, fit_sp, plot_sp,
                #                                        title='Fitted probability distribution frequency', x_label=None, y_label=None)

                predict_prob = np.random.choice(x_predict, size=len(list_predict_year) - 1, p=plot_sp / np.sum(plot_sp))
                # sns_hist_plot(predict_prob, title='Generated point distribution')

                predict_adjacent_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = predict_prob

    # plot_change_matrix(predict_adjacent_matrix[10,:,:], x_label=list_predict_year[10], y_label=list_predict_year[10+1])

    # normalize the transition matrix to make sure each row sum up to 1
    for i in range(0, np.shape(predict_adjacent_matrix)[0]):
        predict_adjacent_matrix[i, :, :] = matrix_normalization_predict(predict_adjacent_matrix[i, :, :])

    return predict_adjacent_matrix


def get_accumulate_indirect_prediction_matrix(predict_adjacent_matrix, list_predict_year, landcover_types):
    """
        get the accumulated change matrix
    """
    predict_accumulate_matrix = np.zeros((len(list_predict_year), landcover_types, landcover_types), dtype=float)

    for i_year in range(0, len(list_predict_year)):
        if i_year == 0:
            predict_accumulate_matrix[i_year, :, :] = np.identity(landcover_types)
        else:

            tmp = np.identity(landcover_types)
            for i_accumulate in range(0, i_year):
                tmp = tmp @ predict_adjacent_matrix[i_accumulate, :, :]

            predict_accumulate_matrix[i_year, :, :] = tmp

    return predict_accumulate_matrix


def plot_predict_pct_with_confident_interval(list_observe_year,
                                             list_predict_year,
                                             df_observe,
                                             df_average,
                                             df_5th,
                                             df_95th,
                                             title=None,
                                             output_flag=0,
                                             output_folder=None,
                                             axis_reverse_flag=True,
                                             x_axis_interval=5.0):
    df_observe = df_observe.astype(float)
    df_5th = df_5th.astype(float)
    df_95th = df_95th.astype(float)

    landcover_types = int((np.shape(df_average)[1] - 1) / 2)

    if landcover_types == 8:
        colors = np.array([np.array([241, 1, 0, 255]) / 255,
                           np.array([179, 175, 164, 255]) / 255,
                           np.array([29, 101, 51, 255]) / 255,
                           np.array([108, 169, 102, 255]) / 255,
                           np.array([208, 209, 129, 255]) / 255,
                           np.array([174, 114, 41, 255]) / 255,
                           np.array([72, 109, 162, 255]) / 255,
                           np.array([200, 230, 248, 255]) / 255
                           ])
    else:
        colors = np.array([np.array([241, 1, 0, 255]) / 255,
                           np.array([179, 175, 164, 255]) / 255,
                           np.array([29, 101, 51, 255]) / 255,
                           np.array([244, 127, 17, 255]) / 255,
                           np.array([108, 169, 102, 255]) / 255,
                           np.array([208, 209, 129, 255]) / 255,
                           np.array([174, 114, 41, 255]) / 255,
                           np.array([72, 109, 162, 255]) / 255,
                           np.array([186, 85, 211, 255]) / 255
                           ])

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(18, 8))
    legend_size = 12
    tick_label_size = 14
    axis_label_size = 16
    title_label_size = 18
    tick_length = 4

    for i_landcover in range(0, landcover_types):
        lc_columns = df_observe.columns[np.where(df_observe.columns == 'TOTAL')[0][0] + i_landcover + 1]
        land_cover_name = lc_columns[2:-2]
        plt.plot(list_observe_year, df_observe[lc_columns].values * 100, label='{} {} obs'.format(i_landcover+1, land_cover_name),
                 color=colors[i_landcover], linestyle='dashed')

        plt.plot(list_predict_year, df_average[lc_columns].values * 100, label='{} {} predict'.format(i_landcover+1, land_cover_name),
                 color=colors[i_landcover], linestyle='solid')

        plt.fill_between(list_predict_year, df_5th[lc_columns].values * 100, df_95th[lc_columns].values * 100,
                         label='{} {} 5%-95%'.format(i_landcover+1, land_cover_name),
                         color=colors[i_landcover], alpha=0.2)

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True,
                     which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True,
                     which='major')

    axes.set_xlabel('predict year', size=axis_label_size)
    axes.set_ylabel('land cover percentage (%)', size=axis_label_size)

    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    axes.yaxis.set_major_locator(plticker.MultipleLocator(base=5.0))

    if axis_reverse_flag == True:
        if list_observe_year[0] < list_predict_year[0]:  # means forecast
            pass
        else:  # means hindcast
            axes.invert_xaxis()

    plt.legend(loc='best', fontsize=legend_size, bbox_to_anchor=(1.04, 1))
    plt.title(title, fontsize=title_label_size)
    plt.tight_layout()

    if output_flag == 0:
        plt.show()
    else:
        output_file = join(output_folder, '{}.jpg'.format(title))
        plt.savefig(output_file, dpi=300)
        plt.close()


# def main():
if __name__ == '__main__':

    predict_flag = 'hindcast'

    # the prediction initial based year, if the observation year range is from 1984 to 2022, the reference year can
    # be 1984 ('start') or 2022 ('end')
    reference_flag = 'end'

    output_version_flag = 'degrade_v2_refine_3_3'
    country_flag = 'hispaniola'

    landcover_system = {'1': 'Developed',
                        '2': 'Barren',
                        '3': 'PrimaryWetForest',
                        '4': 'PrimaryDryForest',
                        '5': 'SecondaryForest',
                        '6': 'ShrubGrass',
                        '7': 'Cropland',
                        '8': 'Water',
                        '9': 'Wetland'}

    # np.set_printoptions(precision=4, suppress=True)
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

    landcover_types = np.shape(transition_prob_matrix_accumulate_direct)[-1]

    ##
    start_time = time.perf_counter()

    iterative_rounds = 50

    df_observe, count_initial = get_initial_count(output_version_flag, list_observe_year, country_flag, predict_flag, reference_flag, landcover_types)

    array_predict_pct = np.zeros((iterative_rounds, len(list_predict_year), 2*landcover_types+1), dtype=float)

    for i_round in range(0, iterative_rounds):
        print(f'simulation round {i_round}')
        predict_adjacent_matrix = generate_iterative_predict_adjacent_matrix(transition_prob_matrix_adjacent, list_predict_year, landcover_types)
        predict_accumulate_matrix = get_accumulate_indirect_prediction_matrix(predict_adjacent_matrix, list_predict_year, landcover_types)
        df_predict = predict_df_generate(count_initial, predict_accumulate_matrix, list_predict_year, landcover_types=landcover_types)

        array_predict_pct[i_round, :, :] = df_predict.values

    end_time = time.perf_counter()
    print('running time of {} rounds is {} seconds'.format(iterative_rounds, np.round((end_time - start_time), 2)))


    ##

    # get the average, 5th and 95th percentile of the prediction, which can be used to plot the confident interval
    array_average = np.nanmean(array_predict_pct, axis=0)
    df_average = df_predict.copy()
    df_average.iloc[:, :] = array_average

    df_95th = df_average.copy()
    df_5th = df_average.copy()

    df_95th.iloc[:, :] = np.nanpercentile(array_predict_pct, 95, axis=0)
    df_5th.iloc[:, :] = np.nanpercentile(array_predict_pct, 5, axis=0)

    ##

    plot_change_matrix(predict_accumulate_matrix[500, :, :], x_label=1996, y_label=list_predict_year[500], landcover_system=landcover_system)
    ensemble_transition_prob_plot(predict_accumulate_matrix, landcover_id=3, landcover_flag='to', list_year=list_predict_year, x_axis_interval=50)
    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_average,
                     title=None,
                     output_flag=0,
                     output_folder=None,
                     axis_reverse_flag=True,
                     x_axis_interval=50.0)

    ##
    plot_predict_pct_with_confident_interval(list_observe_year,
                                             list_predict_year,
                                             df_observe,
                                             df_average,
                                             df_5th,
                                             df_95th,
                                             title=None,
                                             output_flag=0,
                                             output_folder=None,
                                             axis_reverse_flag=True,
                                             x_axis_interval=50.0)




