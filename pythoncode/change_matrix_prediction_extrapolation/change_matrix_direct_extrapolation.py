"""
    generate the future change matrix using different extrapolation methods
        (1) Markov Chain
        (2) ARIMA (auto regression with integrated moving average)
        (3) Complete linear regression (from 1996 to 2022)
        (4) Partial linear regression (such as from 2012 to 2022)
"""

import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.ar_model import ar_select_order
from scipy.stats import linregress
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')   # ignore the warnings

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)


from change_matrix_prediction_extrapolation.plot_utils import (line_plot_each_predict_cell,
                                                               plot_predict_pct,
                                                               plot_change_matrix,
                                                               plot_predict_pf_percentage)

from change_matrix.read_change_matrix import read_prob_matrix
from change_matrix.util_read_hispaniola_lc import read_obs_pct_file


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


def markov_chain_predict_matrix(predict_date, start_year, end_year, transition_matrix):
    """
    generate the transition matrix using the Markon Chain regression-based approach

    Ref:
    Eastman, J. R., & He, J. (2020).
    A regression-based procedure for markov transition probability estimation in land change modeling.
    Land, 9(11), 407.
    https://www.mdpi.com/2073-445X/9/11/407/htm

    the final matrix will be normalized to make each row sum up to 1

    Args:
        predict_date: the year you want to predict using the markov-chain
        start_year: start year
        end_year: end year
        transition_matrix: transition matrix from start year to end year
    """

    matrix_shape = transition_matrix.shape[0]  # variable represent the matrix size, i.e., the land cover types

    lc_date_range = end_year - start_year
    if (predict_date - end_year) % lc_date_range == 0:
        power_id = int((predict_date - end_year) / lc_date_range)
        transition_matrix_predict = np.linalg.matrix_power(transition_matrix, power_id)

    else:
        power_id = int((predict_date - end_year) // lc_date_range)

        if power_id == 0:
            transition_matrix_fitting_1 = np.identity(matrix_shape)
            transition_matrix_fitting_2 = np.linalg.matrix_power(transition_matrix, 1)
            transition_matrix_fitting_3 = np.linalg.matrix_power(transition_matrix, 2)
        else:
            transition_matrix_fitting_1 = np.linalg.matrix_power(transition_matrix, power_id)
            transition_matrix_fitting_2 = np.linalg.matrix_power(transition_matrix, power_id + 1)
            transition_matrix_fitting_3 = np.linalg.matrix_power(transition_matrix, power_id + 2)

        t_predict = (predict_date - (end_year + power_id * lc_date_range)) / lc_date_range

        transition_matrix_predict = np.zeros((matrix_shape, matrix_shape), dtype=float)
        for i in range(0, matrix_shape):
            for j in range(0, matrix_shape):
                probability_fitting = np.array([transition_matrix_fitting_1[i, j],
                                                transition_matrix_fitting_2[i, j],
                                                transition_matrix_fitting_3[i, j]]
                                               )

                t_fitting = np.arange(0, 3)
                coefs = np.polyfit(t_fitting, probability_fitting, 2)

                poly = np.poly1d(coefs)
                prob_predict = poly(t_predict)

                transition_matrix_predict[i, j] = prob_predict

    transition_matrix_predict = matrix_normalization_predict(transition_matrix_predict)

    return transition_matrix_predict


def markov_chain_predict_matrix_generate(transition_prob_matrix_start_end, list_train_year, list_predict_year,
                                         landcover_types=8):
    """
    create the prediction Markov Chain matrix based on the input predict year

    Args:
        transition_prob_matrix_start_end
        list_train_year: list containing the observation year, used to train the markov chain
        list_predict_year: list containing the years you want markov chain to predict
    """

    start_year = list_train_year[0]
    end_year = list_train_year[-1]

    mk_chain_predict_matrix = np.zeros((len(list_predict_year), landcover_types, landcover_types), dtype=float)
    for i_predict in range(0, len(list_predict_year)):
        predict_year = list_predict_year[i_predict]

        mk_chain_predict_matrix[i_predict, :, :] = markov_chain_predict_matrix(predict_year, start_year,
                                                                               end_year,
                                                                               transition_prob_matrix_start_end)

    return mk_chain_predict_matrix


def auto_regression_predict_matrix_generate(transition_prob_matrix_observed,
                                            list_train_year,
                                            list_predict_year,
                                            auto_flag='arima',
                                            partial_linear_offset=10,
                                            landcover_types=8):
    """
        generate the predicted change matrix using auto regression-based approaches, including auto ARIMA, auto regression with selected order,
        complete linear regression, and partial linear regression

        :param transition_prob_matrix_observed:
        :param list_train_year:
        :param list_predict_year:
        :param auto_flag:
        :param partial_linear_offset:
        :param landcover_types:
        :return:
    """

    ar_predict_matrix = np.zeros((len(list_predict_year), landcover_types, landcover_types), dtype=float)

    for landcover_id_from in range(1, landcover_types + 1):
        for landcover_id_to in range(1, landcover_types + 1):

            transition_prob_unit = transition_prob_matrix_observed[:, landcover_id_from - 1, landcover_id_to - 1]

            if auto_flag == 'arima':
                res_arima = pm.auto_arima(transition_prob_unit, suppress_warnings=True)
                pred_ar = res_arima.predict(len(list_predict_year))
            elif auto_flag == 'ar_select_order':
                sel = ar_select_order(transition_prob_unit, 15, seasonal=False, old_names=False)
                res_auto_regression_select_order = sel.model.fit()
                pred_ar = res_auto_regression_select_order.predict(start=len(transition_prob_unit),
                                                                   end=len(transition_prob_unit) +
                                                                       len(list_predict_year) - 1)
            elif auto_flag == 'complete_reg':
                x_fit = np.arange(0, len(transition_prob_unit))
                res = linregress(x_fit, transition_prob_unit)

                x_predict = np.arange(len(transition_prob_unit), len(transition_prob_unit) + len(list_predict_year))
                pred_ar = res.slope * x_predict + res.intercept
            else:
                list_fit_year = list_train_year[-partial_linear_offset::]
                x_fit = np.arange(0, len(list_fit_year))

                if list_train_year[0] < list_predict_year[0]:  # means forecast
                    res = linregress(x_fit, transition_prob_unit[list_train_year >= list_fit_year[0]])
                else:  # means hindcast
                    res = linregress(x_fit, transition_prob_unit[list_train_year <= list_fit_year[0]])

                x_predict = np.arange(len(list_fit_year), len(list_fit_year) + len(list_predict_year))
                pred_ar = res.slope * x_predict + res.intercept

            pred_ar = pred_ar - (pred_ar[0] - transition_prob_unit[-1])
            ar_predict_matrix[:, landcover_id_from - 1, landcover_id_to - 1] = pred_ar

    ar_predict_matrix_normalized = np.zeros(np.shape(ar_predict_matrix), dtype=float)
    for i in range(0, len(list_predict_year)):
        ar_predict_matrix_normalized[i] = matrix_normalization_predict(ar_predict_matrix[i])

    return ar_predict_matrix_normalized


def sum_matrix_generate(list_observe_year, list_prediction_year, reference_flag,
                        transition_prob_matrix_accumulate_direct,
                        transition_prob_matrix_accumulate_direct_inverse,
                        partial_linear_offset=10):
    """
        generate the prediction matrix using Markov Chain, auto ARIMA, complete linear regression, and partial linear regression

        :param list_observe_year:
        :param list_prediction_year:
        :param reference_flag:
        :param transition_prob_matrix_accumulate_direct:
        :param transition_prob_matrix_accumulate_direct_inverse:
        :param partial_linear_offset:
        :return:
    """

    if reference_flag == 'end':
        transition_prob_matrix = transition_prob_matrix_accumulate_direct_inverse
    else:
        transition_prob_matrix = transition_prob_matrix_accumulate_direct

    landcover_types = np.shape(transition_prob_matrix_accumulate_direct)[-1]

    transition_prob_matrix_accumulate_direct_start_end = transition_prob_matrix_accumulate_direct[-1, :, :]

    mk_chain_predict_matrix_from_end = markov_chain_predict_matrix_generate(
        transition_prob_matrix_accumulate_direct_start_end,
        list_observe_year,
        list_prediction_year,
        landcover_types=landcover_types)
    mk_chain_predict_matrix_from_start = transition_prob_matrix_accumulate_direct_start_end @ mk_chain_predict_matrix_from_end

    arima_predict_matrix_normalized = auto_regression_predict_matrix_generate(
        transition_prob_matrix,
        list_observe_year,
        list_prediction_year,
        auto_flag='arima',
        landcover_types=landcover_types)

    complete_reg_predict_matrix_normalized = auto_regression_predict_matrix_generate(
        transition_prob_matrix,
        list_observe_year,
        list_prediction_year,
        auto_flag='complete_reg',
        landcover_types=landcover_types)

    partial_reg_predict_matrix_normalized = auto_regression_predict_matrix_generate(
        transition_prob_matrix,
        list_observe_year,
        list_prediction_year,
        auto_flag='partial_reg',
        partial_linear_offset=partial_linear_offset,
        landcover_types=landcover_types)

    if reference_flag == 'end':
        return (transition_prob_matrix,
            mk_chain_predict_matrix_from_end, arima_predict_matrix_normalized,
            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized)
    else:
        return (transition_prob_matrix,
            mk_chain_predict_matrix_from_start, arima_predict_matrix_normalized,
            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized)


def percentage_calculation(df, landcover_types=8):
    df['TOTAL'] = df.iloc[:, 0: landcover_types].sum(axis=1)
    for i in range(0, landcover_types):
        column_label = df.columns[i]
        df['{} %'.format(column_label)] = df[column_label] / df['TOTAL']
    return df


def predict_df_generate(count_initial, predict_matrix_from_initial, list_predict_year, landcover_types=8):

    count_predict = count_initial @ predict_matrix_from_initial
    count_predict = count_predict.astype(int)

    df_predict = pd.DataFrame(columns=['1 Developed', '2 Primary wet forest', '3 Primary dry forest',
                                       '4 Secondary forest', '5 Shrub/Grass', '6 Water', '7 Wetland', '8 Other',
                                       'TOTAL',
                                       '1 Developed %', '2 Primary wet forest %', '3 Primary dry forest %',
                                       '4 Secondary forest %', '5 Shrub/Grass %', '6 Water %', '7 Wetland %',
                                       '8 Other %'],
                              index=list_predict_year)

    for i_predict in range(0, len(list_predict_year)):
        year_predict = list_predict_year[i_predict]

        df_predict.loc[year_predict, ['1 Developed', '2 Primary wet forest', '3 Primary dry forest',
                                      '4 Secondary forest', '5 Shrub/Grass',
                                      '6 Water', '7 Wetland', '8 Other', ]] = count_predict[i_predict]

    df_predict = percentage_calculation(df_predict, landcover_types)

    return df_predict


def sum_predict_df_generate(sheet_hispaniola, predict_flag, reference_flag, list_observe_year,
                            list_predict_year,
                            mk_chain_predict_matrix, arima_predict_matrix_normalized,
                            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized):

    landcover_types = np.shape(mk_chain_predict_matrix)[1]

    if predict_flag == 'forecast':
        df_observe = sheet_hispaniola
    else:
        df_observe = sheet_hispaniola[::-1]

    list_year = df_observe['Year'].values

    count_latest_year = df_observe.iloc[list_year == list_observe_year[-1], 1:landcover_types + 1].values
    count_oldest_year = df_observe.iloc[list_year == list_observe_year[0], 1:landcover_types + 1].values

    if reference_flag == 'end':
        count_initial = count_latest_year
    else:
        count_initial = count_oldest_year

    df_predict_mk_chain = predict_df_generate(count_initial, mk_chain_predict_matrix, list_predict_year, landcover_types=landcover_types)
    df_predict_arima = predict_df_generate(count_initial, arima_predict_matrix_normalized, list_predict_year, landcover_types=landcover_types)
    df_predict_complete_reg = predict_df_generate(count_initial, complete_reg_predict_matrix_normalized, list_predict_year, landcover_types=landcover_types)
    df_predict_partial_reg = predict_df_generate(count_initial, partial_reg_predict_matrix_normalized, list_predict_year, landcover_types=landcover_types)

    return (df_observe, count_initial, df_predict_mk_chain, df_predict_arima, df_predict_complete_reg, df_predict_partial_reg)


def output_prediction(landcover_version,
                      mk_chain_predict_matrix, arima_predict_matrix_normalized,
                      complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized,
                      df_predict_mk_chain, df_predict_arima,
                      df_predict_complete_reg,
                      df_predict_partial_reg,
                      country_flag='hispaniola'):
    """
        output the prediction matrix and the pixel count dataframe to the output folder

        :param landcover_version:
        :param mk_chain_predict_matrix:
        :param arima_predict_matrix_normalized:
        :param complete_reg_predict_matrix_normalized:
        :param partial_reg_predict_matrix_normalized:
        :param df_predict_mk_chain:
        :param df_predict_arima:
        :param df_predict_complete_reg:
        :param df_predict_partial_reg:
        :param country_flag:
        :return:
    """

    output_rootpath = join(rootpath_project, 'results', 'land_change_modelling',  landcover_version, predict_flag, 'prediction_matrix')

    if not os.path.exists(output_rootpath):
        os.makedirs(output_rootpath, exist_ok=True)

    if country_flag == 'hispaniola':

        np.save(join(output_rootpath, 'matrix_mk_chain.npy'), mk_chain_predict_matrix)
        np.save(join(output_rootpath, 'matrix_arima.npy'), arima_predict_matrix_normalized)
        np.save(join(output_rootpath, 'matrix_complete_reg.npy'), complete_reg_predict_matrix_normalized)
        np.save(join(output_rootpath, 'matrix_partial_reg.npy'), partial_reg_predict_matrix_normalized)

        df_predict_mk_chain.to_csv(join(output_rootpath, 'pixel_count_mk_chain.csv'))
        df_predict_arima.to_csv(join(output_rootpath, 'pixel_count_arima.csv'))
        df_predict_complete_reg.to_csv(join(output_rootpath, 'pixel_count_complete_reg.csv'))
        df_predict_partial_reg.to_csv(join(output_rootpath, 'pixel_count_partial_reg.csv'))
    else:
        np.save(join(output_rootpath, '{}_matrix_mk_chain.npy'.format(country_flag)), mk_chain_predict_matrix)
        np.save(join(output_rootpath, '{}_matrix_arima.npy'.format(country_flag)), arima_predict_matrix_normalized)
        np.save(join(output_rootpath, '{}_matrix_complete_reg.npy'.format(country_flag)), complete_reg_predict_matrix_normalized)
        np.save(join(output_rootpath, '{}_matrix_partial_reg.npy'.format(country_flag)), partial_reg_predict_matrix_normalized)

        df_predict_mk_chain.to_csv(join(output_rootpath, '{}_pixel_count_mk_chain.csv'.format(country_flag)))
        df_predict_arima.to_csv(join(output_rootpath, '{}_pixel_count_arima.csv'.format(country_flag)))
        df_predict_complete_reg.to_csv(join(output_rootpath, '{}_pixel_count_complete_reg.csv'.format(country_flag)))
        df_predict_partial_reg.to_csv(join(output_rootpath, '{}_pixel_count_partial_reg.csv'.format(country_flag)))


# def main():
if __name__ == '__main__':

    predict_flag = 'forecast'

    # the prediction initial based year, if the observation year range is from 1996 to 2022, the reference year can be 1996 ('start') or 2022 ('end')
    reference_flag = 'end'
    country_flag = 'haiti'     # 'hispaniola', 'haiti', or 'dr'
    landcover_version = 'publish_v1'

    ##
    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_predict_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_predict_year = np.arange(1996, 1491, -1)

    print('country flag:', country_flag)
    print('observe years:', list_observe_year)
    print('prediction years', list_predict_year)

    np.set_printoptions(precision=4, suppress=True)

    ##
    # read the change matrix
    (transition_prob_matrix_adjacent,
     transition_prob_matrix_accumulate_indirect,
     transition_prob_matrix_accumulate_direct,
     transition_prob_matrix_accumulate_direct_inverse) = read_prob_matrix(predict_flag, country_flag=country_flag)

    ##
    (transition_prob_matrix,
     mk_chain_predict_matrix,
     arima_predict_matrix_normalized,
     complete_reg_predict_matrix_normalized,
     partial_reg_predict_matrix_normalized) = sum_matrix_generate(list_observe_year,
                                                                  list_predict_year,
                                                                  reference_flag,
                                                                  transition_prob_matrix_accumulate_direct,
                                                                  transition_prob_matrix_accumulate_direct_inverse,
                                                                  partial_linear_offset=10)

    sheet_hispaniola = read_obs_pct_file(list_observe_year=list_observe_year,
                                         country_flag=country_flag)

    (df_observe, count_initial,
     df_predict_mk_chain,
     df_predict_arima,
     df_predict_complete_reg,
     df_predict_partial_reg) = sum_predict_df_generate(sheet_hispaniola,
                                                       predict_flag,
                                                       reference_flag,
                                                       list_observe_year,
                                                       list_predict_year,
                                                       mk_chain_predict_matrix,
                                                       arima_predict_matrix_normalized,
                                                       complete_reg_predict_matrix_normalized,
                                                       partial_reg_predict_matrix_normalized)

    ##
    output_prediction(landcover_version,
                      mk_chain_predict_matrix, arima_predict_matrix_normalized,
                      complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized,
                      df_predict_mk_chain, df_predict_arima,
                      df_predict_complete_reg, df_predict_partial_reg,
                      country_flag=country_flag)

    ##
    index_year = 20
    plot_change_matrix(complete_reg_predict_matrix_normalized[index_year],
                       x_label=f'{list_predict_year[index_year]}',
                       y_label=f'{list_predict_year[0]}',)

    ##
    if predict_flag == 'forecast':
        landcover_id_from = 2
        landcover_id_to = 2
    else:
        landcover_id_from = 5
        landcover_id_to = 5

    line_plot_each_predict_cell(landcover_id_from,
                                landcover_id_to,
                                list_observe_year,
                                list_predict_year,
                                transition_prob_matrix,
                                mk_chain_predict_matrix,
                                arima_predict_matrix_normalized,
                                complete_reg_predict_matrix_normalized,
                                partial_reg_predict_matrix_normalized,
                                output_flag=False,
                                output_folder=' ',
                                x_axis_interval=50
                                )

    ##
    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_mk_chain,
                     title='Markov chain prediction',
                     output_flag=False,
                     output_folder=None)

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_arima,
                     title='ARIMA prediction',
                     output_flag=False,
                     output_folder=None)

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_complete_reg,
                     title='Complete linear regression prediction',
                     output_flag=False,
                     output_folder=None)

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_partial_reg,
                     title='Partial linear regression prediction',
                     output_flag=False,
                     output_folder=None)

    ##

    title = 'Primary forest prediction in {}'.format(country_flag)

    plot_predict_pf_percentage(list_observe_year, list_predict_year, df_observe,
                               df_predict_mk_chain, df_predict_arima, df_predict_complete_reg,
                               df_predict_partial_reg, title=title)













