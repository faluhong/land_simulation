"""
compare prediction matrices generated from auto_ARIMA, Auto Regression with selected order, Markov Chain,
complete linear regression (from 1996 to 2022), partial linear regression (from 2012 to 2022)
"""

import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.ar_model import ar_select_order
from scipy.stats import linregress
import pandas as pd
import matplotlib.ticker as plticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme()

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))

from change_matrix_prediction_extrapolation.plot_utils import line_plot_each_predictor, plot_predict_pct, plot_change_matrix
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


def markov_chain_predict_matrix(predict_date, start_year, end_year, transition_matrix):
    """
    generate the transition matrix using the regression-based apporach
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
    create the prediction markov chain matrix based on the input predict year

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
                                            partial_linear_offset=15,
                                            landcover_types=8):
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
                        transition_prob_matrix_accumulate_direct_inverse, partial_linear_offset=15):

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
        return transition_prob_matrix, \
            mk_chain_predict_matrix_from_end, arima_predict_matrix_normalized, \
            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized
    else:
        return transition_prob_matrix, \
            mk_chain_predict_matrix_from_start, arima_predict_matrix_normalized, \
            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized


def percentage_calculation(df, landcover_types=8):
    df['TOTAL'] = df.iloc[:, 0: landcover_types].sum(axis=1)
    for i in range(0, landcover_types):
        column_label = df.columns[i]
        df['{} %'.format(column_label)] = df[column_label] / df['TOTAL']
    return df


def predict_df_generate(count_initial, predict_matrix_from_initial, list_predict_year, landcover_types=8):
    count_predict = count_initial @ predict_matrix_from_initial
    count_predict = count_predict.astype(int)

    if landcover_types == 8:
        df_predict = pd.DataFrame(columns=['1 Developed', '2 Barren', '3 Primary forest',
                                           '4 Secondary forest', '5 Shrub/Grass', '6 Cropland',
                                           '7 Water', '8 Wetland', 'TOTAL',
                                           '1 Developed %', '2 Barren %', '3 Primary forest %',
                                           '4 Secondary forest %', '5 Shrub/Grass %', '6 Cropland %',
                                           '7 Water %', '8 Wetland %'],
                                  index=list_predict_year)

        for i_predict in range(0, len(list_predict_year)):
            year_predict = list_predict_year[i_predict]

            df_predict.loc[year_predict, ['1 Developed', '2 Barren', '3 Primary forest',
                                           '4 Secondary forest', '5 Shrub/Grass', '6 Cropland',
                                           '7 Water', '8 Wetland']] = count_predict[i_predict]

    else:
        df_predict = pd.DataFrame(columns=['1 Developed', '2 Barren', '3 Primary wet forest', '4 Primary dry forest',
                                           '5 Secondary forest', '6 Shrub/Grass', '7 Cropland',
                                           '8 Water', '9 Wetland', 'TOTAL',
                                           '1 Developed %', '2 Barren %', '3 Primary wet forest %', '4 Primary dry forest %',
                                           '5 Secondary forest %', '6 Shrub/Grass %', '7 Cropland %',
                                           '8 Water %', '9 Wetland %'],
                                  index=list_predict_year)

        for i_predict in range(0, len(list_predict_year)):
            year_predict = list_predict_year[i_predict]

            df_predict.loc[year_predict, ['1 Developed', '2 Barren', '3 Primary wet forest', '4 Primary dry forest',
                                          '5 Secondary forest', '6 Shrub/Grass', '7 Cropland',
                                          '8 Water', '9 Wetland']] = count_predict[i_predict]

    df_predict = percentage_calculation(df_predict, landcover_types)

    return df_predict


def sum_predict_df_generate(sheet_hispaniola, predict_flag, reference_flag, list_observe_year,
                            list_prediction_year,
                            mk_chain_predict_matrix, arima_predict_matrix_normalized,
                            complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized):
    landcover_types = np.shape(mk_chain_predict_matrix)[1]

    if predict_flag == 'forecast':
        df_observe = sheet_hispaniola
    else:
        df_observe = sheet_hispaniola[::-1]

    list_year = df_observe['Year'].values

    count_latest_year = df_observe.iloc[list_year == list_observe_year[-1], 2:landcover_types + 2].values
    count_oldest_year = df_observe.iloc[list_year == list_observe_year[0], 2:landcover_types + 2].values

    if reference_flag == 'end':
        count_initial = count_latest_year
    else:
        count_initial = count_oldest_year

    df_predict_mk_chain = predict_df_generate(count_initial, mk_chain_predict_matrix, list_prediction_year, landcover_types=landcover_types)
    df_predict_arima = predict_df_generate(count_initial, arima_predict_matrix_normalized, list_prediction_year, landcover_types=landcover_types)
    df_predict_complete_reg = predict_df_generate(count_initial, complete_reg_predict_matrix_normalized, list_prediction_year, landcover_types=landcover_types)
    df_predict_partial_reg = predict_df_generate(count_initial, partial_reg_predict_matrix_normalized, list_prediction_year, landcover_types=landcover_types)

    return df_observe, count_initial, df_predict_mk_chain, df_predict_arima, df_predict_complete_reg, df_predict_partial_reg


def read_obs_pct_file(output_version_flag, list_observe_year, country_flag='hispaniola'):
    """read the land cover statistical file

    Args:
        output_version_flag (_type_): the land cover version
        list_observe_year (_type_): the observation year, e.g., np.arange(1984, 2023), np.arange(1996, 2023)

    Returns:
        sheet_hispaniola: dataframe containing each land cover percentage
    """
    filename_percentile = join(rootpath, 'data',  '{}_landcover_analysis.xlsx'.format(output_version_flag))
    if country_flag == 'hispaniola':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Hispaniola')
    elif country_flag == 'dr':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Dominican')
    elif country_flag == 'haiti':
        sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Haiti')

    sheet_hispaniola = sheet_hispaniola.loc[sheet_hispaniola['Year'].isin(list_observe_year)]

    return sheet_hispaniola


def get_initial_count(output_version_flag, list_observe_year, country_flag, predict_flag, reference_flag, landcover_types):
    """
        get the initial land cover pixel count
    """
    sheet_hispaniola = read_obs_pct_file(output_version_flag=output_version_flag,
                                         list_observe_year=list_observe_year,
                                         country_flag=country_flag)

    if predict_flag == 'forecast':
        df_observe = sheet_hispaniola
    else:
        df_observe = sheet_hispaniola[::-1]

    list_year = df_observe['Year'].values

    count_latest_year = df_observe.iloc[list_year == list_observe_year[-1], 2:landcover_types + 2].values
    count_oldest_year = df_observe.iloc[list_year == list_observe_year[0], 2:landcover_types + 2].values

    if reference_flag == 'end':
        count_initial = count_latest_year
    else:
        count_initial = count_oldest_year

    return df_observe, count_initial



def pf_percentage_predict_plot(list_observe_year,
                               list_prediction_year,
                               df_observe,
                               df_predict_mk_chain,
                               df_predict_arima,
                               df_predict_complete_reg,
                               df_predict_partial_reg,
                               title=None,
                               output_flag=0,
                               output_folder=None):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(18, 8))
    legend_size = 12
    tick_label_size = 14
    axis_label_size = 16
    title_label_size = 18
    tick_length = 4

    plt.plot(list_observe_year, (df_observe['3 Primary wet forest %'].values + df_observe['4 Primary dry forest %'].values) * 100,
             label='Primary forest observe', color=np.array([29, 101, 51, 255]) / 255, linestyle='dashed')

    plt.plot(list_prediction_year, (df_predict_mk_chain['3 Primary wet forest %'].values + df_predict_mk_chain['4 Primary dry forest %'].values) * 100,
             label='Markov Chain predict', color='#d7191c')

    plt.plot(list_prediction_year, (df_predict_arima['3 Primary wet forest %'].values + df_predict_arima['4 Primary dry forest %'].values) * 100,
             label='Arima predict', color='#fdae61')

    plt.plot(list_prediction_year, (df_predict_complete_reg['3 Primary wet forest %'].values + df_predict_complete_reg['4 Primary dry forest %'].values) * 100,
             label='Complete linear regression predict', color='#5e3c99')

    plt.plot(list_prediction_year, (df_predict_partial_reg['3 Primary wet forest %'].values + df_predict_partial_reg['4 Primary dry forest %'].values) * 100,
             label='Partial linear regression predict', color='#2b83ba')

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True,
                     which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True,
                     which='major')

    axes.set_xlabel('predict year', size=axis_label_size)
    axes.set_ylabel('primary forest percentage (%)', size=axis_label_size)

    # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=5.0))

    if list_observe_year[0] < list_prediction_year[0]:  # means forecast
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

    predict_flag = 'forecast'

    # the prediction initial based year, if the observation year range is from 1984 to 2022, the reference year can
    # be 1984 ('start') or 2022 ('end')
    reference_flag = 'end'
    output_version_flag = 'degrade_v2_refine_3_3'
    country_flag = 'hispaniola'

    ##
    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_predict_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        # list_predict_year = np.arange(1996, 1895, -1)
        list_predict_year = np.arange(1996, 1491, -1)

    print('observe years:', list_observe_year)
    print('prediction years', list_predict_year)

    np.set_printoptions(precision=4, suppress=True)

    ##
    (transition_prob_matrix_adjacent, transition_prob_matrix_accumulate_indirect,
        transition_prob_matrix_accumulate_direct, transition_prob_matrix_accumulate_direct_inverse) = read_prob_matrix(predict_flag, country_flag=country_flag)

    ##
    transition_prob_matrix, mk_chain_predict_matrix, arima_predict_matrix_normalized, \
        complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized \
        = sum_matrix_generate(list_observe_year, list_predict_year, reference_flag,
                              transition_prob_matrix_accumulate_direct,
                              transition_prob_matrix_accumulate_direct_inverse,
                              partial_linear_offset=10)

    if predict_flag == 'forecast':
        landcover_id_from = 3
        landcover_id_to = 3
    else:
        landcover_id_from = 5
        landcover_id_to = 5

    ##
    # landcover_id_from = 4
    # landcover_id_to = 3

    line_plot_each_predictor(landcover_id_from,
                             landcover_id_to,
                             list_observe_year,
                             list_predict_year,
                             transition_prob_matrix,

                             mk_chain_predict_matrix,
                             arima_predict_matrix_normalized,
                             complete_reg_predict_matrix_normalized,
                             partial_reg_predict_matrix_normalized,
                             output_flag=0, output_folder=' ',
                             x_axis_interval=50
                             )
    ##

    sheet_hispaniola = read_obs_pct_file(output_version_flag=output_version_flag,
                                         list_observe_year=list_observe_year,
                                         country_flag=country_flag)

    df_observe, count_initial, df_predict_mk_chain, df_predict_arima, df_predict_complete_reg, df_predict_partial_reg = \
        sum_predict_df_generate(sheet_hispaniola, predict_flag, reference_flag, list_observe_year,
                                list_predict_year,
                                mk_chain_predict_matrix, arima_predict_matrix_normalized,
                                complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized)

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_mk_chain,
                     title='Markov chain prediction',
                     output_flag=0, output_folder=' ')

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_arima,
                     title='ARIMA prediction',
                     output_flag=0, output_folder=' ')

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_complete_reg,
                     title='Complete linear regression prediction',
                     output_flag=0, output_folder=' ')

    plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict_partial_reg,
                     title='Partial linear regression prediction',
                     output_flag=0, output_folder=' ')

    ##

    title = 'Primary forest prediction in {}'.format(country_flag)

    pf_percentage_predict_plot(list_observe_year, list_predict_year, df_observe,
                               df_predict_mk_chain, df_predict_arima, df_predict_complete_reg,
                               df_predict_partial_reg, title=title)


