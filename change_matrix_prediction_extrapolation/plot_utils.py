
import numpy as np
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as plticker
import seaborn as sns

def transition_prob_plot(transition_prob_matrix, landcover_id_from, landcover_id_to,
                         list_year=np.arange(1996, 2023),
                         x_axis_interval=3.0,
                         landcover_system=None):

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryWetForest',
                            '4': 'PrimaryDryForest',
                            '5': 'SecondaryForest',
                            '6': 'ShrubGrass',
                            '7': 'Cropland',
                            '8': 'Water',
                            '9': 'Wetland'}

    change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                               landcover_id_to, landcover_system[str(landcover_id_to)])

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(17, 8))
    legend_size = 20
    tick_label_size = 17
    axis_label_size = 20
    title_label_size = 24
    tick_length = 4

    axes.plot(list_year, transition_prob_matrix[:, landcover_id_from - 1, landcover_id_to - 1])
    axes.set_title('{}'.format(change_info))

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True,
                     which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True,
                     which='major')

    axes.set_xlabel('year', size=axis_label_size)
    axes.set_ylabel('change percentage', size=axis_label_size)

    # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))

    if list_year[0] > list_year[-1]:
        axes.invert_xaxis()

    plt.title(change_info, fontsize=title_label_size)
    plt.tight_layout()


def ensemble_transition_prob_plot(transition_prob_matrix, landcover_id, landcover_flag='from',
                                  list_year=np.arange(1996, 2023),
                                  x_axis_interval=3.0,
                                  landcover_system=None):

    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 16))
    legend_size = 20
    tick_label_size = 17
    axis_label_size = 20
    title_label_size = 24
    tick_length = 4

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryWetForest',
                            '4': 'PrimaryDryForest',
                            '5': 'SecondaryForest',
                            '6': 'ShrubGrass',
                            '7': 'Cropland',
                            '8': 'Water',
                            '9': 'Wetland'}

    landcover_types = np.shape(transition_prob_matrix)[-1]

    if landcover_flag == 'from':
        landcover_from = landcover_id
        for landcover_id_to in range(1, landcover_types + 1):

            change_info = 'from_{}_{}_to_{}_{}'.format(landcover_from, landcover_system[str(landcover_from)],
                                                       landcover_id_to, landcover_system[str(landcover_id_to)])

            plot_row_id = (landcover_id_to - 1) // nrows
            plot_col_id = (landcover_id_to - 1) % ncols

            axes[plot_row_id, plot_col_id].plot(list_year, transition_prob_matrix[:, landcover_from - 1, landcover_id_to - 1])
            axes[plot_row_id, plot_col_id].set_title('{}'.format(change_info))

            axes[plot_row_id, plot_col_id].tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
            axes[plot_row_id, plot_col_id].tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

            axes[plot_row_id, plot_col_id].set_xlabel('year', size=axis_label_size)
            axes[plot_row_id, plot_col_id].set_ylabel('percentage', size=axis_label_size)

            # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
            axes[plot_row_id, plot_col_id].xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))

            if list_year[0] > list_year[-1]:
                axes[plot_row_id, plot_col_id].invert_xaxis()

            axes[plot_row_id, plot_col_id].set_title(change_info, fontsize=title_label_size)

            axes[plot_row_id, plot_col_id].yaxis.offsetText.set_fontsize(tick_label_size)
    else:
        landcover_id_to = landcover_id

        for landcover_from in range(1, landcover_types + 1):

            change_info = 'from_{}_{}_to_{}_{}'.format(landcover_from, landcover_system[str(landcover_from)],
                                                       landcover_id_to, landcover_system[str(landcover_id_to)])

            plot_row_id = (landcover_from - 1) // nrows
            plot_col_id = (landcover_from - 1) % ncols

            axes[plot_row_id, plot_col_id].plot(list_year, transition_prob_matrix[:, landcover_from - 1, landcover_id_to - 1])
            axes[plot_row_id, plot_col_id].set_title('{}'.format(change_info))

            axes[plot_row_id, plot_col_id].tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
            axes[plot_row_id, plot_col_id].tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

            axes[plot_row_id, plot_col_id].set_xlabel('year', size=axis_label_size)
            axes[plot_row_id, plot_col_id].set_ylabel('percentage', size=axis_label_size)

            # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
            axes[plot_row_id, plot_col_id].xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))

            if list_year[0] > list_year[-1]:
                axes[plot_row_id, plot_col_id].invert_xaxis()

            axes[plot_row_id, plot_col_id].set_title(change_info, fontsize=title_label_size)

            axes[plot_row_id, plot_col_id].yaxis.offsetText.set_fontsize(tick_label_size)

    if landcover_types == 8:
        axes[-1, -1].axis('off')

    plt.tight_layout()

    return None


def line_plot_each_predictor(landcover_id_from,
                             landcover_id_to,
                             list_observe_year,
                             list_predict_year,
                             transition_prob_observe,

                             mk_chain_predict_matrix_from_1984,
                             arima_predict_matrix_normalized,
                             complete_reg_predict_matrix_normalized,
                             partial_reg_predict_matrix_normalized,
                             output_flag=0,
                             output_folder=' ',
                             x_axis_interval=5.0,
                             landcover_system=None
                             ):
    """
    plot the observe and predict transition prob of five predictors for one change type
    for example, plot the transition prob from primary forest to secondary forest
    """

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryWetForest',
                            '4': 'PrimaryDryForest',
                            '5': 'SecondaryForest',
                            '6': 'ShrubGrass',
                            '7': 'Cropland',
                            '8': 'Water',
                            '9': 'Wetland'}

    change_info = 'from_{}_{}_to_{}_{}'.format(landcover_id_from, landcover_system[str(landcover_id_from)],
                                               landcover_id_to, landcover_system[str(landcover_id_to)])
    # transition_prob_unit = transition_prob_matrix_accumulate_direct[:, landcover_id_from - 1, landcover_id_to - 1]

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(17, 8))
    legend_size = 20
    tick_label_size = 17
    axis_label_size = 20
    title_label_size = 24
    tick_length = 4

    plt.plot(list_observe_year, transition_prob_observe[:, landcover_id_from - 1, landcover_id_to - 1],
             color='black', label='observed change prob')

    plt.plot(list_predict_year, mk_chain_predict_matrix_from_1984[:, landcover_id_from - 1, landcover_id_to - 1],
             label='Markov chain predict')
    plt.plot(list_predict_year, arima_predict_matrix_normalized[:, landcover_id_from - 1, landcover_id_to - 1],
             label='ARIMA predict')
    plt.plot(list_predict_year,
             complete_reg_predict_matrix_normalized[:, landcover_id_from - 1, landcover_id_to - 1],
             label='complete_reg')
    plt.plot(list_predict_year, partial_reg_predict_matrix_normalized[:, landcover_id_from - 1, landcover_id_to - 1],
             label='partial_reg')

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True,
                     which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True,
                     which='major')

    axes.set_xlabel('year', size=axis_label_size)
    axes.set_ylabel('change probability (%)', size=axis_label_size)

    # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))

    if list_observe_year[0] < list_predict_year[0]:  # means forecast
        pass
    else:  # means hindcast
        axes.invert_xaxis()

    plt.legend(loc='best')
    plt.title(change_info)

    # plt.legend(loc='best', fontsize=legend_size, bbox_to_anchor=(1.04, 1))
    plt.legend(loc='best', fontsize=legend_size)
    plt.title(change_info, fontsize=title_label_size)
    plt.tight_layout()

    if output_flag == 0:
        plt.show()
    else:
        output_file = join(output_folder, '{}.jpg'.format(change_info))
        plt.savefig(output_file, dpi=300)
        plt.close()


def plot_change_matrix(change_matrix, x_label, y_label, title=None, vmin=0, vmax=1,
                       landcover_system=None):
    """
        plot the change matrix
        Args:
            change_matrix: the change matrix
            x_label
            y_label
            title
    """

    # landcover_system =

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryWetForest',
                            '4': 'PrimaryDryForest',
                            '5': 'SecondaryForest',
                            '6': 'ShrubGrass',
                            '7': 'Cropland',
                            '8': 'Water',
                            '9': 'Wetland'}

    df_cm = pd.DataFrame(change_matrix, index=landcover_system.values(), columns=landcover_system.values())

    figure, ax = plt.subplots(ncols=1, nrows=1, figsize=(11, 8))
    cmap = matplotlib.cm.GnBu

    tick_labelsize = 14
    axis_labelsize = 18
    annosize = 14
    ticklength = 4
    axes_linewidth = 1.5

    im = sns.heatmap(df_cm, annot=True, annot_kws={"size": annosize}, fmt='.4f', cmap=cmap, vmin=vmin, vmax=vmax)
    im.figure.axes[-1].yaxis.set_tick_params(labelsize=annosize)

    ax.tick_params('y', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major', rotation=0)
    ax.tick_params('x', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, top=False, which='major', rotation=30)

    ax.set_xlabel(x_label, size=axis_labelsize)
    ax.set_ylabel(y_label, size=axis_labelsize)
    ax.set_title(title, size=axis_labelsize)

    plt.tight_layout()


def plot_predict_pct(list_observe_year,
                     list_predict_year,
                     df_observe,
                     df_predict,
                     title=None,
                     output_flag=0,
                     output_folder=None,
                     axis_reverse_flag=True,
                     x_axis_interval=5.0):

    """    plot the predict percentage
    observe curve is plotted with dash line
    predict curve is plotted with solid line

    Args:
        list_observe_year: list, list including the observe year
        list_predict_year: list including the predict year
        df_observe: dataframe including the observed percentage
        df_predict: dataframe including the predict percentage
        title: plot title
        output_flag: flag indicating whether to output the figure
        output_folder: string, output folder
    """

    landcover_types = int((np.shape(df_predict)[1] - 1) / 2)
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
        plt.plot(list_observe_year, df_observe[lc_columns].values * 100, label=lc_columns, color=colors[i_landcover], linestyle='dashed')

    for i_landcover in range(0, landcover_types):
        lc_columns = df_predict.columns[np.where(df_predict.columns == 'TOTAL')[0][0] + i_landcover + 1]
        plt.plot(list_predict_year, df_predict[lc_columns].values * 100, label=lc_columns, color=colors[i_landcover], linestyle='solid')

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
    output_version_flag = 'irf_v52_0_5'
    country_flag = 'hispaniola'

    ##
    if predict_flag == 'forecast':
        list_observe_year = np.arange(1996, 2023)
        list_prediction_year = np.arange(2022, 2123, 1)
    else:
        list_observe_year = np.arange(2022, 1995, -1)
        list_prediction_year = np.arange(1996, 1895, -1)

    print('observe years:', list_observe_year)
    print('prediction years', list_prediction_year)


    ##
    from change_matrix.read_change_matrix import read_prob_matrix
    from change_matrix_prediction_extrapolation.change_matrix_direct_extrapolation import sum_matrix_generate

    transition_prob_matrix_adjacent, transition_prob_matrix_accumulate_indirect, \
    transition_prob_matrix_accumulate_direct, transition_prob_matrix_accumulate_direct_inverse \
        = read_prob_matrix(predict_flag, country_flag=country_flag)

    ensemble_transition_prob_plot(transition_prob_matrix_adjacent, landcover_id=2, landcover_flag='from', list_year=list_observe_year)
    ensemble_transition_prob_plot(transition_prob_matrix_adjacent, landcover_id=3, landcover_flag='to', list_year=list_observe_year)

    ##
    transition_prob_matrix, mk_chain_predict_matrix, arima_predict_matrix_normalized, \
    complete_reg_predict_matrix_normalized, partial_reg_predict_matrix_normalized \
        = sum_matrix_generate(list_observe_year, list_prediction_year, reference_flag,
                              transition_prob_matrix_accumulate_direct,
                              transition_prob_matrix_accumulate_direct_inverse,
                              partial_linear_offset=10)

    ##
    if predict_flag == 'forecast':
        landcover_id_from = 3
        landcover_id_to = 3
    else:
        landcover_id_from = 5
        landcover_id_to = 5

    line_plot_each_predictor(landcover_id_from,
                             landcover_id_to,
                             list_observe_year,
                             list_prediction_year,
                             transition_prob_matrix,

                             mk_chain_predict_matrix,
                             arima_predict_matrix_normalized,
                             complete_reg_predict_matrix_normalized,
                             partial_reg_predict_matrix_normalized,
                             output_flag=0, output_folder=' '
                             )

