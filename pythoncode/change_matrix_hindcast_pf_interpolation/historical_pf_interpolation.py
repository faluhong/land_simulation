"""
    test the sensitivity of the PCHIP interpolation method by using different years combination

    using different years to build the PCHIP interpolation function, then compare the interpolation results

    use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation method to
"""

import numpy as np
from os.path import join
import pandas as pd
from osgeo import gdal, gdal_array, gdalconst
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.ticker as plticker
import matplotlib
from scipy.interpolate import PchipInterpolator

pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)

from change_matrix.util_read_hispaniola_lc import land_cover_map_read_hispaniola, read_obs_pct_file


def get_mask_topography_images():
    """
        get the mask boundary images and the DEM image

        These files are used to calculate the anchor points of historical PF percentage
    """

    img_country_mask = gdal_array.LoadFile(join(rootpath_project, 'data', 'countryid_hispaniola.tif'))

    filename_haiti_inside_mask = join(rootpath_project, 'data', 'historical_boundaries',
                                      'Western-Tiburon_clip.tif')
    img_haiti_inside_mask = gdal_array.LoadFile(filename_haiti_inside_mask)  # 255 indicates the western Tiburon, 0 indicates other

    filename_dr_inside_mask = join(rootpath_project, 'data', 'historical_boundaries',
                                   'Western-DR_clip.tif')
    img_dr_inside_mask = gdal_array.LoadFile(filename_dr_inside_mask)  # 255 indicates the western DR, 0 indicates other

    filename_amerindian_agriculture = join(rootpath_project, 'data', 'historical_boundaries',
                                           'Amerindian_Agriculture_clip.tif')
    img_amerindian_agriculture = gdal_array.LoadFile(filename_amerindian_agriculture)

    predictor_variable_dem_path = join(rootpath_project, 'data', 'dem', 'hispaniola_dem_info')
    img_dem = gdal_array.LoadFile(join(predictor_variable_dem_path, 'dem_mosaic.tif'))

    return (img_country_mask, img_haiti_inside_mask, img_dr_inside_mask, img_amerindian_agriculture, img_dem)


def generate_anchor_points_dataframe(img_country_mask, img_haiti_inside_mask, img_dr_inside_mask,
                                     img_amerindian_agriculture, img_dem):
    """
        generate the anchor points dataframe for the historical primary forest percentage in Haiti and DR
    """

    array_anchor_year = np.array([1492, 1697, 1804, 1875, 1921])

    df_anchor_points = pd.DataFrame(columns=['year', 'haiti', 'dr'],
                                    index=np.arange(0, len(array_anchor_year)))

    df_anchor_points['year'] = array_anchor_year

    # 1492, all PF except for the Amerindian agriculture
    mask_haiti_1492 = (img_country_mask == 1) & (img_amerindian_agriculture == 0)
    mask_dr_1492 = (img_country_mask == 2) & (img_amerindian_agriculture == 0)

    df_anchor_points.loc[0, 'haiti'] = np.count_nonzero(mask_haiti_1492) / np.count_nonzero(img_country_mask == 1)
    df_anchor_points.loc[0, 'dr'] = np.count_nonzero(mask_dr_1492) / np.count_nonzero(img_country_mask == 2)

    # 1697, Haiti > 100 meters, DR > 50 meters, except for the Amerindian agriculture
    mask_haiti_1697 = (img_country_mask == 1) & (img_dem > 100) & (img_amerindian_agriculture == 0)
    mask_dr_1697 = (img_country_mask == 2) & (img_dem > 50) & (img_amerindian_agriculture == 0)

    df_anchor_points.loc[1, 'haiti'] = np.count_nonzero(mask_haiti_1697) / np.count_nonzero(img_country_mask == 1)
    df_anchor_points.loc[1, 'dr'] = np.count_nonzero(mask_dr_1697) / np.count_nonzero(img_country_mask == 2)

    # 1804,except for the Amerindian agriculture
    # Haiti-other > 600 meters, HAITI-W-Tiburon > 300 meters
    # Western DR > 300 meters, Eastern DR > 100 meters
    mask_haiti_1804_w_tiburon = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 300)
                                 & (img_haiti_inside_mask == 255))
    mask_haiti_1804_other = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 600)
                             & (img_haiti_inside_mask == 0))
    mask_haiti_1804 = mask_haiti_1804_other | mask_haiti_1804_w_tiburon

    mask_dr_1804_western = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 300)
                            & (img_dr_inside_mask == 255))
    mask_dr_1804_eastern = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 100)
                            & (img_dr_inside_mask == 0))
    mask_dr_1804 = mask_dr_1804_western | mask_dr_1804_eastern

    df_anchor_points.loc[2, 'haiti'] = np.count_nonzero(mask_haiti_1804) / np.count_nonzero(img_country_mask == 1)
    df_anchor_points.loc[2, 'dr'] = np.count_nonzero(mask_dr_1804) / np.count_nonzero(img_country_mask == 2)

    # 1875, except for the Amerindian agriculture
    # Haiti-other > 900 meters, HAITI-W-Tiburon > 600 meters
    # Western DR > 600 meters, Eastern DR > 200 meters
    mask_haiti_1875_w_tiburon = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 600)
                                 & (img_haiti_inside_mask == 255))
    mask_haiti_1875_other = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 900)
                             & (img_haiti_inside_mask == 0))
    mask_haiti_1875 = mask_haiti_1875_other | mask_haiti_1875_w_tiburon

    mask_dr_1875_western = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 600)
                            & (img_dr_inside_mask == 255))
    mask_dr_1875_eastern = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 200)
                            & (img_dr_inside_mask == 0))
    mask_dr_1875 = mask_dr_1875_western | mask_dr_1875_eastern

    df_anchor_points.loc[3, 'haiti'] = np.count_nonzero(mask_haiti_1875) / np.count_nonzero(img_country_mask == 1)
    df_anchor_points.loc[3, 'dr'] = np.count_nonzero(mask_dr_1875) / np.count_nonzero(img_country_mask == 2)

    # 1921, except for the Amerindian agriculture
    # Haiti-other > 1200 meters, HAITI-W-Tiburon > 900 meters
    # Western DR > 900 meters, Eastern DR > 300 meters
    mask_haiti_1921_w_tiburon = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 900)
                                 & (img_haiti_inside_mask == 255))
    mask_haiti_1921_other = ((img_country_mask == 1) & (img_amerindian_agriculture == 0) & (img_dem > 1200)
                             & (img_haiti_inside_mask == 0))
    mask_haiti_1921 = mask_haiti_1921_other | mask_haiti_1921_w_tiburon

    mask_dr_1921_western = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 900)
                            & (img_dr_inside_mask == 255))
    mask_dr_1921_eastern = ((img_country_mask == 2) & (img_amerindian_agriculture == 0) & (img_dem > 300)
                            & (img_dr_inside_mask == 0))
    mask_dr_1921 = mask_dr_1921_western | mask_dr_1921_eastern

    df_anchor_points.loc[4, 'haiti'] = np.count_nonzero(mask_haiti_1921) / np.count_nonzero(img_country_mask == 1)
    df_anchor_points.loc[4, 'dr'] = np.count_nonzero(mask_dr_1921) / np.count_nonzero(img_country_mask == 2)

    df_anchor_points['haiti'] = df_anchor_points['haiti'] * 100
    df_anchor_points['dr'] = df_anchor_points['dr'] * 100

    return df_anchor_points


def plot_hindcast_curve(df_anchor_points_extend,
                        array_hindcast_year,
                        array_haiti_pchip,
                        array_dr_pchip,
                        array_observe_year,
                        array_pf_obs_haiti,
                        array_pf_obs_dr,
                        title=None,
                        fig_size=(16, 8),
                        ):
    """
        plot the PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) curves for the hindcast PF percentage in Haiti and DR
    """

    array_anchor_year = df_anchor_points_extend['year'].values

    array_anchor_year_plot = np.array([1492, 1697, 1804, 1875, 1921])

    matplotlib.rcParams['font.family'] = 'arial'

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=fig_size)
    legend_size = 22
    tick_label_size = 26
    axis_label_size = 28
    title_label_size = 30
    tick_length = 4

    line_width = 3.0

    x_axis_interval = 100
    x_label = 'Year'
    y_label = 'Primary forest percentage (%)'

    color_haiti = '#f97306'
    color_dr = '#0165fc'

    for spine in axes.spines.values():
        spine.set_linewidth(2.0)  # Set the desired width here

    plt.scatter(array_anchor_year_plot, df_anchor_points_extend['haiti'].values[np.isin(array_anchor_year, array_anchor_year_plot)],
                marker='o', s=100, color=color_haiti,
                label='Haiti anchor points')

    plt.plot(array_observe_year, array_pf_obs_haiti, linewidth=line_width,
             linestyle=(0, (3, 2)), label='Haiti observed', color=color_haiti)

    plt.plot(array_hindcast_year, array_haiti_pchip, linewidth=line_width,
             linestyle='solid', label='Haiti hindcast', color=color_haiti)

    plt.scatter(array_anchor_year_plot, df_anchor_points_extend['dr'].values[np.isin(array_anchor_year, array_anchor_year_plot)],
                marker='^', s=150,
                label='DR anchor points', color=color_dr)

    plt.plot(array_observe_year, array_pf_obs_dr, linewidth=line_width,
             linestyle=(0, (3, 2)), label='DR observed', color=color_dr)

    plt.plot(array_hindcast_year, array_dr_pchip, linewidth=line_width,
             linestyle='solid', label='DR hindcast', color=color_dr)

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

    axes.set_xlabel(x_label, size=axis_label_size)
    axes.set_ylabel(y_label, size=axis_label_size)

    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))
    # axes.yaxis.set_major_locator(plticker.MultipleLocator(base=5.0))

    axes.set_title(title, size=title_label_size)

    # axes.set_xlim([2023, 2122])
    plt.legend(fontsize=legend_size, loc='best')
    plt.tight_layout()
    plt.show()



# def main():
if __name__ == '__main__':

    (img_country_mask, img_haiti_inside_mask,
     img_dr_inside_mask, img_amerindian_agriculture, img_dem) = get_mask_topography_images()

    df_anchor_points_hindcast = generate_anchor_points_dataframe(img_country_mask,
                                                                 img_haiti_inside_mask,
                                                                 img_dr_inside_mask,
                                                                 img_amerindian_agriculture,
                                                                 img_dem)

    array_observe_year = np.arange(1996, 2023)

    sheet_obs_haiti = read_obs_pct_file(list_observe_year=array_observe_year,
                                        country_flag='haiti')

    sheet_obs_dr = read_obs_pct_file(list_observe_year=array_observe_year,
                                     country_flag='dr')

    array_pf_obs_haiti = sheet_obs_haiti['2 Primary wet forest %'].values + sheet_obs_haiti['3 Primary dry forest %'].values
    array_pf_obs_dr = sheet_obs_dr['2 Primary wet forest %'].values + sheet_obs_dr['3 Primary dry forest %'].values

    array_pf_obs_haiti = array_pf_obs_haiti * 100
    array_pf_obs_dr = array_pf_obs_dr * 100

    # add the 1996-2022 observed primary forest percentage to the dataframe
    new_row = pd.DataFrame({'year': array_observe_year,
                            'haiti': array_pf_obs_haiti,
                            'dr': array_pf_obs_dr})

    df_anchor_points_extend = pd.concat([df_anchor_points_hindcast, new_row], ignore_index=True)

    ##
    array_anchor_year = df_anchor_points_extend['year'].values

    df_hindcast_interpolation_all = pd.DataFrame()

    array_hindcast_year = np.arange(1996, array_anchor_year[0], -1)

    # interpolate the hindcast curve using PCHIP interpolation method
    pchip_haiti = PchipInterpolator(array_anchor_year, df_anchor_points_extend['haiti'].values)
    array_haiti_pchip = pchip_haiti(array_hindcast_year)

    pchip_dr = PchipInterpolator(array_anchor_year, df_anchor_points_extend['dr'].values)
    array_dr_pchip = pchip_dr(array_hindcast_year)

    df_hindcast_interpolation = pd.DataFrame({'year': array_hindcast_year,
                                              'haiti': array_haiti_pchip,
                                              'dr': array_dr_pchip})

    df_hindcast_interpolation_all = pd.concat([df_hindcast_interpolation_all, df_hindcast_interpolation],
                                              ignore_index=True,
                                              axis=1)

    # plot the hindcast curve
    plot_hindcast_curve(df_anchor_points_extend,
                        array_hindcast_year,
                        array_haiti_pchip,
                        array_dr_pchip,
                        array_observe_year,
                        array_pf_obs_haiti,
                        array_pf_obs_dr,
                        title=f'Anchor years: {array_anchor_year[0]}-{array_anchor_year[-1]}',
                        fig_size=(19.5, 14),
                        )









