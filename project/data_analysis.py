import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotter as plotter
import scipy.stats as stats



def read_data(filename):
    # Used Columns:
    # 0 EventId
    # 1 Year
    # 2 Month
    # 3 Day
    # 7 Country
    # 8 Country TXT
    # 12 City

    raw = pd.read_csv(filename, encoding = "ISO-8859-1")

    # Make for each column a nparray

    data = {}

    for columnname in raw.columns.values:
        columndata = np.array(raw[columnname])
        data[columnname] = {'data': columndata, 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': ''}

    return data

def fill_nan_cells(ndarray):

    ndarray[np.isnan(ndarray)] = np.mean(ndarray[~np.isnan(ndarray)])

    return ndarray

def make_index_list(string_array):

    strings = np.unique(string_array)
    ids = np.range(0,len(strings))

    return zip(strings, ids)

def make_index_dict(string_array):

    strings = np.unique(string_array)

    return dict(enumerate(strings))

def make_key_id_dict(string_array):

    if string_array is None:
        return {None: None}

    strings = np.unique(string_array)
    ids = range(0, len(strings))

    return {key: index for key, index in zip(strings, ids)}

def descriptive_analysis(data, plot=True, pairwise_plot=False):

    data_keys = list(data.keys())
    # Descriptive statistics

    pairwise_plot_first = []
    pairwise_plot_second = []

    if pairwise_plot is not False:
        pairwise_plot_first, pairwise_plot_second = zip(*pairwise_plot)

    print('Printing data table with values')
    print('| id | columname | mean | median | variance | W | normality |')
    print('|-|-|-|-|-|-|-|')

    desc_stats_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))
    lineplots_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))
    histplots_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))

    for key, datacolumn in data.items():
        column = datacolumn['data']
        if column.dtype != 'object':
            if key in desc_stats_for:
                column_wo_nans = column[~np.isnan(column)]
                col_id = data_keys.index(key)
                mean = np.mean(column_wo_nans)
                median = np.median(column_wo_nans)
                variance = np.var(column_wo_nans)
                shapiro_W, shapiro_p = stats.shapiro(column_wo_nans)

                datacolumn['mean'] = mean
                datacolumn['median'] = median
                datacolumn['variance'] = variance
                datacolumn['shapiro_W'] = shapiro_W
                datacolumn['shapiro_p'] = shapiro_p

                # print('{}: mean: {}, median: {}, variance: {}'.format(key, mean, median, variance))
                print("| %s | %s | %0.2f | %0.2f | %0.2f | %0.2f | %0.2f |" % (col_id, key, mean, median, variance, shapiro_W, shapiro_p))

            if key in lineplots_for and plot:
                plotter.lines(column, title=key, label=key,figpath='plots',show=False,filename=key+'_lines')

            if key in histplots_for and plot:
                plotter.histogram(column[~np.isnan(column)], title=key, label=key,figpath='plots',show=False,filename=key+'_hist', normalize=False)

            # Descriptive stats II: plottings against two value

            if plot and key in pairwise_plot_first:

                key2 = pairwise_plot_second[pairwise_plot_first.index(key)]

                column2 = data[key2]['data']

                if column2.dtype != 'object':

                    # plotter.lines(column2, column, title=key+' '+key2, label=key + ' ' + key2,figpath='plots',show=False,filename=key+'_'+key2+'_lines')
                    # plotter.scatter(column, column2, title=key+' '+key2, label=key + ' ' + key2,figpath='plots',show=False,filename=key+'_'+key2+'_scatter')

                    plt.gcf().clear()
                    plt.scatter(column, column2, label=key + ' ' + key2)
                    # plt.plot(column, column2, label=key + ' ' + key2, c="r")

                    plotter.plotfinisher("Plot " + key + " vs. " + key2, savefig=True, legend=True, figpath='plots', filename=key+'_'+key2, show=False)


                    # resolve nan values by inserting their mean
                    column = fill_nan_cells(column)
                    column2 = fill_nan_cells(column2)


                    plotter.heatmap(column, column2, title=key+' '+key2, label=key + ' ' + key2,figpath='plots',show=False,filename=key+'_'+key2+'_heatmap', bins=(30,30), legend=False)

                    #plotter.scatterheat(column, column2, title=key+' '+key2, label=key + ' ' + key2,figpath='plots',show=False,filename=key+'_'+key2+'_scatterheat')

    return data

def calculate_correlations(data, alpha=0.5):

    data_keys = list(data.keys())

    coef_matrix = np.zeros((len(data_keys),len(data_keys)))

    print('| Column 1 | Column 2 | Coefficient |')
    print('|-|-|-|')

    correlating_pairs = []

    for key, datacolumn in data.items():

        datacolumn = datacolumn['data']

        if datacolumn.dtype == 'object':
            # print('skipping {}'.format(key))
            continue

        for key2_id in range(data_keys.index(key)+1,len(data_keys)):

            key2 = data_keys[key2_id]
            datacolumn2 = data[key2]['data']

            if datacolumn2.dtype == 'object':
                # print('skipping {}'.format(key2))
                continue

            datacolumn = fill_nan_cells(datacolumn)
            datacolumn2 = fill_nan_cells(datacolumn2)


            corrcoef = np.corrcoef(datacolumn,datacolumn2)

            coef_XY = corrcoef[0][1]
            coef_YX = corrcoef[1][0]

            coef_matrix[data_keys.index(key)][data_keys.index(key2)] = coef_XY
            coef_matrix[data_keys.index(key2)][data_keys.index(key)] = coef_YX

            if abs(coef_XY) > alpha:
                correlating_pairs.append([key, key2])
                print('|%s|%s|%0.2f|' % (key, key2, coef_XY))

    plt.gcf().clear()

    plt.imshow(coef_matrix)
    plt.colorbar()

    line_x = range(0,len(data_keys))
    line_y = range(0,len(data_keys))
    plt.xticks(line_x, data_keys, rotation='vertical')
    plt.yticks(line_y, data_keys)

    plt.plot(line_x,line_y, 'ro', c="r", label="identity line")
    plotter.plotfinisher("Corelation coefficiencies of each column", savefig=True, legend=True, figpath='plots', filename='correlations', show=False)


    return coef_matrix, correlating_pairs



filename = "globalterrorismdb_0617dist.csv"

data = read_data(filename)

# Add natlty_differ for hypothesis
natlty_differ = np.array(data['natlty1']['data'] != data['country']['data'], dtype=int)
data['natlty_differ'] = {'data': natlty_differ, 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': ''}

# Add an index list for gname
gname_index_dict = make_key_id_dict(data['gname']['data'])
gname_ids = [gname_index_dict[x] for x in data['gname']['data']]
data['gname_ids'] = {'data': np.array(gname_ids), 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': ''}

_, correlating_pairs = calculate_correlations(data, alpha=0.5)

data = descriptive_analysis(data, plot=True, pairwise_plot=correlating_pairs)



