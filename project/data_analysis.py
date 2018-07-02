import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotter as plotter
import scipy.stats as stats
import math

coef_filename = "table_coefs.md"
desc_filename = "table_desc_variables.md"

def read_data(filename):
    # Used Columns:
    # 0 EventId
    # 1 Year
    # 2 Month
    # 3 Day
    # 7 Country
    # 8 Country TXT
    # 12 City

    raw = pd.read_csv(filename, encoding = "ISO-8859-1", usecols=lambda x: x not in ['attacktype2','claims2','claimmode2','claimmode3','gname2'])

    # Make for each column a nparray

    data = {}

    numerical_columns = ['eventid','imonth','iday','latitude','longitude','nkill','nkillus','nkillter','nwound','nwoundus','nwoundte' ,'nperps','nhostkid','nhostkidus','nhours','nperpcap','propvalue', 'ransomamt','ransomamtus','ransompaid','ransompaidus','ndays','nreleased']

    num_of_columns = len(raw.columns.values)

    print('Starting data reading and data augmentation process...')

    for i, columnname in zip(range(0,num_of_columns),raw.columns.values):
        columndata = np.array(raw[columnname])

        is_categorial = columnname not in numerical_columns

        data[columnname] = {'data': columndata, 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': '', 'isCategorial': is_categorial, 'isBool':False}

        print('{}{} Data reading (key: {}, column {} of {})                '.format('█' * math.floor((i/num_of_columns)*20),'░' * math.ceil(20-(i/num_of_columns)*20),columnname,i, num_of_columns), end='\r')

        if is_categorial and columndata.dtype != 'object':
            indexes = np.sort(np.unique(columndata[~np.isnan(columndata)]))

            for index in indexes:
                print('{}{} Data augmentation (index {} for key: {}, column {} of {})                '.format('█' * math.floor((i/num_of_columns)*20),'░' * math.ceil(20-(i/num_of_columns)*20),index, columnname,i, num_of_columns), end="\r")

                ix_columndata = columndata == index
                data[columnname+'_'+str(index)] = {'data': ix_columndata, 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': '', 'isCategorial': True, 'isBool':True}


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

    tf = open(desc_filename, 'w')
    tf.write('The descriptive analysis of the data has been made by calculating the mean, median, variance and Shapiro p value for standard deviation test.'.format(alpha,alpha))
    tf.write('\n')
    tf.write('\n')
    tf.write('| id | columname | mean | median | variance | W | normality |')
    tf.write('\n')
    tf.write('|-|-|-|-|-|-|-|')
    tf.write('\n')


    print('Printing data table with values')
    print('| id | columname | mean | median | variance | W | normality |')
    print('|-|-|-|-|-|-|-|')

    desc_stats_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))
    lineplots_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))
    histplots_for = np.unique(np.vstack((pairwise_plot_first,pairwise_plot_second)))

    for key, datacolumn in data.items():
        column = datacolumn['data']
        if column.dtype != 'object':
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

            tf.write("| %s | %s | %0.2f | %0.2f | %0.2f | %0.2f | %0.2f |" % (col_id, key, mean, median, variance, shapiro_W, shapiro_p))
            tf.write('\n')

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

    tf.close()

    return data

def calculate_correlations(data, alpha=0.5):

    data_keys = list(data.keys())

    coef_matrix = np.zeros((len(data_keys),len(data_keys)))


    print('\n')
    print('| Column 1 | Column 2 | Coefficient |')
    print('|-|-|-|')

    tf = open(coef_filename, 'w')
    tf.write('The calculation of the correlation coefficiences is computed with the Pearson correlation coefficient. The results were filtered by an alpha of {}/-{} and are listed below.'.format(alpha,alpha))
    tf.write('\n')
    tf.write('\n')
    tf.write('| Column 1 | Column 2 | Coefficient |')
    tf.write('\n')
    tf.write('|-|-|-|')
    tf.write('\n')

    correlating_pairs = []

    for key, datacolumn in data.items():

        datacolumn = datacolumn['data']

        if datacolumn.dtype == 'object':
            # print('skipping {}'.format(key))
            continue

        for key2_id in range(data_keys.index(key)+1,len(data_keys)):

            key2 = data_keys[key2_id]
            print('Calculate correlation for {} vs. {}...                    '.format(key, key2),end="\r")

            datacolumn2 = data[key2]['data']

            if datacolumn2.dtype == 'object':
                # print('skipping {}'.format(key2))
                continue

            datacolumn = fill_nan_cells(datacolumn)
            datacolumn2 = fill_nan_cells(datacolumn2)


            corrcoef = np.corrcoef(datacolumn,datacolumn2)

            if data[key]['isCategorial'] and data[key2]['isCategorial']:
                corrcoef = np.corrcoef(datacolumn,datacolumn2)[0][1]
            elif not data[key]['isCategorial'] and data[key2]['isCategorial']:
                corrcoef = np.corrcoef(datacolumn,datacolumn2)[0][1]
            elif data[key]['isCategorial'] and not data[key2]['isCategorial']:
                corrcoef = np.corrcoef(datacolumn,datacolumn2)[0][1]
            elif not data[key]['isCategorial'] and not data[key2]['isCategorial']:
                corrcoef = np.corrcoef(datacolumn,datacolumn2)[0][1]

            coef_matrix[data_keys.index(key)][data_keys.index(key2)] = corrcoef
            print('                                                                           ',end="\r")
            if abs(corrcoef) > alpha:
                correlating_pairs.append([key, key2])
                tf.write('|%s|%s|%0.2f|' % (key, key2, corrcoef))
                tf.write('\n')
                print('|%s|%s|%0.2f|' % (key, key2, corrcoef))

    tf.close()
    plt.gcf().clear()

    plt.imshow(coef_matrix)
    plt.colorbar()

    line_x = range(0,len(data_keys))
    line_y = range(0,len(data_keys))
    plt.xticks(line_x, data_keys, rotation='vertical')
    plt.yticks(line_y, data_keys)

    plt.plot(line_x,line_y, 'ro', c="r", label="identity line")
    plotter.plotfinisher("Corelation coefficiencies of each column", savefig=True, legend=True, figpath='plots', filename='correlations', show=True)


    return coef_matrix, correlating_pairs



filename = "globalterrorismdb_0617dist.csv"

data = read_data(filename)

# Add natlty_differ for hypothesis
natlty_differ = np.array(data['natlty1']['data'] != data['country']['data'], dtype=int)
data['natlty_differ'] = {'data': natlty_differ, 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': '', 'isCategorial':True, 'isBool': True}

# Add an index list for gname
gname_index_dict = make_key_id_dict(data['gname']['data'])
gname_ids = [gname_index_dict[x] for x in data['gname']['data']]
data['gname_ids'] = {'data': np.array(gname_ids), 'labels': [], 'mean': '', 'median': '', 'variance' : '', 'shapiro_W': '', 'shapiro_p': '',  'isCategorial':True, 'isBool': False}

_, correlating_pairs = calculate_correlations(data, alpha=0.5)

data = descriptive_analysis(data, plot=False, pairwise_plot=correlating_pairs)



## Supporess warinings
import warnings
warnings.filterwarnings("ignore")