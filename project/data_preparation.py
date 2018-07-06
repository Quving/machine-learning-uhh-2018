import numpy as np
import pandas as pd
import math
import os

# Plotter library
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data

def read_data(filename, usecols=[]):

    df = pd.read_csv(filename, encoding = "ISO-8859-1")
    print('Finished reading csv file. Dimensions: {}'.format(df.shape))

    return df

def split_categories(df, categories):

    for category in categories:
        df = df[category].get_dummies()

def str_to_index_arr(arr):

    '''
    Takes an unordered list of strings, orders them, assigns an unique ID to every string and builds an equally long list of indexes respectively to the strings.
    '''
    strings = np.sort(np.unique(arr))
    ids = range(0, len(strings))

    id_dict = {key: index for key, index in zip(strings, ids)}
    id_arr = [id_dict[string] for string in arr]

    return id_arr, id_dict

def make_bool_arr(arr, conditions):

    return np.array(arr) == conditions

#####
# The purpose of our classifier is to predict the hostkidoutcome category and a percentage of released persons.
# Y: hostkidoutcome, npreleased
# X: extended, iyear, gname_id, nhostkid, ndays, ransom, ransompaid, ishostkid
#####

### Data filtering

# Read data and exclude cols
# @Snippet: To exclude: lambda x: x not in ["eventid","imonth","iday", "attacktype2","claims2","claimmode2","claimmode3","gname2"]
df = read_data('globalterrorismdb_0617dist.csv')

# Filter for only kidnapping data (1st, 2nd or 3rd attack type)
kidnap_cats = [5,6]
df = df[df.attacktype1.isin(kidnap_cats) | df.attacktype2.isin(kidnap_cats) | df.attacktype3.isin(kidnap_cats)]

# Filter also broken data from our classes
df = df[df.hostkidoutcome.notnull()]

# Filter data for NaN nreleased or value -99
df = df[df.nreleased.notnull()]
df = df[df.nreleased != -99]

# Filter also data where nhostkid is lower than nreleased
df = df[df.nhostkid >= df.nreleased ]

### Data augmentation

# Add an ID group for gname to the DataFrame
df['gname_id'],_ = str_to_index_arr(df['gname'])

# Add a normalisation for how many of the hostage victims survived
df['nreleased_p'] = np.divide(df.nreleased,df.nhostkid)

### Data plots

# First: a plot about number of kidnapped persons
sns.set(style="darkgrid", color_codes=True)

g1 = sns.jointplot(
    'iyear',
    'nhostkid',
    data = df,
    kind = "reg",
    color = 'r',
    size = 7,
    xlim = [1970,2016]
)
g1.set_axis_labels('Years','Number of kidnapped victims')

plt.show()
plt.gcf().clear()

# Outcomes vs percentage of released victims

g2 = sns.violinplot(
    x = 'hostkidoutcome',
    y = 'nreleased_p',
    data = df,
    hue = 'ransom'
)

plt.show()
plt.gcf().clear()

#### Correlation over all columns

corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=.3,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5}
)


plt.show()
plt.gcf().clear()

### Separate set into train, validation, test by assigning each to the preferred class randomly.
train = df.sample(frac=0.6, replace=True)
validation = df.sample(frac=0.2, replace=True)
test = df.sample(frac=0.2, replace=True)

def separate_labels(df, labels):
    '''
    Separates the labels columns from the dataframe and returns the dataframe and the labels
    '''

    return df.drop(labels, axis = 1), df[labels]

labels = ['hostkidoutcome','nreleased_p']
train, train_labels = separate_labels(train, labels)
validation, validation_labels = separate_labels(validation, labels)
test, test_labels = separate_labels(test, labels)

########

Training