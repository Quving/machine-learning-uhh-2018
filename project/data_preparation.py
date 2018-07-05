import numpy as np
import pandas as pd
import math
import os

# Plotter library
from ggplot import *

### Before everything else: Add extension to pandas DataFrame
### to accept deprecated sort() call from ggplot
import builtins
class DataFrameExtended(pd.DataFrame):
    def sort(self):
        if self:
            return pd.DataFrame.sort_values(self)
        else:
            return ''

builtins.DataFrame = DataFrameExtended
###

# Read the data

def read_data(filename, usecols=[]):

    df = pd.read_csv(filename, encoding = "ISO-8859-1", usecols=usecols)
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
df = read_data('globalterrorismdb_0617dist.csv', usecols=['nreleased','attacktype1','attacktype2','attacktype3','extended','iyear','gname','nhostkid','nhours','ndays','ransom','ransompaid','ransompaidus','ishostkid','hostkidoutcome'])

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
df['nreleased_p'] = np.divide(np.subtract(np.array(df.nhostkid),np.array(df.nreleased)),df.nhostkid)

### Data plots

# First: a plot about number of kidnapped persons
print (ggplot(
    df,
    aes(x = 'iyear', y = 'nhostkid')
) + geom_line() + stat_smooth(colour='blue', span=0.2))

###