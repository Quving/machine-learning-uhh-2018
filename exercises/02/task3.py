#!/usr/bin/python3
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import time
import os
from scipy.stats import gaussian_kde

## TASK 3a + 3b +3c #######################################################

filename = "housing.csv"
bin_total = 50.0
data_props = {}
data = np.genfromtxt(filename,
        skip_header=1,
        usecols=list(range(0,8)),
        dtype=None,
        delimiter=',')
directory = "histograms"
if not os.path.exists(directory):
    os.makedirs(directory)
# Because np. cannot parse strings, I use csv.reader to get the column names.
with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = list(reader)[0]

for header,column in zip(headers,list(zip(*data))):
    column = list(column)

    # Remove nans.
    column = [x for x in column if str(x) != 'nan']
    min_val = min(column)
    max_val = max(column)
    min_idx = column.index(min_val)
    max_idx = column.index(max_val)
    mean = np.mean(column)
    bin_size = np.abs(max_val - min_val) / bin_total

    print(header, "\n\tmin:", min_val , "\t( index:", min_idx,")")
    print("\tmax:", max_val, "\t( index:", max_idx,")", "\n\tmean", mean)
    print("\tbin_size", bin_size)

    # Create histograms.
    a = np.hstack((column, list(np.arange(min_val, max_val, bin_size))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title("Histogram for column '" + header+ "'")
    plt.savefig(directory + "/" + header + ".png", bbox_inches='tight')


response3d = "Following categories would fit into a normal distribution: 'households', 'median_income', 'population', 'total_bedrooms', 'total_rooms'.  The distribution of 'housing_median_age' isn't clearly a normal distribution since the high amount of outlyers. The columns of 'latitude' and 'longitude' don't fit into the shape of a  normal distribution."
print("\n",response3d)


## TASK 3d #######################################################

plt.gcf().clear()
lg = list(list(zip(*data))[0])
lat = list(list(zip(*data))[1])
lg_lat = np.vstack([lg,lat])
z = gaussian_kde(lg_lat)(lg_lat)
plt.gcf().clear()
plt.scatter(lg, lat, c=z, s=2.5, alpha=1)
plt.savefig(directory + "/geographic_map_1" + ".png", bbox_inches='tight')

plt.gcf().clear()
plt.hist2d(lg, lat, (100,100), cmap=plt.cm.jet, alpha=1)
plt.colorbar()
plt.savefig(directory + "/geographic_map_2" + ".png", bbox_inches='tight')
