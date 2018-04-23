#!/usr/bin/python3
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import time
import os
from scipy.stats import gaussian_kde
import random

filename = "housing.csv"
hist_dir = "histograms"
gmap_dir = "geographic_maps"
bin_total = 50
random_seed = 128
data = np.genfromtxt(filename,
        skip_header=1,
        usecols=list(range(0,8)),
        dtype=None,
        delimiter=',')


lg = list(list(zip(*data))[0])
lat = list(list(zip(*data))[1])

response3c = "Task 3c: \tFollowing categories would fit into a normal distribution: 'households', 'median_income', 'population', 'total_bedrooms', 'total_rooms'.  The distribution of 'housing_median_age' isn't clearly a normal distribution since the high amount of outlyers. The columns of 'latitude' and 'longitude' don't fit into the shape of a  normal distribution."

response3e = "Task 3e: \tSince the test_set and the training_set is randomly taken from the original data-set. The distribution should be the same; not absolutely but relatively to eachother (expectancy value). Hence, the max and min and also the mean shouldn't differ that much in comparison to the original data-set. The training-set's max,min and mean should be closer to the ones of the original set, because the sample is drawn bigger than the test-set."


for directory in [hist_dir, gmap_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

## TASK 3a + 3b +3c #######################################################
# Because np. cannot parse strings, I use csv.reader to get the column names.
with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = list(reader)[0]

def task3abc(headers, data, generateHist=False):
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
        if generateHist:
            a = np.hstack((column, list(np.arange(min_val, max_val, bin_size))))
            plt.gcf().clear()
            plt.hist(a, bins='auto')
            plt.title("Histogram for column '" + header+ "'")
            plt.savefig(hist_dir + "/" + header + ".png", bbox_inches='tight')


## TASK 3d #######################################################

def plot_geographic_map_1(lg, lat):
    print("\n>> Task3d: Plot plot_geographic_map_1")
    plt.gcf().clear()
    lg_lat = np.vstack([lg,lat])
    z = gaussian_kde(lg_lat)(lg_lat)
    plt.title("Geographic map 1")
    plt.scatter(lg, lat, c=z, s=2.5, alpha=1)
    plt.savefig(gmap_dir + "/geographic_map_1" + ".png", bbox_inches='tight')

def plot_geographic_map_2(lg, lat):
    print("\n>> Task3d: Plot plot_geographic_map_2")
    plt.gcf().clear()
    plt.hist2d(lg, lat, (100,100), cmap=plt.cm.jet, alpha=1)
    plt.colorbar()
    plt.title("Geographic map 2")
    plt.savefig(gmap_dir+"/geographic_map_2" + ".png", bbox_inches='tight')



## TASK 3e ########################################################

def split_dataset(data):
    random.seed(random_seed)
    test_set = []
    training_set = []
    for column in data:
        if random.randint(1,10) <=2:
            test_set.append(column)
        else:
            training_set.append(column)
    print("\n>> Print max, min, mean for test-set. ")
    print("===================================================================================================================================")
    task3abc(headers, test_set)
    print("\n>> Print max, min, mean for training-set.")
    print("===================================================================================================================================")
    task3abc(headers, training_set)


if __name__ == "__main__":
    print("\n>> Print max, min, mean for original data-set.")
    print("===================================================================================================================================")
    task3abc(headers, data, generateHist=True) # task abc
    print("\n",response3c) # task c (written answer)
    plot_geographic_map_1(lg, lat) # task d version 1
    plot_geographic_map_2(lg, lat) # task d version 2
    split_dataset(data) # task e
    print("\n", response3e) # task e (written answer)


