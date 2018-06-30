import copy
import json
import math
import os
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class GlobalTerrorismDBParser():
    # Reads in the csv file and stores it as DataFrame object.
    def __init__(self):
        self.data_dir = "data"
        self.plot_dir = "plots"
        self.data_filename = "globalterrorismdb_0617dist.csv"
        self.data_path = os.path.join(self.data_dir, self.data_filename)

        self.data = pd.read_csv(self.data_path,
                                encoding="latin1",
                                low_memory=False)

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Must be instance of DataFrame, but got " + type(self.data) + ".")

    # Stores a json with the content of each column in the csv file.
    def to_json(self):
        for column in self.data.columns:
            with open(os.path.join(self.data_dir, column + ".json"), "w") as f:
                json.dump(obj=list(self.data.get(column)), fp=f, sort_keys=True, indent=4)

    # Returns from the data-set a column as list.
    def __get_column(self, column):
        return list(self.data.get(column))

    def plot_histogram_for_column(self, column_name, bins, xlabel, ylabel):
        column = self.__get_column(column_name)
        plt.gcf().clear()
        n, bins, patches = plt.hist(column, bins=bins, density=True, facecolor='g', alpha=0.75)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Histogram for column '" + column_name + "'")
        plt.savefig(os.path.join(self.plot_dir, column_name + "_histogram.png"), dpi='figure', bbox_inches='tight')

    def plot_geographical_heatmap(self, filename):

        # Preprocessing
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        lg = copy.deepcopy(self.__get_column(column="longitude"))
        lat = copy.deepcopy(self.__get_column(column="latitude"))

        idxs = list()
        for val, (a, b) in enumerate(zip(lg, lat)):
            if math.isnan(a) or math.isnan(b):
                idxs.append(val)
        incr = 0
        for idx in idxs:
            del lg[idx - incr]
            del lat[idx - incr]
            incr += 1

        # Plotting
        p1 = Process(target=self.plot_heatmap_1,
                     args=[lg, lat, "Geographical heatmap of terrorism attacks", filename, self.plot_dir,
                           "Longitude", "Latitude"])

        p1.start()
        p1.join()

    def plot_heatmap_1(self, lg, lat, title, filename, plot_dir, xlabel, ylabel):
        lg_lat = np.vstack([lg, lat])
        z = gaussian_kde(lg_lat)(lg_lat)

        plt.gcf().clear()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.scatter(lg, lat, c=z, s=2.5, alpha=1)
        plt.savefig(os.path.join(plot_dir, filename), dpi='figure', bbox_inches='tight')
        plt.show()

    def plot_heatmap_2(self, lg, lat, title, filename, plot_dir):
        plt.gcf().clear()
        plt.hist2d(x=lg,
                   y=lat,
                   bins=[50, 50],
                   cmap=plt.cm.jet,
                   alpha=1)
        plt.colorbar()
        plt.ylabel('latitude')
        plt.xlabel('longitude')
        plt.title(title)
        plt.savefig(os.path.join(plot_dir, filename),
                    transparent=False,
                    dpi='figure',
                    bbox_inches='tight')
