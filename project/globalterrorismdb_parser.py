import copy
import json
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from multiprocessing import Process


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

    def __get_column(self, column):
        return list(self.data.get(column))

    def plot_geographical_heatmap(self):
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        lg = copy.deepcopy(self.__get_column(column="longitude"))
        lat = copy.deepcopy(self.__get_column(column="latitude"))

        ## clean NaN's
        idxs = list()
        for val, (a, b) in enumerate(zip(lg, lat)):
            if math.isnan(a) or math.isnan(b):
                idxs.append(val)
        incr = 0
        for idx in idxs:
            del lg[idx - incr]
            del lat[idx - incr]
            incr += 1
        p1 = Process(target=plot_heatmap_1,
                     args=[lg, lat, "Geographical heatmap of terrorism attacks", "geo_heatmap_1.png", self.plot_dir])

        p2 = Process(target=plot_heatmap_2,
                     args=[lg, lat, "Geographical heatmap of terrorism attacks", "geo_heatmap_2.png", self.plot_dir])
        p1.start()
        p2.start()

        p1.join()
        p2.join()


def plot_heatmap_1(lg, lat, title, filename, plot_dir):
    plt.gcf().clear()
    lg_lat = np.vstack([lg, lat])
    z = gaussian_kde(lg_lat)(lg_lat)
    plt.title(title)
    plt.scatter(lg, lat, c=z, s=2.5, alpha=1)
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.show()


def plot_heatmap_2(lg, lat, title, filename, plot_dir):
    plt.gcf().clear()
    plt.hist2d(lg, lat, (150, 150), cmap=plt.cm.jet, alpha=1)
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.show()
