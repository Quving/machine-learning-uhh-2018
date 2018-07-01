import copy
import math
import os
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from globalterrorismdb_parser import GlobalTerrorismDBParser


class Computer:
    def __init__(self):
        self.gt_parser = GlobalTerrorismDBParser()
        self.plot_dir = "plots"
        self.font = {'family': 'serif',
                     'color': 'darkred',
                     'weight': 'normal',
                     'size': 16,
                     }

    def plot_geographical_heatmap(self, filename):
        """
        Plots the longitude and latitude of terrorism attacks to png.
        :param filename:
        :return:
        """

        # Pre-processing
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        lg = copy.deepcopy(self.gt_parser.get_column(column="longitude"))
        lat = copy.deepcopy(self.gt_parser.get_column(column="latitude"))

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
                     args=[lg, lat, "Geographical heatmap of terrorism attacks",
                           filename, self.plot_dir, "Longitude", "Latitude"])
        p1.start()
        p1.join()

    def plot_histogram_for_column(self, column_name, bins, xlabel, ylabel, info_threshold,
                                  textbox_x_positional_percentage=0.75,
                                  textbox_drop_percentage=0.05):
        """
        Plots a histogram of a given column_column_name.
        :param column_name:
        :param bins:
        :param xlabel:
        :param ylabel:
        :param info_threshold:
        :return:
        """

        column = self.gt_parser.get_column(column_name)
        bin_size = (max(column) + math.fabs(min(column))) / bins
        hist, bins_1 = np.histogram(column,
                                    bins=np.arange(min(column), max(column), bin_size))
        max_n_elements = sorted(hist, reverse=True)[:info_threshold]
        distribution = np.where(hist >= min(max_n_elements))[0]
        interests = [hist[i] for i in distribution]
        distribution[:] = list(map(lambda x: float(x * bin_size), distribution))

        plt.gcf().clear()
        for x, y in zip(distribution, interests):
            plt.annotate("x: " + str(x) + "-" + str(int(x + bin_size)) + " \ny: " + str(y),
                         xy=(x, y),
                         xytext=(textbox_x_positional_percentage * (max(column) + math.fabs(min(column))),
                                 y - float(textbox_drop_percentage * max(max_n_elements))),
                         style='italic',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2.5})

        n, bins_2, patches = plt.hist(column,
                                      bins=bins,
                                      facecolor='g',
                                      alpha=0.75)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Histogram for column '" + column_name + "'", fontdict=self.font)
        plt.savefig(os.path.join(self.plot_dir, column_name + "_histogram.png"),
                    dpi='figure',
                    bbox_inches='tight')

    def plot_heatmap_1(self, lg, lat, title, filename, plot_dir, xlabel, ylabel):
        """
        Plots heatmap using gaussian colors.
        :param lg:
        :param lat:
        :param title:
        :param filename:
        :param plot_dir:
        :param xlabel:
        :param ylabel:
        :return:
        """
        lg_lat = np.vstack([lg, lat])
        z = gaussian_kde(lg_lat)(lg_lat)

        plt.gcf().clear()
        plt.title(title, fontdict=self.font)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.scatter(lg, lat, c=z, s=2.5, alpha=1)
        plt.savefig(os.path.join(plot_dir, filename),
                    dpi='figure',
                    bbox_inches='tight')
        plt.show()

    def plot_heatmap_2(self, lg, lat, title, filename, plot_dir):
        """
         Plot heatmap using default matplotlib functions.
        :param lg:
        :param lat:
        :param title:
        :param filename:
        :param plot_dir:
        :return:
        """
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
