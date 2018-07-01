import copy
import os
from multiprocessing import Process
import numpy as np
from globalterrorismdb_parser import GlobalTerrorismDBParser
from pyhelpers.cleaner import Cleaner
from pyhelpers.plotter import Plotter
from sklearn.neighbors import KNeighborsClassifier


class Computer:
    def __init__(self):
        self.gt_parser = GlobalTerrorismDBParser()
        self.plot_dir = "plots"

    def plot_geographical_heatmap(self, filename, heatmap):
        """
        Plots the longitude and latitude of terrorism attacks to png.
        :param filename:
        :return:
        """

        # Pre-processing
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        x = copy.deepcopy(self.gt_parser.get_column(column="longitude"))
        y = copy.deepcopy(self.gt_parser.get_column(column="latitude"))

        x, y = Cleaner.eliminate_nans(x, y)

        # Plotting
        title = "Geographical heatmap of terrorism attacks"
        xlabel = "Longitude"
        ylabel = "Latitude"

        if heatmap:
            Plotter.plot_heatmap_1(x, y,
                                   title=title,
                                   filename=os.path.join(self.plot_dir, "heatmap_"+filename),
                                   xlabel=xlabel,
                                   ylabel=ylabel)
        else:
            Plotter.plot_scatter(x, y,
                                 title=title,
                                 filename=os.path.join(self.plot_dir, "scatter_" + filename),
                                 xlabel=xlabel,
                                 ylabel=ylabel)

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

        title = "Histogram for column '" + column_name + "'"
        filename = os.path.join(self.plot_dir, column_name + "_histogram.png")
        column = self.gt_parser.get_column(column_name)

        Plotter.plot_histogram_for_column(xs=column, title=title, filename=filename, bins=bins, xlabel=xlabel,
                                          ylabel=ylabel, info_threshold=info_threshold,
                                          textbox_x_positional_percentage=textbox_x_positional_percentage,
                                          textbox_drop_percentage=textbox_drop_percentage)

    def compute_knn_on_location(self):
        lg = copy.deepcopy(self.gt_parser.get_column(column="longitude"))
        lat = copy.deepcopy(self.gt_parser.get_column(column="latitude"))

        lg, lat = Cleaner.eliminate_nans(lg, lat)
        X = [[float(i), float(i * 2)] for i in lg]
        y = [int(i) for i in lat]
        neigh = KNeighborsClassifier()
        neigh.fit(X, y)

        return neigh