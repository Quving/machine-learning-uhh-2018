import copy
import os
from multiprocessing import Process

from globalterrorismdb_parser import GlobalTerrorismDBParser
from pyhelpers.cleaner import Cleaner
from pyhelpers.plotter import Plotter


class Computer:
    def __init__(self):
        self.gt_parser = GlobalTerrorismDBParser()
        self.plot_dir = "plots"

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

        lg, lat = Cleaner.eliminate_nans(lg, lat)

        # Plotting
        title = "Geographical heatmap of terrorism attacks"
        xlabel = "Longitude"
        ylabel = "Latitude"
        p1 = Process(target=Plotter.plot_heatmap_1,
                     args=[lg, lat, title, os.path.join(self.plot_dir, filename), xlabel, ylabel])
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

        title = "Histogram for column '" + column_name + "'"
        filename = os.path.join(self.plot_dir, column_name + "_histogram.png")
        column = self.gt_parser.get_column(column_name)

        Plotter.plot_histogram_for_column(xs=column, title=title, filename=filename, bins=bins, xlabel=xlabel,
                                          ylabel=ylabel, info_threshold=info_threshold,
                                          textbox_x_positional_percentage=textbox_x_positional_percentage,
                                          textbox_drop_percentage=textbox_drop_percentage)
