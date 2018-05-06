import numpy as np
import matplotlib.pyplot as plt
import os

default_dir = 'plots'
default_color = 'g'
default_format = '.png'

# Sets the plot directory default
def set_plot_dir(dir):
    default_dir = dir

# Sets the color default
def set_color(color):
    default_color = color

# Sets the default file format
def set_file_format(format):
    default_format = format

# Checks if a dir exists and makes dir if not
def check_dir(dir):
    os.makedirs(dir, exist_ok=True)


# handles all other attributes of the plot that is equal throughout all plots
def plotfinisher(title="", xlabel="", ylabel="", savefig=True, figpath=default_dir, filename="hist_plot", format=default_format, grid=True, legend=True, show=True):

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid(True)
    if savefig: 
        check_dir(figpath)
        plt.savefig(figpath + '/' + filename + format, bbox_inches='tight')
    if show:
        plt.show()


# Plots histograms
def histogram(data, title="", xlabel="", ylabel="", label="Histogram", bins=None, savefig=True, figpath=default_dir, filename="hist_plot", format=default_format, grid=True, legend=True, clear=True, show=True, normalize=True, histtype="bar", color=default_color, stackableplot=False):

    if clear:
        plt.gcf().clear()

    if bins is not None:
        plt.hist(data, bins=bins, label=label, density=normalize)
    else:
        plt.hist(data, label=label, density=normalize)

    if not stackableplot:
        plotfinisher(title=title, xlabel=xlabel, ylabel=ylabel, savefig=savefig, figpath=figpath, filename=filename, format=format, grid=grid, legend=legend, show=show)


# Plots data as bar chart
def bars(data, x_ticks=["No", "Yes"], title="", xlabel="", ylabel="", label="Bar plot", savefig=True, figpath=default_dir, filename="bar_plot", format=default_format, grid=True, legend=True, clear=True, show=True, normalize=True, color=default_color, stackableplot=False):

    if clear:
        plt.gcf().clear()

    if normalize:
        datasize = len(data)
        data = np.divide(data,datasize)

    plt.bar(x_ticks, data, label=label)

    if not stackableplot:
        plotfinisher(title=title, xlabel=xlabel, ylabel=ylabel, savefig=savefig, figpath=figpath, filename=filename, format=format, grid=grid, legend=legend, show=show)


# Preprocesses the data for displaying boolean values as bar chart
def bool_bars(data, x_ticks=["No", "Yes"], title="", xlabel="", ylabel="", label="Bar plot", savefig=True, figpath=default_dir, filename="bar_plot", format=default_format, grid=True, legend=True, clear=True, show=True, normalize=True, color=default_color, stackableplot=False):

    data = np.array([len(data[data[:] == 0]), len(data[data[:] == 1])])

    bars(data=data, x_ticks=x_ticks, title=title, xlabel=xlabel, ylabel=ylabel, label=label, savefig=savefig, figpath=figpath, filename=filename, format=format, grid=grid, legend=legend, clear=clear, show=show, normalize=normalize, color=color, stackableplot=stackableplot)


# Lines
def lines(data, title="", xlabel="", ylabel="", label="Lines", bins=None, savefig=True, figpath=default_dir, filename="lines_plot", format=default_format, grid=True, legend=True, clear=True, show=True, normalize=True, color=default_color, stackableplot=False):

    if clear:
        plt.gcf().clear()

    plt.plot(data, label=label)
    
    if not stackableplot:
        plotfinisher(title=title, xlabel=xlabel, ylabel=ylabel, savefig=savefig, figpath=figpath, filename=filename, format=format, grid=grid, legend=legend, show=show)

