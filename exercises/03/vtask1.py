#!/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt


random_seed = 128
desired_list_size = 1000
desired_bin_numbers = 50
random.seed(random_seed)

def task1a():
    uniform_randoms = random.sample(range(0,1000), desired_list_size)
    max_value = max(uniform_randoms)
    bin_size = max_value/ float(desired_bin_numbers)
    a = np.hstack((uniform_randoms, list(np.arange(0, max_value, bin_size))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title("Histogram of uniformed random numbers")
    plt.savefig("hist_1a.png", bbox_inches='tight')
    print("Because the set is generated randomly, the dataset's distribution is balanced. That means every element occurs in an equal density.")

def task1b():
    mu, sigma = 0, 0.9# mean and standard deviation
    gauss_randoms = np.random.normal(mu, sigma, desired_list_size)
    if abs(mu - np.mean(gauss_randoms)) < sigma:
        plt.gcf().clear()
        count, bins, ignored = plt.hist(gauss_randoms, 30, normed=True)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=2, color='r')
        plt.savefig("hist_1b.png", bbox_inches='tight')

def task1c():
    n, p  = 20, 0.5
    binom_randoms = np.random.binomial(n, p, desired_list_size)
    max_value = max(binom_randoms)
    min_value = min(binom_randoms)
    bin_size = max_value/ float(desired_bin_numbers)
    a = np.hstack((binom_randoms, list(np.arange(min_value, max_value, bin_size))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title("Histogram of binomial distribution.")
    plt.savefig("hist_1c.png", bbox_inches='tight')

def plot_histogram(raw_list, name):
    max_value = max(raw_list)
    min_value = min(raw_list)
    bin_size = max_value/ float(desired_bin_numbers)
    a = np.hstack((raw_list, list(np.arange(min_value, max_value, bin_size))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title(name)
    plt.savefig("hist_1c.png", bbox_inches='tight')

task1a()
task1b()
task1c()
