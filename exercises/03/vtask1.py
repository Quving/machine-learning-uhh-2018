#!/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt


random_seed = 128
desired_list_size = 1000
desired_bin_numbers = 50
random.seed(random_seed)

def task1a():
    min_rand, max_rand = 0, 1000
    uniform_randoms = random.sample(range(min_rand, max_rand), desired_list_size)
    plot_histogram(
            raw_list = uniform_randoms,
            plot_title= "Histogram of normal random distribution.",
            save_as = "1a.png")
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
        plt.savefig("1b.png", bbox_inches='tight')

def task1c():
    n, p  = 20, 0.5
    binom_randoms = np.random.binomial(n, p, desired_list_size)
    plot_histogram(
            raw_list=binom_randoms,
            plot_title="Histogram of binomial distribution.",
            save_as="1c.png")

def task1d():
    m = [2, 3, 5, 10, 20]
    min_rand, max_rand = 0, 1000
    for current_m in m:
        summed_randoms = []
        for random_sum in range(0, desired_list_size):
            summed_element = sum(random.sample(range(min_rand, max_rand), current_m))
            summed_randoms.append(summed_element)
        plot_histogram(
                raw_list=summed_randoms,
                plot_title="Histogram of summed randoms distribution. m = "+ str(current_m),
                save_as="1d_m"+ str(current_m) +".png")

def plot_histogram(raw_list, plot_title, save_as):
    max_value = max(raw_list)
    min_value = min(raw_list)
    bin_size = max_value/ float(desired_bin_numbers)
    a = np.hstack((raw_list, list(np.arange(min_value, max_value, bin_size))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title(plot_title)
    plt.savefig(save_as, bbox_inches='tight')
# task1a()
# task1b()
# task1c()
task1d()
