#!/usr/bin/python3

import numpy as np
import random
import matplotlib.pyplot as plt


random_seed = 128
desired_list_size = 1000
random.seed(random_seed)

def task1a():
    uniform_randoms = random.sample(range(0,1000), desired_list_size)
    print (uniform_randoms)

    a = np.hstack((uniform_randoms, list(np.arange(0, 1000, 10))))
    plt.gcf().clear()
    plt.hist(a, bins='auto')
    plt.title("Histogram of uniformed random numbers")
    plt.savefig("hist_1a.png", bbox_inches='tight')
    print("Because the set is generated randomly, the dataset's distribution is balanced. That means every element occurs in an equal density.")


def task1b():
    mu, sigma = 0, 0.9# mean and standard deviation
    gauss_randoms = np.random.normal(mu, sigma, 1000)
    print (gauss_randoms)
    if abs(mu - np.mean(gauss_randoms)) < sigma:
        plt.gcf().clear()
        count, bins, ignored = plt.hist(gauss_randoms, 30, normed=True)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                linewidth=2, color='r')
        plt.show()

task1a()
task1b()
