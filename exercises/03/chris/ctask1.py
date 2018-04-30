import numpy as np
import math
import matplotlib.pyplot as plt
import resource
import time

# Create arrays of n random numbers

# a. Read the documentation for the numpy.random functions.
# DONE: https://docs.scipy.org/doc/numpy/reference/routines.random.html
# Interesting: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html

def make_hist(data, show=True):

    plt.hist(data, bins = 10)
    plt.title("Random number histogram")
    plt.xlabel("Bins")
    plt.ylabel("Value")
    plt.grid(True)

    plt.show()


# a. Create arrays of n ∈ [100, 1000, 10000, 100 000] random numbers with uniform distribution.

def task1a():
    hundreds = np.random.rand(100,)
    thousands = np.random.rand(1000,)
    tenthousands = np.random.rand(10000,)
    hundredthousands = np.random.rand(100000,)

    # Plot the raw data, then generate and plot histograms with 10 bins.
    print(hundreds)
    print("Number of digits: ", len(hundreds))
    make_hist(hundreds)
    make_hist(thousands)
    make_hist(tenthousands)
    make_hist(hundredthousands)

    # How do the mean, minimum and maximum values of the bins (occupation counts) behave?
    print('Q: How do the mean, minimum and maximum values of the bins (occupation counts) behave?')
    print('A: They are vary around, but getting more similar by increased number of random numbers')

####

# b. Create random numbers from a Gaussian distribution with mean μ and variance σ2.
def task1b():
    mean, variance, size = 0, 0.1, 1000

    # Make random data
    gauss_random = np.random.normal(mean, variance, size)

    # Plot data
    count, bins, ignored = plt.hist(gauss_random, 30, normed=True)
    plt.plot(bins, 1/(variance * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * variance**2) ), linewidth=2, color='r')
    
    # Show data
    print(gauss_random)
    plt.show()

# c. As before, but using the Binomial distribution with parameters n and p.
def task1c():
    rounds, propability = 1000, 0.5

    # Make random data
    binomial_random = np.random.binomial(rounds, propability)

    # Plot data
    count, bins, ignored = plt.hist(binomial_random, 30, normed=True)
    
    # Show data
    print(binomial_random)
    plt.show()

def task1d():

    # Dev: Start timer
    starttime = time.time()

    # Settings
    number_of_rand_numbers = [2,3,5,10,20]

    # open plot
    plt.figure(1)

    # Iterator
    iterator = 1

    for i in number_of_rand_numbers:

        result = None

        for j in range(i-1):
            if result is None:
                result = np.random.random(1000,)
            else: 
                result = result + np.random.random(1000,)

        plt.hist(result)
        #plt.scatter(i, result)
        plt.subplot(2,3, iterator)

        iterator += 1

    plt.tight_layout()

    endtime = time.time() - starttime
    print("Computing time: {}s, used resources {} mb".format(endtime, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000))
    plt.show()


    # Für jedes m aus M
    # Erzeuge m randoms und summiere sie auf.
    # Wiederhole Schritt 2 n mal und speichere die Summen in einer Liste.
    # Plotte die Liste fuer das jeweilige m.
    

def task1e():

    result = np.random.random(1000,)
    r = 1

    result = np.sqrt(-result**2 + r**2)

    plt.hist(result)

    print(result)

    plt.show()

task1e()

# d. Maybe combining multiple random numbers is even better than using single ones?
# Use numpy to generate new random numbers from a sum of individual numbers, si = 􏰕Mj=1 rj, where the rj are generated from a uniform distribution. Plot scatter plots and histograms of the resulting data sets for M ∈ [2, 3, 5, 10, 20].
# e. Generate random numbers with a uniform distribution in a circle of radius r. (Recent versions of numpy actually have a function for this, but the goal here is to understand the issue first and then to come up with your own solution.)
