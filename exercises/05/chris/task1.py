import numpy as np
import matplotlib.pyplot as plt
import filehelpers as fh
import scipy.io as io
from sklearn import datasets

### TASK 5

file    = 'iris_multiclass.mat'

def task5a():

    # Load the file, extract indices_train, indices_test
    data, species, meas, indices_train, indices_test = fh.get_from_matlab(file, 'species', 'meas' , 'indices_train', 'indices_test')

    print(indices_train)
    print(indices_test)

    # Estimate w_setsoa, w_versicolor and w_virginica based on indices_train
    


if __name__ == "__main__":
    task5a()