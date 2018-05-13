import numpy as np
import matplotlib.pyplot as plt
import filehelpers as fh
import scipy.io as io
from sklearn import datasets
from sklearn.linear_model import LinearRegression

### TASK 5

file        = 'iris_multiclass.mat'
cpu_mode    = -1 

def task5_1():

    # Load the file, extract indices_train, indices_test
    data, species, meas, indices_train, indices_test = fh.get_from_matlab(file, 'species', 'meas' , 'indices_train', 'indices_test')

    new_species = []
    for item in species:
        for species_name in item:
            new_species += [str(species_name[0][0])]

    # Unpack species names
    species = [str(species_name[0][0]) for list_entry in species for species_name in list_entry]
    species = np.array(species)
    # Make subclasses
    species_setosa = species[:] == 'setosa'
    species_versicolor = species[:] == 'versicolor'
    species_virginica = species[:] == 'virginica'

    # format meas
    meas = meas[0]

    # Make train and test data by indices
    # We have to reduce the saved indice by one because it seems like that the 
    # base is 1, not 0 (because they reach from 1 to 150, not from 0 to 149)
    train_data = [meas[i-1] for i in indices_train][0][0]
    train_label = [species[i-1] for i in indices_train][0][0]
    test_data = [meas[i-1] for i in indices_test][0][0]
    test_label = [species[i-1] for i in indices_test][0][0]

    print('---- Training data set summary:')
    print('training indices: {}'.format(indices_train))
    print('test indices: {}'.format(indices_test))
    print('species: {}'.format(species))

    print('\nMeas:\n')

    print('         l s, w s, l p, w p')
    print('{}'.format(meas))

    print('\n')

    print('Train data:\n{}\nSize: {}'.format(train_data, len(train_data)))
    print('Test data:\n{}\nSize: {}'.format(test_data, len(test_data)))

    print('Train label:\n{}\nSize: {}'.format(train_label, len(train_label)))
    print('Test label:\n{}\nSize: {}'.format(test_label, len(test_label)))

    print('\n')

    # Estimate w_setosa, w_versicolor and w_virginica based on training data
    w_setosa = np.average(species_setosa)
    w_versicolor = np.average(species_versicolor)
    w_virginica = np.average(species_virginica)

    print('-- Weight estimations for species:')
    print("- setosa: {}\n- versicolor: {}\n- virginica: {}".format(w_setosa, w_versicolor, w_virginica))

    lm = LinearRegression(normalize=True, n_jobs=cpu_mode)

    lm.fit(train_data, train_label)


if __name__ == "__main__":
    task5_1()