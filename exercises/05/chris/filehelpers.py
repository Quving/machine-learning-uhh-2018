# V2
# Changes:
# - Accepts now a list of keys and returns the keys comma separated from data array

import numpy as np
import scipy.io as io

def get_from_matlab(file, *argv):
    '''
    Loads the Matlab .mat file and returns the data and label set for test or training data.

    Call it like that:
    
    training_data, training_label = get_from_matlab(fn_training, 'train_data', 'train_label')
    '''

    file_data = io.loadmat(file)

    # Search keys
    result = []
    for key in argv:
        result.append([file_data[key]])

    # return the data as unpacked data
    return result, (*result)
    
