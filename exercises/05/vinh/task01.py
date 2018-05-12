#!/usr/bin/python3
import copy
import numpy as np
import scipy.io as sio
import pprint
from sklearn.neighbors import KNeighborsClassifier

# Reads in a .mat  file and returns a dictionary.
def import_mat(filename):
    mat_dict = sio.loadmat(filename)
    return mat_dict

# Returns a dictionary with the class as key and its features as value.
# Input 1: The dictionary generated from .mat-file.
# Inout 2: Key of if its train,- or test-set.
def sort_indices_x(iris_multiclass_dict, key):
    out = dict()
    species = iris_multiclass_dict["species"]
    meas = iris_multiclass_dict["meas"]
    for idx in iris_multiclass_dict[key][0]:
        idx-= 1
        key = species[idx][0][0]

        if key in out:
            out[key].append(list(meas[idx]))
        else:
            out[key] = [list(meas[idx])]
    return out

# Returns a trained knn, its training scores and test_scores packaged in a dictionary.
def get_trained_knn_classifier(training_data, test_data):
    knn = KNeighborsClassifier(n_neighbors=3)
    X,y,X_test, y_test = [], [], [], []
    for key in training_data.keys() & test_data.keys():
        # Training samples
        X += training_data[key]
        y += [key]*len(training_data[key])
        # Test samples
        X_test += test_data[key]
        y_test += [key]*len(test_data[key])

    knn.fit(X,y)
    training_score = knn.score(X,y)
    test_score = knn.score(X_test,y_test)

    return {"knn": knn, "training_score": training_score, "test_score": test_score}

def binary_prediction(test_data, knn):
    out = dict()
    for species, meas in copy.deepcopy(test_data).items():
        if not species in out:
            out[species] = []

        for mea in meas:
            proba =  np.amax(knn.predict_proba([mea]))
            pred = knn.predict([mea])[0]
            out[species].append({"mea": mea, "pred" : pred, "proba": proba})

    return out

def reply_task_1c():
    print("A label of the highest probability is predicted. By training a classifier to distingisch")
    print("between label a and b it always predict classes a or b doesnt matter what is given as input.")
    print("A prediction like 'Nor a or b' would be awesome.")

def task1():
    # 1a
    iris_multiclass_dict = import_mat('../iris_multiclass.mat')
    indices_train_sorted = sort_indices_x(iris_multiclass_dict, "indices_train")
    indices_test_sorted = sort_indices_x(iris_multiclass_dict, "indices_test")
    knn_dict = get_trained_knn_classifier(indices_train_sorted, indices_test_sorted)
    # 1b
    binary_prediction(indices_test_sorted, knn_dict["knn"])
    # 1c
    reply_task_1c()

if __name__ == "__main__":
    task1()
