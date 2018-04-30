#!/usr/bin/python3

import time
import pickle
import numpy as np
from PIL import Image
import scipy.io as sio
from matplotlib import pyplot as plt

def mat_to_dict(filename):
    mat_dict = sio.loadmat(filename)
    print(filename,"has the following keys:",mat_dict.keys())
    return mat_dict

def export_np_to_pngs(train_dict, indices):
    for index in indices:
        image_np = train_dict["train_data"][index]
        image_np = np.reshape(image_np, [16,16])
        plt.imshow(image_np, interpolation='nearest')
        plt.savefig("usps/" + str(index) + ".png", bbox_inches='tight')

def sort_dict(dict_raw, type):
    dict_sorted = dict()
    for label, sample in zip(dict_raw[type +"_label"], dict_raw[type + "_data"]):
        if label[0] in dict_sorted:
            dict_sorted[label[0]].append(sample)
        else:
            dict_sorted[label[0]] = [sample]
    return dict_sorted

def train_knn(train_dict_sorted, test_dict_sorted, class_to_train, k_list):
    from sklearn.neighbors import KNeighborsClassifier
    scores = dict()
    class_size_train = len(train_dict_sorted[class_to_train[0]])
    class_size_test = len(test_dict_sorted[class_to_train[0]])
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for class_label in class_to_train:
            x_train += train_dict_sorted[class_label]
            y_train += ([class_label] * class_size_train)
            x_test += test_dict_sorted[class_label]
            y_test += ([class_label] * class_size_test)
        knn.fit(x_train,y_train)
        score_train = knn.score(x_train, y_train) # Collect scores
        score_test = knn.score(x_test, y_test) # Collect scores
        print(k, ": train", score_train, "test", score_test)
        scores[k] = {"history_train":score_train, "history_test": score_test, "knn": knn}
    return scores

def persist(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def plot_scores(models, title, filename):
    x_train = list(models.keys())
    y_train = []
    y_test = []
    for key, value in models.items():
        y_train.append(value["history_train"])
        y_test.append(value["history_test"])
    plt.plot(x_train, y_train)
    plt.plot(x_train, y_test)

    plt.xlabel('k')
    plt.ylabel('error')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()

if __name__ == "__main__":
    start = time.time()
    train_dict_raw = mat_to_dict('usps/usps_train.mat')
    train_dict_sorted = sort_dict(train_dict_raw, "train")

    test_dict_raw = mat_to_dict('usps/usps_test.mat')
    test_dict_sorted = sort_dict(test_dict_raw, "test")
    print("2 vs 3")
    models_2_3 = train_knn(
            train_dict_sorted = train_dict_sorted,
            test_dict_sorted = test_dict_sorted,
            class_to_train = [2,3],
            k_list = [1,3,5,7,10,15]
            )

    print("8 vs 3")
    test_dict_sorted = sort_dict(test_dict_raw, "test")
    models_3_8 = train_knn(
            train_dict_sorted = train_dict_sorted,
            test_dict_sorted = test_dict_sorted,
            class_to_train = [8,3],
            k_list = [1,3,5,7,10,15]
            )
    # persist(models, 'models.p')
    # models = pickle.load( open( "models.p", "rb" ))
    plot_scores(models_3_8, "2 vs 3", "2vs3.png")
    plot_scores(models_2_3, "3 vs 8", "3vs8.png")
    export_np_to_pngs(train_dict_raw, [0,1001,2001,3001,4001,5001,6001,7001])
    print(time.time() - start)
