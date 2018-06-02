#!/usr/bin/python
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

class LfwPeople():
    def __init__(self):
        self.lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        self.n_samples, h, w = self.lfw_people.images.shape
        self.n_features = self.lfw_people.data.shape[1]
        target_names = self.lfw_people.target_names
        self.n_classes = target_names.shape[0]
        self.height, self.width = self.lfw_people.images[0].shape
        print(self.height, self.width)
        self.value_max, self.value_min = 0, 0
        for image in self.lfw_people.images:
            max_local =  np.amax(image)
            min_local =  np.amin(image)
            self.value_max = max_local if max_local > self.value_max else self.value_max
            self.value_min = min_local if min_local > self.value_min else self.value_min

    def print_dataset_description(self):
        print("Total dataset size:")
        print("\t", "n_samples: %d" % self.n_samples)
        print("\t", "n_features: %d" % self.n_features)
        print("\t", "n_classes: %d" % self.n_classes)
        print("\t", "value_range: [", self.value_min,",", self.value_max, "]")

    def plot_n_samples_of_each_class(self, n):
        bag = dict()
        for class_index, image in zip(self.lfw_people.target, self.lfw_people.images):
            if class_index in bag:
                if len(bag[class_index]) < n:
                    bag[class_index].append(image)
            else:
                bag[class_index] = [image]

        col, row = n,len(bag.keys())
        plt.figure(figsize=(1.8 * col, 2.4 * row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

        index = 0
        for class_index, images in bag.items():
            for image in images:
                index += 1
                plt.subplot(row, col , index)
                plt.imshow(image.reshape((self.height, self.width)), cmap=plt.cm.gray)
                plt.title(self.lfw_people.target_names[class_index], size=12)
            plt.xticks(())
            plt.yticks(())
        plt.show()



