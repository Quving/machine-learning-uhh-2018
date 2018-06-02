#!/usr/bin/python
from sklearn.datasets import fetch_lfw_people
import numpy as np

class LfwPeople():
    def __init__(self):
        self.lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        self.n_samples, h, w = self.lfw_people.images.shape
        self.n_features = self.lfw_people.data.shape[1]
        target_names = self.lfw_people.target_names
        self.n_classes = target_names.shape[0]

        self.value_max, self.value_min = 0, 0
        for image in self.lfw_people.images:
            max_local =  np.amax(image)
            min_local =  np.amin(image)
            self.value_max = max_local if max_local > self.value_max else self.value_max
            self.value_min = min_local if min_local > self.value_min else self.value_min

    def print_informations(self):
        print("Total dataset size:")
        print("\t", "n_samples: %d" % self.n_samples)
        print("\t", "n_features: %d" % self.n_features)
        print("\t", "n_classes: %d" % self.n_classes)
        print("\t", "value_range: [", self.value_min,",", self.value_max, "]")





