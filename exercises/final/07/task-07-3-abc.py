#!/usr/bin/python3
from lfw_people import LfwPeople

lfwpeople = LfwPeople()
lfwpeople.print_dataset_description()
lfwpeople.plot_n_samples_of_each_class(n=3)



