#!/usr/bin/python3
import numpy as np
import csv

filename = "housing.csv"

## TASK 3a ##############################################################

# Because np. cannot parse strings, I use csv.reader to get the column names.
data = np.genfromtxt(filename, delimiter=',')
with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = list(reader)[0]

## TASK 3b ##############################################################
for head,column in zip(header,list(zip(*data))):
    column = list(column)
    column.pop(0)
    min_val = min(column)
    max_val = max(column)
    min_idx = column.index(min_val)
    max_idx = column.index(max_val)
    mean = np.mean(column)
    print(head, "\n\tmin:", min_val , "\t( index:", min_idx,")", "\n\tmax:", max_val, "\t( index:", min_idx,")", "\n\tmean", mean)

## TASK 3c ##############################################################

