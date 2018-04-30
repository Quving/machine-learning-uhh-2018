#!/usr/bin/python3
import scipy.io as sio

def task2a():
    mat_dict = sio.loadmat('Adot.mat')
    print(mat_dict["X"])
    for key in  mat_dict.keys():
        print(key)



if __name__ == "__main__":
    task2a()
