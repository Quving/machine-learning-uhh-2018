import numpy as np
from PIL import Image
import scipy.io as sio
from matplotlib import pyplot as plt

def task3a(filename):
    mat_dict = sio.loadmat(filename)
    print(filename,"has the following keys:",mat_dict.keys())
    return mat_dict


def export_np_to_pngs(filename):
    train_dict = task3a(filename)
    i = 0
    for image_np in train_dict["train_data"]:
        image_np = np.reshape(image_np, [16,16])
        plt.imshow(image_np, interpolation='nearest')
        plt.savefig("usps/" + str(i) + ".png", bbox_inches='tight')
        i += 1
        print(i)
        # plt.show()

if __name__ == "__main__":
    export_np_to_pngs('usps/usps_train.mat')
