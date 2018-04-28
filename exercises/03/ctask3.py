import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from sklearn.neighbors import KNeighborsClassifier


# knn classifier to recognize digits
# 16 x 16 sized images, greyscale, v: 0 - 255
# 10 classes: 0 - 9
# Size of training data: 1000, stored in 1000 x 256 Matlab matrix
# (usps_train.mat)
# training label vector: 10000 x 1
# 100 test images (usps_test.mat)

fn_training =   'usps_train.mat'
fn_test =       'usps_test.mat'
plots_dir =     'ctask3_img'


####### 2 vs. 3 comparison
### A: Read and prepare
# Read the Matlab matrices
def prepare_matdata(file, key_data, key_label):
    file_data = io.loadmat(file)
    # Convert the data type from uint8 to double
    return np.array(file_data[key_data]).astype(np.float64), np.array(file_data[key_label]).astype(np.float64)

training_data, training_label = prepare_matdata(fn_training, 'train_data', 'train_label')
test_data, test_label = prepare_matdata(fn_test, 'test_data', 'test_label')

# Get the relevant arrays out of them (for a: 2, 3)
def filter_data(data, label, numbers):

    filter = None
    for i in numbers:
        if filter is None:
            filter = label == i
        filter += label == i
    
    return data[filter.flatten()], label[filter.flatten()]

def kmeans_digit_classifier(numbers):
    f_training_data, f_training_label = filter_data(training_data, training_label, numbers)
    f_test_data, f_test_label = filter_data(test_data, test_label, numbers)

    ### B: Plot images
    # Choose some images
    # Every tenth
    number_of_images = 10
    breakint = 0
    for i in range(len(f_training_data)):
        
        # Reshape them into 16 x 16
        img = f_training_data[i].reshape(16,16)

        # Plot with pyplot.imshow and cmap='grey'
        plt.imshow(img, cmap='gray')

        save_as = str(i) + '.png'
        plt.savefig(plots_dir + '/' + save_as, bbox_inches='tight')

        i += 10
        breakint += 1

        if breakint is number_of_images:
            break

    ### C: Evaluate
    # Test classifier with values k = 1,3,5,7,10,15
    k = 1,3,5,7,10,15

    scores = np.array([])

    for i in k:
        classifier = KNeighborsClassifier(i)
        classifier.fit(training_data, training_label)
        score = classifier.score(test_data, test_label)
        scores = np.append(scores, [score])
        print(score)

    # plot training and test errors
    plt.gcf().clear()
    print(scores)
    plt.plot(k,scores, color="g")
    plt.savefig('ctask3-kmeans' + str(numbers) + '.png', bbox_inches='tight')
    plt.show()

kmeans_digit_classifier([2,3])

####### 3 to 8 comparison
### D: Classify other digits
# Run algorithm for digits: 3,4,5,6,7,8
# Compare performance

kmeans_digit_classifier([3,4,5,6,7,8])