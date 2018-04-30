import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import io
from sklearn.neighbors import KNeighborsClassifier
import time
import resource
import platform


# knn classifier to recognize digits
# 16 x 16 sized images, greyscale, v: 0 - 255
# 10 classes: 0 - 9
# Size of training data: 1000, stored in 1000 x 256 Matlab matrix
# (usps_train.mat)
# training label vector: 10000 x 1
# 100 test images (usps_test.mat)

fn_training =   'usps/usps_train.mat'
fn_test =       'usps/usps_test.mat'
img_plots_dir = 'ctask3_img'
knn_plots_dir = 'ctask3_knn'


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

def plot_images_from_arrays(data,number_of_images, filename):
    print("Plotting {} images".format(number_of_images))

    breakint = 0

    for i in range(len(data)):
        
        # Reshape them into 16 x 16
        img = data[i].reshape(16,16)

        plt.gcf().clear()

        # Plot with pyplot.imshow and cmap='grey'
        plt.imshow(img, cmap='gray')

        save_as = filename + '_' + str(i) + '.png'
        plt.savefig(img_plots_dir + '/' + save_as, bbox_inches='tight')

        i += 10
        breakint += 1

        if breakint is number_of_images:
            break

def knn_digit_classifier(numbers):

    starttime = time.time()
    print("----------------------------------------")
    print("Building the datasets...")
    f_training_data, f_training_label = filter_data(training_data, training_label, numbers)
    f_test_data, f_test_label = filter_data(test_data, test_label, numbers)
    ### B: Plot images
    # Choose some images
    # Every tenth
    img_filename = str(numbers).strip('[]').replace(', ','')
    plot_images_from_arrays(f_training_data,10, img_filename)

    ### C: Evaluate
    # Test classifier with values k = 1,3,5,7,10,15
    k = 1,3,5,7,10,15

    training_scores = np.array([])
    test_scores = np.array([])
    prediction_probs = np.array([])

    print("Start training and classifying. Calculating the scores...")
    for i in k:

        # Instantiate learning model
        classifier = KNeighborsClassifier(n_neighbors=i)

        # Fitting the model
        classifier.fit(f_training_data, f_training_label.ravel())

        # Scoring
        training_score = classifier.score(f_training_data, f_training_label)
        training_scores = np.append(training_scores, [training_score])

        test_score = classifier.score(f_test_data, f_test_label)
        test_scores = np.append(test_scores, [test_score])
    
    print("Finished. Plotting now result graphics.")

    # plot training and test errorsgit 
    plt.gcf().clear()
    ki = [i for i in range(0, len(k))]
    print("Prediction probabilities: {}".format(prediction_probs))
    plt.xticks(ki, k)
    print("Numbers: {}, Training Performance: {}, Test Performance: {}".format(numbers, training_scores, test_scores))
    plt.plot(ki,training_scores, color="g", label="Training Performance", marker='o')
    plt.plot(ki,test_scores, color="r", label="Test Performance", marker='o')
    plt.title("KNN Classifier. Classified digits: {}".format(numbers))
    plt.xlabel('Number of neighbours (k)')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(knn_plots_dir + '/' + str(numbers).strip('[]').replace(', ','') + '.png', bbox_inches='tight')
    
    endtime = time.time() - starttime

    # Calculate Bytes to megabytes, based on OS
    b_to_mb = 0
    if platform.system() == "Darwin":
        b_to_mb = 1000000
    else:
        b_to_mb = 1000
    
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / b_to_mb

    print("Finished. Processing time: {}s. Memory usage: {}mb".format(endtime, memory))

if __name__ == "__main__":
    start = time.time()
    knn_digit_classifier([2,3])

    ####### 3 to 8 comparison
    ### D: Classify other digits
    # Run algorithm for digits: 3,4,5,6,7,8
    # Compare performance

    knn_digit_classifier([3,4,5,6,7,8])

    # What about 3 and 8?
    knn_digit_classifier([3,8])

    # 1 vs. 7
    knn_digit_classifier([1,7])


    print(time.time() - start)
