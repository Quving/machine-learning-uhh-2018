## Import always these libraries. For science!
import numpy as np
from sklearn import metrics # To get some nice metrics

### Import special libraries
from sklearn.datasets import fetch_20newsgroups # The Dataset
from sklearn.feature_extraction.text import CountVectorizer # Count occurencies
from sklearn.feature_extraction.text import TfidfTransformer # Normalise and optimize feature vectors
from sklearn.pipeline import Pipeline # For using the pipeline feature of SKLearn

# Classifiers
from sklearn.naive_bayes import MultinomialNB # Naïve Bayes
from sklearn.linear_model import SGDClassifier # SVM

# Optimizers
from sklearn.model_selection import GridSearchCV


## Load data
def load_data(categories, subset='train', shuffle=True, random_state=42, remove=('headers','footers','quotes')):
    # Load the data from newsgroup dataset
    # Output some meta information
    '''
    Loads the training data from the 20 newsgroup dataset

    Parameters to choose:
    - subset: 'train', 'test' or 'all'
    - shuffle: shuffled data or not
    - random_state: used for shuffling the dataset
    - remove: tuple of ('headers','footers','quotes') to hinder the classifier in the end to overfit on arbitrary data

    Access the data from return like:

    - filenames:      dataset.filenames
    - data:           dataset.data
    - target:         dataset.target # category integer ID
    - target names:   dataset.target_names  # category names
    '''

    dataset = fetch_20newsgroups(
        categories=categories,
        subset=subset,
        shuffle=shuffle,
        random_state=random_state,
        remove=remove,
        download_if_missing=True
    )

    # Metadata output
    print('''
    target:      {}
    # filenames: {}
    # data:      {}
    # target:    {}
    '''.format(
        dataset.target_names,
        len(dataset.filenames),
        len(dataset.data),
        len(dataset.target)
    ))

    # Assignment-specific outputs
    print('''
    Q: How many texts have been loaded?
    A: {} texts have been downloaded.
    '''.format(
        len(dataset.filenames)
    ))

    return dataset


def print_report(test_dataset, predicted, classifier_method):

    # Prediction results
    classification_report = metrics.classification_report(test_dataset.target, predicted, target_names=test_dataset.target_names)
    confusion_matrix = metrics.confusion_matrix(test_dataset.target, predicted)

    print('\n')
    print('- Test results -------------------------------------')
    print('Classifier method: {}'.format(classifier_method))
    print(classification_report)
    print('Confusion matrix:')
    print(confusion_matrix)
    print('\n')


class Text_Classifier:

    def __init__(self, dataset):

        # Save global parameters
        self.data = dataset.data
        self.categories = dataset.target
        self.category_names = dataset.target_names


    ## Extract features
    def extract_features(self):
        '''
        Turns the text into numerical feature vectors. This function is based on the bag of words theory where we first assign an integer ID to each word occuring in every document in the training set and then store the occurence per document of each word in vectors.

        We first tokenize the text with CountVectorizer, where we can get the occurence of each word.

        Then we use TfidfTransformer to transform the occurences into frequencies to basically normalise the occurence of words (Term Frequencies) and downscaling unimportant words which occur the most, like "the", "I", "this" and so on (Inverse Document Frequency).
        '''

        # Load a vectorizer
        self.vectorizer = CountVectorizer()
        # Save occurencies of words per document
        self.X_train_counts = self.vectorizer.fit_transform(self.data)

        self.number_words = self.X_train_counts.shape[1]

        self.tfidf_transformer = TfidfTransformer()
        self.features = self.tfidf_transformer.fit_transform(self.X_train_counts) # fits and transforms in the same time. Much processing boost!

        # Assignment-specific output
        print('Q: How many words were found?')
        print('A: There are {} different words in the documents'.format(self.number_words))
        print('\n')
        print('Q: How do you access the list of words?')
        print('A: By calling .get_feature_names()')
        print('\n')
        print('Q: How do you find the numerical index of a word in the feature vectors?')
        print('A: By calling .vocabulary_.get("word")')
        print('\n')

    ## Make classifier
    def make_classifier(self):
        '''
        Makes a multinomial Naïve Bayes classifier.

        - X: features
        - y: categories (self.dataset.target)
        '''

        self.classifier = MultinomialNB().fit(self.features, self.categories)

    ## Train linear model
    def train_classifier(self,train_data):
        '''
        Trains the classifier based on train data.
        '''

        X_train_counts = self.vectorizer.transform(train_data)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)

        predicted = self.classifier.predict(X_train_tfidf)
        predicted_confidences = self.classifier.predict_proba(X_train_tfidf)

        # for train_document, category, predicted_confidence in zip(train_data, predicted, predicted_confidences):
            # Print predictions for trained data
            # print('{} \n Predicted category: {}, confidence: {}'.format(train_document, self.category_names[category], predicted_confidence))

    def pipeline_make_classifier(self, estimators=[
            ('vectorizer', CountVectorizer()),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ]):
        '''
        Use the sklearn.pipeline feature to make a model and change it quickly.
        '''

        self.pipeline = Pipeline(estimators).fit(self.data, self.categories)


    def predict(self, test_dataset):

        if hasattr(self, 'pipeline'):
            print('Predict using pipeline method.')
            predicted = self.pipeline.predict(test_dataset.data)
        else:
            print('Predict using naïve classifier method.')
            X_new_counts = self.vectorizer.transform(test_dataset.data)
            X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
            predicted = self.classifier.predict(X_new_tfidf)

        accuracy = np.mean(predicted == test_dataset.target)

        return predicted, accuracy



def task_a_c(train_dataset, test_dataset):

    classifier_methods = [
        MultinomialNB(),
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
    ]

    parameters = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'transformer__use_idf': (True, False),
        'classifier__alpha': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    }

    classifier = Text_Classifier(train_dataset)

    for classifier_method in classifier_methods:

        # Setup estimators
        estimators=[
            ('vectorizer', CountVectorizer()),
            ('transformer', TfidfTransformer()),
            ('classifier', classifier_method)
        ]

        # Setup classifier
        classifier.pipeline_make_classifier(estimators)

        # Prediction without altering the parameters
        test_predicted, accuracy = classifier.predict(test_dataset)
        print_report(test_dataset, test_predicted, classifier_method)

        # Make a search for the best parameters
        gs_classifier = GridSearchCV(classifier.pipeline, parameters, n_jobs=-1)
        gs_classifier = gs_classifier.fit(train_dataset.data[:400], train_dataset.target[:400])

        print(gs_classifier.best_score_)

        for param_name in sorted(parameters.keys()):
            print("{}: {}".format(param_name, gs_classifier.best_params_[param_name]))


def task_d_e(train_dataset, test_dataset):

    classifier = Text_Classifier(train_dataset)
    classifier.extract_features()
    classifier.make_classifier()
    classifier.train_classifier(train_dataset.data)

    test_predicted, accuracy = classifier.predict(test_dataset)

    print(
        '''
        TASK 08.2d: 

        Q: What is the motivation for the stop words argument of CountVectorizer?
        A: Stop words may be removed from the search scope. This is
           helpful to reduce the set by removing redundant words 
           specifically, like "this", "the" or any other words.

        Q: How many words are found?
        A: {} words were found.

        Q: What is the performance (accuracy)?
        A: The accuracy is: {}

        TASK 08.2e:

        Q: What is the motivation for working with frequencies instead of 
           word counts?
        A: We normalise the data and are able to compare datasets with 
           different word count sizes.
        '''.format(classifier.number_words, accuracy)

    )


if __name__ == "__main__":

    categories=[
        'alt.atheism',
        'comp.graphics',
        'sci.med',
        'soc.religion.christian'
    ]

    # Setup dataset
    train_dataset = load_data(categories, subset='train')
    test_dataset = load_data(categories, subset='test')

    # task_a_c(train_dataset, test_dataset)
    task_d_e(train_dataset, test_dataset)