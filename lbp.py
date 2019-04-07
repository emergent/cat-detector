import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
import sklearn.svm
import pickle

LBP_POINTS = 24
LBP_RADIUS = 3
CELL_SIZE = 4

def get_histogram(image):
    lbp = feature.local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, 'uniform')

    bins = LBP_POINTS + 2
    histogram = np.zeros(shape = (image.shape[0] // CELL_SIZE,
                                image.shape[1] // CELL_SIZE, bins),
                                dtype = np.int)
    for y in range(0, image.shape[0] - CELL_SIZE, CELL_SIZE):
        for x in range(0, image.shape[1] - CELL_SIZE, CELL_SIZE):
            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    histogram[y // CELL_SIZE, x // CELL_SIZE,
                                int(lbp[y + dy, x + dx])] += 1

    return histogram

def get_features(directory):
    features = []
    for fn in iglob('{}/*.png'.format(directory)):
        image = color.rgb2gray(io.imread(fn))
        features.append(get_histogram(image).reshape(-1))
        features.append(get_histogram(np.fliplr(image)).reshape(-1))
    return features

def learn_svm(X, y):
    classifier = sklearn.svm.LinearSVC(C = 0.0001)
    classifier.fit(X, y)
    return classifier

def test_svm(X, y, classifier):
    y_predict = classifier.predict(X)
    correct = 0
    for i in range(len(y)):
        if y[i] == y_predict[i]: correct += 1
    print('Accuracy: {}'.format(float(correct) / len(y)))

def get_lbp(positive_dir, negative_dir):
    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)

    X = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_positives)] +
                [0 for i in range(n_negatives)])

    return (X, y)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('{} [pos_dir] [neg_dir] [pos_test_dir] [neg_test_dir] [serialized_output_file]')

    # training
    X, y = get_lbp(sys.argv[1], sys.argv[2])
    classifier = learn_svm(X, y)

    # test
    X2, y2 = get_lbp(sys.argv[3], sys.argv[4])
    test_svm(X2, y2, classifier)

    pickle.dump(classifier, open(sys.argv[5], 'wb'))
