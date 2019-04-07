import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
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

def main():
    if len(sys.argv) < 4:
        print('{} pos_dir neg_dir serialized_output_file')

    positive_dir = sys.argv[1]
    negative_dir = sys.argv[2]
    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)

    X = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_positives)] + 
                [0 for i in range(n_negatives)])

    pickle.dump((X, y), open(sys.argv[3], 'wb'))

if __name__ == '__main__':
    main()

