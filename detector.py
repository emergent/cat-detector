import sys
import sklearn.svm
from skimage import io, feature, color, transform
import pickle
import lbp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

model_file = 'ser_svm'

WIDTH, HEIGHT = (64, 64)
CELL_SIZE = 4
THRESHOLD = 3.0

def main():
    svm = pickle.load(open(model_file, 'rb'))
    target = color.rgb2gray(io.imread(sys.argv[1]))
    target_scaled = target + 0

    scale_factor = 2.0 ** (-1.0 / 8.0)
    detections = []
    for s in range(16):
        histogram = lbp.get_histogram(target_scaled)

        for y in range(0, histogram.shape[0] - HEIGHT // CELL_SIZE):
            for x in range(0, histogram.shape[1] - WIDTH // CELL_SIZE):
                features = histogram[y:y + HEIGHT // CELL_SIZE,
                                    x:x + WIDTH // CELL_SIZE].reshape(1, -1)
                score = svm.decision_function(features)

                if score[0] > THRESHOLD:
                    print(score, features)
                    scale = (scale_factor ** s)
                    detections.append({
                        'x': x * CELL_SIZE // scale,
                        'y': y * CELL_SIZE // scale,
                        'width': WIDTH // scale,
                        'height': HEIGHT // scale,
                        'score': score[0]
                    })
        target_scaled = transform.rescale(target_scaled, scale_factor)

    print(detections)
    ax = plt.axes()
    ax.imshow(target, cmap=cm.Greys_r)
    ax.set_axis_off()
    for d in detections:
        ax.add_patch(plt.Rectangle((d['y'], d['x']), d['width'], d['height'], edgecolor='r', facecolor='none'))
    #plt.show()
    plt.savefig('out/{}'.format(os.path.basename(sys.argv[1])))

if __name__ == '__main__':
    main()
