import sys
import sklearn.svm
import pickle
X, y = pickle.load(open(sys.argv[1], 'rb'))
classifier = sklearn.svm.LinearSVC(C = 0.0001)
classifier.fit(X, y)
pickle.dump(classifier, open(sys.argv[2], 'wb'))
