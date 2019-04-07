import sys
import sklearn.svm
import pickle

classifier = pickle.load(open(sys.argv[1], 'rb'))
X, y = pickle.load(open(sys.argv[2], 'rb'))
y_predict = classifier.predict(X)
correct = 0
for i in range(len(y)):
    if y[i] == y_predict[i]: correct += 1
print('Accuracy: %f' % (float(correct) / len(y)))
