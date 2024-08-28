import csv
import numpy as np
from sklearn.metrics import accuracy_score

from ica_prepare_dataset_cnn import *

file_name = '../../../../data/pima-indians-diabetes.csv'
def get_datat(file_name, split_ratio = 0.1):
    with open(file_name) as f:
        reader = csv.reader(f)
        data = list(reader)
        split_idx = int(len(data)*split_ratio)
        data_train = data[:split_idx]
        data_test = data[split_idx:]
        X_train = [ map(float, x[:-1])  for x  in data_train ]
        y_train = [ float(x[-1]) for x  in data_train ]
        X_test = [map(float, x[:-1]) for x in data_test]
        y_test = [float(x[-1]) for x in data_test]
    return X_train,y_train, X_test, y_test



feature_per_digit = 10
datasets = get_ica_conv_dataset_after_pooling(feature_per_digit)

X_train, y_train = datasets[0]
X_test, y_test = datasets[1]
valid_set_x, valid_set_y = datasets[2]

print np.shape(y_train)
#
# X_train,y_train, X_test, y_test = get_datat(file_name)
# print np.shape(X_train)
#
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

clf = BernoulliNB()
clf.fit(X_train, y_train)

predicts = clf.predict(X_test)

# Calculate Accuracy Rate manually
count = len(["ok" for idx, label in enumerate(y_test) if label == predicts[idx]])
print "Accuracy Rate, which is calculated manually is: %f" % (float(count) / len(y_test))

# Calculate Accuracy Rate by using accuracy_score()
print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predicts)
