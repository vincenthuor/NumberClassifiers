'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = self.l2_distance(test_point)
        index_array = digit.argsort()[:k]
        index_labels = [self.train_labels[x] for x in index_array]
        c = Counter(index_labels)
        label, count = c.most_common()[0]
        # Pick a random (usually with a consistent seed, so results are
        # reproducable) point from the 2 points found.

        return label

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    accuracy_array = []
    for k in k_range:
        X = train_data
        kf = KFold(n_splits=10, shuffle=True)
        accuracy_temp = []
        for train, test in kf.split(X):
            knn = KNearestNeighbor(train_data[train], train_labels[train])
            acc_score_test = classification_accuracy(knn, k, train_data[test], train_labels[test])
            accuracy_temp.append(acc_score_test)
        accuracy_mean = np.mean(accuracy_temp)
        accuracy_array.append(accuracy_mean)
    return accuracy_array

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predictions = []
    for i in range(len(eval_data)):
        prediction_val = knn.query_knn(eval_data[i], k)
        predictions.append(prediction_val)

    return accuracy_score(eval_labels, predictions)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # # Example usage:
    # predicted_label = knn.query_knn(test_data[0], 1)
    # print('Predicted label query_knn:')
    # print(predicted_label)

    # # Accuracy
    accuracy_1 = classification_accuracy(knn, 1, train_data, train_labels)
    print('train Accuracy KNN-1:')
    print(accuracy_1)
    accuracy_15 = classification_accuracy(knn, 15, train_data, train_labels)
    print('train Accuracy KNN-15:')
    print(accuracy_15)
    # Accuracy
    # KNN - 1:
    # 1.0
    # Accuracy
    # KNN - 15:
    # 0.963714285714
    accuracy_2 = classification_accuracy(knn, 1, test_data, test_labels)
    print('test Accuracy KNN-1:')
    print(accuracy_2)
    accuracy_12 = classification_accuracy(knn, 15, test_data, test_labels)
    print('test Accuracy KNN-15:')
    print(accuracy_12)

    # Cross validation
    b = cross_validation(train_data, train_labels)
    print("K array: ")
    print(b) # index 4 is the highest value, so K = 4 is optimal K. 0.96557142857142841
    print(b[b.index(max(b))], b.index(max(b)) + 1)


    # Accuracy train classification accuracy AFTER CHOOSING K
    print('Train accuracy:')
    accuracy_train = classification_accuracy(knn, b.index(max(b)) + 1, train_data, train_labels)
    print(accuracy_train)

    print('Avg accuracy:')
    # average accuracy across folds. index 3 -> k = 4
    print(b[b.index(max(b))])

    # test accuracy.
    print('Test accuracy:')
    accuracy_test = classification_accuracy(knn, b.index(max(b)) + 1, test_data, test_labels)
    print(accuracy_test)


if __name__ == '__main__':
    main()