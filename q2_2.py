'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
from numpy import matlib
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import scipy as sp
from scipy import misc
from scipy.spatial.distance import mahalanobis


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    # means = np.zeros((10, 64))
    #
    means = []
    class_dict = separateClass(train_data, train_labels)
    for key in range(10):
        vector = class_dict[key]
        vector = np.array(vector)
        mean = np.mean(vector, axis=0)
        means.append(mean)
    means = np.array(means)
    return means

def separateClass(train_data, train_labels):
    """
    Separates the classes into an dictionary with keys representing the class

    :param train_data:
    :param train_labels:
    :return: {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
    """
    separated = {}
    for i in range(len(train_data)):
        vector = train_data[i]
        if (train_labels[i] not in separated):
            separated[train_labels[i]] = []
        separated[train_labels[i]].append(vector)

    return separated

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    # covariances = np.zeros((10, 64, 64))
    # Compute covariances

    covariances = []
    class_dict = separateClass(train_data, train_labels)
    for key in range(0, 10):
        vector_class = np.array(class_dict[key])
        vector_class_cov = np.matrix(covariance(vector_class))
        identity_matrix = 0.01 * np.identity(vector_class_cov.shape[0])
        vector_class_stab = identity_matrix + vector_class_cov
        covariances.append(vector_class_stab)

    covariances = np.array(covariances)
    return covariances

def covariance(matrix):
    """
    Returns a covariance matrix given a 2D array

    :param matrix: 700 by 64
    :return: 64 by 64
    """

    N = len(matrix)
    X = matrix
    X -= X.mean(axis=0)
    denom = N - 1
    by_hand = np.dot(X.T, X.conj()) / denom
    return np.array(by_hand)

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_diag = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov_diag = np.log(cov_diag)
        log_diag.append(cov_diag.reshape(8, 8))

    # Plot all means on same axis
    all_concat = np.concatenate(log_diag, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    digit = train data
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    matrix = []

    first_k = (-1/2)*(64) * np.log(2 * np.pi)
    for datum in digits:
        vector_10 = []

        for i in range(10):
            second_k = (-1/2) * np.log(np.linalg.det(covariances[i]))
            third_k = (-1/2) * (datum - means[i]).T.dot(np.linalg.inv(covariances[i])).dot(datum - means[i])
            final_k = first_k + second_k + third_k
            vector_10.append(final_k)

        matrix.append(vector_10)
    matrix = np.array(matrix)
    return matrix

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    cond_l = np.zeros((len(digits), 10))

    num = generative_likelihood(digits, means, covariances)
    for i in range(len(cond_l)):
        for j in range(len(cond_l[0])):
            cond_l[i][j] = num[i][j] - sp.misc.logsumexp(num[i])

    return cond_l

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    conditionals = []
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    for i in range(len(cond_likelihood)):
        trueClass_cond = cond_likelihood[i][int(labels[i])]

        conditionals.append(trueClass_cond)
    mean = np.mean(conditionals)
    return mean

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #
    # # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    # print(means[0])
    # print(means.shape)
    #
    covariances = compute_sigma_mles(train_data, train_labels)
    # print(covariances.shape)
    # print(covariances[0])

    # Evaluation
    # c = compute_mean_mles(train_data[:100], train_labels[:100])
    # print(c)

    #covariance function testing
    # np.random.seed(1)
    # b1 = np.random.rand(64)
    # b2 = np.random.rand(64)
    # c = covariance(b1, b2)
    # d = np.cov(b1, b2)
    # print(c)
    # print(d)
    #
    # np.cov()

    #testing using np.cov
    # q = separateClass(train_data[5:7], train_labels[5:7])
    # print(q)
    # print(train_labels[5:7])
    # qp = np.array(q[3])
    # print(qp.shape)
    # w = np.cov(qp, rowvar=False)
    # print(w.shape)
    # print(w)
    #
    # qpq = np.array(q[3])
    # w2 = covariance(qpq)
    # print(w2)
    # print(w2.shape)

    # b1 = np.random.rand(64)
    # b2 = np.random.rand(64)
    # print(b1)
    # print(b2)
    # b3=b1*b2
    # print(
    #     b3
    # )

    # gl = generative_likelihood(train_data, means, covariances)
    # print('generative_likelihood:')
    # print(gl.shape)
    # print(gl)

    # cl = conditional_likelihood(train_data, means, covariances)
    # print('cond_likelihood:')
    # print(cl)
    # print(cl.shape)

    ## 2.2.2
    avg_cond = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print('avg cond likelihood train:')
    print(avg_cond)
    ## avg cond likelihood train: -0.124587972087
    # print(np.exp(avg_cond))
    avg_cond_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('avg cond likelihood test:')
    print(avg_cond_test)
    ## avg cond likelihood test: -0.196609089674

    # v1 = np.array([1, 2, 3])
    # v2 = np.array([2, 3, 4])
    # v3 = np.array(v1 - v2)

    # mean = np.mean(v3, axis=0)
    # print(v3)

    # m = []
    # vector = q[9]
    # vector = np.array(vector)
    # mean = np.mean(vector, axis=0)
    # m.append(mean)
    # m = np.array(m)
    # print(m)

    # x = [-2.1, -1]
    # y = [3, 1.1]
    # X = np.vstack((x, y))
    # print(X)
    # l = np.linalg.det(X)
    # print(l)
    # print("___")
    # print(np.cov(X.T))
    # print("___")
    # print(covariance(X))
    # plot_cov_diagonal(covariances)

    # l = np.linalg.det(covariances[3])
    # print(l)


    ## 2.2.3
    cd = classify_data(train_data, means, covariances)
    ac = accuracy_score(train_labels, cd)
    print('train accuracy:')
    ## accuracytrain: 0.981428571429
    print(ac)

    # means_test = compute_mean_mles(train_data, train_labels)
    # covariances_test = compute_sigma_mles(train_data, train_labels)

    cd_test = classify_data(test_data, means, covariances)
    ac_test = accuracy_score(test_labels, cd_test)
    print('test accuracy:')
    ## accuracytest: 0.97275
    print(ac_test)

    # a = np.array([[1., 2.], [3., 4.]])
    # print(a)
    # ainv = np.linalg.inv(a)
    # print(ainv)
    # print(a)

    ## 2.2.1 plot
    plot_cov_diagonal(covariances)

if __name__ == '__main__':
    main()