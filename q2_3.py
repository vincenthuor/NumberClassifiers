'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import q2_2 as help
import scipy as sp
from scipy import misc

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))

    eta_matrix = []
    for key in range(10):
        vector_class = data.get_digits_by_label(train_data, train_labels, key)
        nums_1 = np.count_nonzero(vector_class, axis=0)
        eta = (nums_1 + 2 - 1)/(vector_class.shape[0] + 2 + 2 - 2)
        eta_matrix.append(eta)
    return np.array(eta_matrix)

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    image = []
    for i in range(10):
        img_i = class_images[i]

        image.append(img_i.reshape(8, 8))

    # Plot all means on same axis
    all_concat = np.concatenate(image, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    r = np.random.rand(10, 64)
    for i in range(len(r)):
        for j in range(len(r[i])):
            if r[i][j] >= eta[i][j]:
                r[i][j] = 0
            else:
                r[i][j] = 1
    plot_images(r)

def generative_likelihood(bin_digits, etas):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''

    matrix = []

    for datum in bin_digits: # 1 -> 64
        vector_1x10 = []

        for i in range(len(etas)): # 1 -> 10
            success = datum * np.log(etas[i])
            failure = (1 - datum) * np.log(1 - etas[i])
            binomial_1 = success + failure
            sum_binomial_1 = np.sum(binomial_1) # sum of log = log(product). scalar?
            vector_1x10.append(sum_binomial_1)

        matrix.append(vector_1x10)
    matrix = np.array(matrix)
    return matrix

def conditional_likelihood(bin_digits, etas):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cond_l = np.zeros((len(bin_digits), 10))

    num = generative_likelihood(bin_digits, etas)
    for i in range(len(cond_l)):
        for j in range(len(cond_l[0])):
            cond_l[i][j] = num[i][j] - sp.misc.logsumexp(num[i])

    return cond_l

def avg_conditional_likelihood(bin_digits, labels, etas):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    conditionals = []
    cond_likelihood = conditional_likelihood(bin_digits, etas)

    for i in range(len(cond_likelihood)):
        trueClass_cond = cond_likelihood[i][int(labels[i])]

        conditionals.append(trueClass_cond)
    mean = np.mean(conditionals)

    return mean


def classify_data(bin_digits, etas):
    '''
    Classify new points by taking the most likely posterior class
    '''
    # Compute and return the most likely class

    cond_likelihood = conditional_likelihood(bin_digits, etas)
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    # print(len(eta))

    # gl = generative_likelihood(train_data, eta)
    # print('generative_likelihood:')
    # print(gl)
    # print(gl.shape)

    # cl = conditional_likelihood(train_data, eta)
    # print('cond_likelihood:')
    # print(cl)
    # print(cl.shape)

    ## 2.3.5
    avg_c = avg_conditional_likelihood(train_data, train_labels, eta)
    print('avg cond likelihood train:')
    print(avg_c)
    # # average conditional train; likelihood: -0.9437538618
    avg_c_test = avg_conditional_likelihood(test_data, test_labels, eta)
    print('avg cond likelihood test:')
    print(avg_c_test)
    # # avg cond likelihood test: -0.987270433725


    ## 2.3.6
    cd_train = classify_data(train_data, eta)
    ac_train = help.accuracy_score(train_labels, cd_train)
    print('train accuracy:')
    print(ac_train)
    # # train accuracy 0.774142857143
    cd_test = classify_data(test_data, eta)
    ac_test = help.accuracy_score(test_labels, cd_test)
    print('test accuracy:')
    print(ac_test)
    # # test accuracy: 0.76425

    # Evaluation
    ## 2.3.3
    plot_images(eta)
    ## 2.3.4
    generate_new_data(eta)

if __name__ == '__main__':
    main()
