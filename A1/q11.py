'''
Question 2.2 Skeleton Code
Here you should implement and evaluate the Conditional Gaussian classifier.
'''


import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.misc import logsumexp


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class
    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        digits_from_class_i = data.get_digits_by_label(train_data, train_labels, i)
        sum_i = digits_from_class_i.shape[0]
        sum_digit = np.sum(digits_from_class_i, axis=0)
        means[i, :] = sum_digit / sum_i
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class
    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        digits_from_class_i = data.get_digits_by_label(train_data, train_labels, i)
        sum_i = digits_from_class_i.shape[0]
        mean = means[i]
        delta = digits_from_class_i - mean
        temp = np.dot(np.transpose(delta), delta)
        I_matrix = np.dot(0.01, np.identity(64))
        covariances[i, :, :] = np.add(temp / sum_i, I_matrix)
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)
    Should return an n x 10 numpy array
    '''
    results = np.zeros((digits.shape[0], 10))
    for j in range(digits.shape[0]):
        digit = digits[j]
        for i in range(10):
            mean = means[i]
            covariance = covariances[i]
            d = digits.shape[1]/2
            pi_2 = 2 * np.pi
            delta = digit - mean
            sig_k_inv = np.linalg.inv(covariance)
            det_sig_k = np.linalg.det(covariance)
            term_1 = - 1/2 * d * np.log2(pi_2)
            term_2 = - 1/2 * np.log(det_sig_k)
            term_exp = - 1/2 * delta.dot(sig_k_inv).dot(np.transpose(delta))
            result = term_1 + term_2 + term_exp
            results[j][i] = result
    return results


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:
        log p(y|x, mu, Sigma)
    This should be a numpy array of shape (n, 10)
    Where n is the number of data points and 10 corresponds to each digit class
    '''
    temp = generative_likelihood(digits, means, covariances)
    log_sum = logsumexp(temp, axis=1).reshape(digits.shape[0], 1)
    result = temp - log_sum
    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels
        AVG( log p(y_i|x_i, mu, Sigma) )
    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    count = 0
    total = digits.shape[0]
    for i in range(total):
        count += cond_likelihood[i, int(labels[i])]
    avg = count / total
    return avg


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    result = []
    for i in range(len(digits)):
        result.append(cond_likelihood[i].argmax())
    return result


def plot_lead_eig(covariances):
    results = []
    for i in range(10):
        w, v = np.linalg.eig(covariances[i])
        lead_eig = np.reshape(v[:, np.argmax(w)], (8, 8))
        results.append(lead_eig)
    all_concat = np.concatenate(results, 1)
    plt.imshow(all_concat)
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    covariances2 = compute_sigma_mles(test_data, test_labels)
    train_con = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_con = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("the avg conditional log-likelihood for train data is: " + str(train_con))
    print("the avg conditional log-likelihood for test data is: " + str(test_con))

    predict_train_labels = classify_data(train_data, means, covariances)
    predict_test_labels = classify_data(test_data, means, covariances)
    acc = 0
    total_predict = len(predict_train_labels)
    for i in range(total_predict):
        if predict_train_labels[i] == train_labels[i]:
            acc += 1
    train_accuracy = acc / total_predict
    acc2 = 0
    total_test = len(predict_test_labels)
    for j in range(total_test):
        if predict_test_labels[j] == test_labels[j]:
            acc2 += 1
    test_accuracy = acc2 / total_test
    print("the accuracy for train data is: " + str(train_accuracy))
    print("the accuracy for test data is: " + str(test_accuracy))

    # plot_lead_eig(covariances)
    plot_lead_eig(covariances2)

    # Evaluation


if __name__ == '__main__':
    main()
