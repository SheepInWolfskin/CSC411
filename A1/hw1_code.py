import numpy as np
import random
from sklearn import tree
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle


def load_data():
    f_file = open("clean_fake.txt", "r")
    r_file = open("clean_real.txt", "r")
    f_line = f_file.readlines()
    r_line = r_file.readlines()
    data = []
    Real_Fake = []
    for item in f_line:
        data.append(item)
        Real_Fake.append(0)
    for item in r_line:
        data.append(item)
        Real_Fake.append(1)
    for i in range(len(data)):
        temp = data[i][:-1]
        data[i] = temp

    # initialize the vector
    t_vec = TfidfVectorizer()
    np.asarray(Real_Fake)
    t_fitted = t_vec.fit_transform(data).toarray()
    vocabulary = t_vec.get_feature_names()
    t_fitted, Real_Fake = shuffle(t_fitted, Real_Fake, random_state=0)

    # mark the length
    seventy_marker = round(0.7*len(t_fitted))
    eight_five_marker = round(0.85 * len(t_fitted))

    # define the labels and data set
    train_X = t_fitted[: seventy_marker]
    train_Y = Real_Fake[: seventy_marker]
    valid_X = t_fitted[seventy_marker: eight_five_marker]
    valid_Y = Real_Fake[seventy_marker: eight_five_marker]
    test_X = t_fitted[eight_five_marker:]
    test_Y = Real_Fake[eight_five_marker:]
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y, vocabulary


def select_model():
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y, vocabulary = load_data()
    max_depth_array = [5, 50, 150, 250, 400]
    mode = ["gini", "entropy"]
    for mod in mode:
        for i in range(len(max_depth_array)):
            temp = tree.DecisionTreeClassifier(max_depth=max_depth_array[i], criterion=mod)
            temp.fit(train_X, train_Y)

            score = 0
            for j in range(len(valid_X)):
                if temp.predict(valid_X)[j] == valid_Y[j]:
                    score += 1

            mark = score/len(valid_X)

            temp.score(valid_X, valid_Y)
            # print("mode is " + mod)
            # print(max_depth_array[i])
            # print("Validation Accuracy: " + str(mark))

    # Visualization example
    tree.export_graphviz(temp, out_file="\.test1.dot", max_depth=2, feature_names=vocabulary, filled=True)

def compute_information_gain(input):
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y, vocabulary = load_data()
    total_real = 0
    total_fake = 0
    with_input_real = 0
    with_input_fake = 0
    no_input_real = 0
    no_input_fake = 0

    voc_index = vocabulary.index(input)
    for item in train_Y:
        if item == 0:
            total_fake += 1

    for item in train_Y:
        if item == 1:
            total_real += 1

    for i in range(len(train_X)):
        if train_Y[i] == 0:
            if train_X[i][voc_index] == 0:
                no_input_fake += 1

            if train_X[i][voc_index] != 0:
                with_input_fake += 1

        if train_Y[i] == 1:
            if train_X[i][voc_index] == 0:
                no_input_real += 1

            if train_X[i][voc_index] != 0:
                with_input_real += 1

    p_real = total_real / (total_real + total_fake)
    p_fake = total_fake / (total_real + total_fake)
    HY = - p_real * np.log2(p_real) - p_fake * np.log2(p_fake)

    p_fake_given_with = with_input_fake / (with_input_fake + with_input_real)
    p_real_given_with = with_input_real / (with_input_fake + with_input_real)
    p_fake_given_no = no_input_fake / (no_input_fake + no_input_real)
    p_real_given_no = no_input_real / (no_input_fake + no_input_real)
    no_factor = (no_input_fake+no_input_real) / (total_fake + total_real)
    with_factor = (with_input_real+with_input_fake) / (total_fake + total_real)

    HY_X = no_factor * (- p_fake_given_no * np.log2(p_fake_given_no) - p_real_given_no * np.log2(p_real_given_no)) + with_factor * (-p_fake_given_with * np.log2(p_fake_given_with) - p_real_given_with * np.log2(p_real_given_with))

    Info_Gain = HY - HY_X
    print(Info_Gain)
    return Info_Gain
# select_model()
compute_information_gain("hillary")

