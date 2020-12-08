import numpy as np
import torch

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def svm_preprocessing(negative_pairs, positive_pairs, idx_to_vec):

    pairs = np.concatenate((negative_pairs, positive_pairs))
    y = np.concatenate((np.zeros(len(negative_pairs)), np.ones(len(positive_pairs))))
    x = []

    for img1_idx, img2_idx in pairs:
        img1_feature_vec = np.array(idx_to_vec[str(img1_idx)])
        img2_feature_vec = np.array(idx_to_vec[str(img2_idx)])
        x_i = img1_feature_vec - img2_feature_vec
        x.append(x_i)

    x = np.array(x)    
    return x, y


def train(train_negative_pairs, train_positive_pairs, idx_to_vec, cv=True):
    svm = SVC(kernel='rbf')
    if cv:
        svm = GridSearchCV(cv=5,
                             estimator=svm,
                             param_grid={"C": [10**(i) for i in range(-5,6)]},
                             scoring='accuracy',
                             refit=True
                             )
    train_x, train_y = svm_preprocessing(train_negative_pairs, train_positive_pairs, idx_to_vec)
    svm = svm.fit(train_x,train_y)

    if cv:
        print(svm.best_params_)
    return svm


def get_classification_accuracy(svm, test_negative_pairs, test_positive_pairs, idx_to_vec):
    test_x, test_y = svm_preprocessing(test_negative_pairs, test_positive_pairs, idx_to_vec)
    return svm.score(test_x, test_y)

def main(train_negative_pairs, train_positive_pairs, test_negative_pairs, test_positive_pairs, idx_to_vec):
    model = train(train_negative_pairs, train_positive_pairs, idx_to_vec, False)
    accuracy = get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec)
    return accuracy