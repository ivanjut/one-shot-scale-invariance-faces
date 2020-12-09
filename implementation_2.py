from itertools import chain
import numpy as np
import torch
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.neural_network import MLPClassifier

from similarity_utils import *
from preprocessing_utils import get_idx_to_vec, k_fold_sample, get_test_pairs

def deepest_layer(data, MLP, layer=0):
    L = ACTIVATIONS['relu'](np.matmul(data, MLP.coefs_[layer]) + MLP.intercepts_[layer])
    layer += 1
    if layer >= len(MLP.coefs_)-1:
        return L
    else:
        return deepest_layer(L, MLP, layer=layer)

def mlp2_preprocessing(idx_to_vec, train_faces_indices):
    x = []
    for i in train_faces_indices:
        img_feature_vec = np.array(idx_to_vec[str(i)])
        x.append(img_feature_vec)
    x = np.array(x)
    
    return x

def train(train_negative_pairs, train_positive_pairs, idx_to_vec):

    train_faces = list(chain.from_iterable(train_positive_pairs)) + list(chain.from_iterable(train_negative_pairs))
    train_faces_indices = np.unique(train_faces)
    train_faces_labels = [(i-1) // 5 for i in train_faces_indices]
    train_faces = mlp2_preprocessing(idx_to_vec, train_faces_indices)


    mlp = MLPClassifier(hidden_layer_sizes=[512], max_iter=10000)
    model = mlp.fit(train_faces, train_faces_labels)
    print(model.loss_)
    return model

def get_classification_accuracy(mlp, test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr=False):
    
    input_x = []
    for idx in idx_to_vec.keys():
        input_x.append(np.array(idx_to_vec[str(idx)]))

    input_x = np.array(input_x)
    output_x = deepest_layer(input_x, mlp)

    mlp_idx_to_vec = {}
    for i in range(output_x.shape[0]):
        mlp_idx_to_vec[str(i+1)] = output_x[i]

    return similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, mlp_idx_to_vec, use_corr)

def main(train_negative_pairs, train_positive_pairs, test_negative_pairs, test_positive_pairs, idx_to_vec):
    model = train(train_negative_pairs, train_positive_pairs, idx_to_vec)
    accuracy = get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec)
    return accuracy


if __name__ == "__main__":

    idx_to_vec = get_idx_to_vec('cornet2')
    c_test_offset_values = [0,1,2,3,4]
    for train_pairs, test_pairs in k_fold_sample(2, [0,1,2,3,4], c_test_offset_values):
        train_negative_pairs, train_positive_pairs = train_pairs[0], train_pairs[1]
        test_negative_pairs, test_positive_pairs = get_test_pairs(test_pairs[0], test_pairs[1], 1, len(c_test_offset_values))

        train_faces = set(list(chain.from_iterable(train_positive_pairs)) + list(chain.from_iterable(train_negative_pairs)))
        test_faces = set(list(chain.from_iterable(test_positive_pairs)) + list(chain.from_iterable(test_negative_pairs)))
        assert len(train_faces.intersection(test_faces)) == 0

        print("Num Negative Test Pairs: {}".format(len(test_negative_pairs)))
        print("Num Positive Test Pairs: {}".format(len(test_positive_pairs)))
        model = train(train_negative_pairs, train_positive_pairs, idx_to_vec)
        accuracy = get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec)
        print("Accuracy: {}".format(accuracy))
