from itertools import chain
import numpy as np
import torch
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.neural_network import MLPClassifier

from similarity_utils import *

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

    mlp = MLPClassifier(hidden_layer_sizes=[4096,512,512])
    model = mlp.fit(train_faces, train_faces_labels)
    return model

def get_classification_accuracy(mlp, test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr=False):
    
    input_x = []
    for idx in idx_to_vec.keys():
        input_x.append(np.array(idx_to_vec[str(idx)]))

    input_x = np.array(input_x)
    output_x = deepest_layer(input_x, mlp)

    mlp_idx_to_vec = {}
    for i in range(output_x.shape[0]):
        mlp_idx_to_vec[str(i)] = output_x[i]

    return similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, mlp_idx_to_vec, use_corr)

def main(train_negative_pairs, train_positive_pairs, test_negative_pairs, test_positive_pairs, idx_to_vec):
    model = train(train_negative_pairs, train_positive_pairs, idx_to_vec)
    accuracy = get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec)
    return accuracy