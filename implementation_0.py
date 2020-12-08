from similarity_utils import *

def get_classification_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr):
    return similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr)

def main(train_negative_pairs, train_positive_pairs, test_negative_pairs, test_positive_pairs, idx_to_vec):
    return get_classification_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, False)