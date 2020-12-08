from similarity_utils import *

# just for compatibility reasons
def train(train_negative_pairs, train_positive_pairs, idx_to_vec, cv=True):
	return None

def get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr=False):
    return similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr)

def main(train_negative_pairs, train_positive_pairs, test_negative_pairs, test_positive_pairs, idx_to_vec):
    return get_classification_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, False)