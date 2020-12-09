import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def cos_sim(a,b):
    dot = np.dot(a, b.T)
    norm_product = np.linalg.norm(a)*np.linalg.norm(b)
    return dot / (norm_product + 1e-6)

def compare_img_pairs(pair_idxs, idx_to_vec, use_corr=False):
    
    similarities = []
    i = 0
    
    for img1_idx, img2_idx in pair_idxs:
        img1_feature_vec = np.array(idx_to_vec[str(img1_idx)])
        img2_feature_vec = np.array(idx_to_vec[str(img2_idx)])
        if use_corr:
            similarities.append(np.corrcoef(img1_feature_vec, img2_feature_vec)[0][1])
        else:
            similarities.append(cos_sim(img1_feature_vec, img2_feature_vec))
        i += 1
        
    return similarities

def find_highest_accuracy_threshold(samples):
    """
    Returns threshold c for classifier of the form sign(x - c)
    """
    similarities = np.array([sample[0] for sample in samples])
    labels = np.array([sample[1] for sample in samples])

    best_accuracy, best_threshold = 0,0

    for threshold,_ in samples:

        predicted_positive_mask = similarities > threshold
        predicted_negative_mask = similarities <= threshold

        num_true_positives = np.sum(labels[predicted_positive_mask])
        num_true_negatives = np.sum(labels[predicted_negative_mask] == 0)

        accuracy = (num_true_positives + num_true_negatives) / len(samples)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold
            
    return best_threshold, best_accuracy

def similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr=False):

    positive_pair_similarities = compare_img_pairs(test_positive_pairs, idx_to_vec, use_corr=use_corr)
    negative_pair_similarities = compare_img_pairs(test_negative_pairs, idx_to_vec, use_corr=use_corr)
    
    print("Positive Pair Similarity Median: {}".format(np.median(positive_pair_similarities)))
    print("Negative Pair Similarity Median: {}".format(np.median(negative_pair_similarities)))

    similarities = positive_pair_similarities + negative_pair_similarities
    labels = np.concatenate((np.ones(len(positive_pair_similarities)), np.zeros(len(negative_pair_similarities))))
    samples = [(similarities[i],labels[i]) for i in range(len(similarities))]

    print("AUC: {}".format(roc_auc_score(labels, similarities)))

    best_threshold, best_accuracy = find_highest_accuracy_threshold(samples)

    print("Accuracy: {}".format(best_accuracy))
    print("Threshold: {}".format(best_threshold))

    return best_accuracy


