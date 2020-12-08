import numpy as np

from sklearn.metrics import roc_auc_score

def cos_sim(a,b):
    dot = np.dot(a, b.T)
    norm_product = np.linalg.norm(a)*np.linalg.norm(b)
    return dot / norm_product

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
    sorted_samples = sorted(samples, key=lambda x: x[0])

    true_positives = np.sum([sample[1] for sample in samples])
    true_negatives = 0

    best_accuracy, best_threshold = 0,0

    for threshold,label in sorted_samples:
        if label == 0:
            true_negatives += 1
        else:
            true_positives -= 1

        accuracy = (true_positives + true_negatives) / len(samples)
        if accuracy > best_accuracy:
            best_accuracy, best_threshold = accuracy, threshold
            
    return best_threshold, best_accuracy

def similarity_classifier_accuracy(test_negative_pairs, test_positive_pairs, idx_to_vec, use_corr=False):

    positive_pair_similarities = compare_img_pairs(test_positive_pairs, idx_to_vec, use_corr=use_corr)
    negative_pair_similarities = compare_img_pairs(test_negative_pairs, idx_to_vec, use_corr=use_corr)

    similarities = negative_pair_similarities + positive_pair_similarities
    labels = np.concatenate((np.zeros(len(negative_pair_similarities)), np.ones(len(positive_pair_similarities))))
    samples = [(similarities[i],labels[i]) for i in range(len(similarities))]

    best_threshold, best_accuracy = find_highest_accuracy_threshold(samples)

    return best_accuracy