import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from cornet_S import CORnet_S

def get_model(model_name):
    if model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1] )
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        model.classifier = torch.nn.Identity()
    elif model_name == 'cornet2':
        model = CORnet_S()
        model.decoder = nn.Sequential(*list(model.decoder.children())[:-2])
    else:
        raise NameError
    return model

def get_idx_to_vec(model_name):
    with open('./{}-feature-vecs.json'.format(model_name), 'r') as f:
        idx_to_vec = json.load(f)
    return idx_to_vec

def get_negative_pairs(face_idxs, c_offset_values):
    negative_pairs = []
    for c in c_offset_values:
        for i in face_idxs:
            for j in face_idxs:
                # ensures i != j and that there are no duplicate pairs
                if i > j: 
                    pair_idxs = (1+(5*i),1+(5*j)+c)
                    negative_pairs.append(pair_idxs)
    return negative_pairs

def get_dataset_idxs(face_idxs, c_offset_values, for_training=True):
    positive_pairs = np.array([(1+(5*i),1+(5*i)+c) for c in c_offset_values for i in face_idxs])
    all_negative_pairs = np.array(get_negative_pairs(face_idxs, c_offset_values))
    if for_training:
        negative_pairs = all_negative_pairs[list(np.random.choice(len(all_negative_pairs), face_idxs.shape[0] * len(c_offset_values), replace=False)), :]
    else:
        negative_pairs = all_negative_pairs
    return negative_pairs, positive_pairs

def k_fold_sample(k, train_c_offset_values, test_c_offset_values):
    for i in range(k):
        test_face_idxs = np.arange((400//k)*i, (400//k)*(i+1))
        mask = np.ones(400, dtype=bool)
        mask[test_face_idxs] = False
        train_face_idxs = np.arange(400)[mask]

        assert len((set(train_face_idxs).intersection(test_face_idxs))) == 0

        train_negative_pairs, train_positive_pairs = get_dataset_idxs(train_face_idxs, train_c_offset_values, for_training=True)
        test_negative_pairs, test_positive_pairs = get_dataset_idxs(test_face_idxs, test_c_offset_values, for_training=False)

        yield ((train_negative_pairs, train_positive_pairs), (test_negative_pairs, test_positive_pairs))

def get_test_pairs(all_sizes_test_negative_pairs, all_sizes_test_positive_pairs, c, num_different_sizes):
    """
    Filters test set to only include the pairs of a specific relative size
    """
    offset_neg = all_sizes_test_negative_pairs.shape[0]//num_different_sizes
    offset_pos = all_sizes_test_positive_pairs.shape[0]//num_different_sizes
    test_positive_pairs = all_sizes_test_positive_pairs[c*offset_pos:(c+1)*offset_pos]
    test_negative_pairs = all_sizes_test_negative_pairs[c*offset_neg:(c+1)*offset_neg]
    test_negative_pairs = test_negative_pairs[list(np.random.choice(len(test_negative_pairs), len(test_positive_pairs), replace=False)), :]

    return test_negative_pairs, test_positive_pairs

if __name__ == "__main__":

    for train_pairs, test_pairs in k_fold_sample(2, range(5), range(5)):
        break