
import pandas as pd

from preprocessing_utils import *
import implementation_0
import implementation_1
import implementation_2

def main():
    """
    Load idx_to_vec dictionaries for each model

    For each model:
        Split into train and test
        For each CV split:
            # Implementation 0
            For each test relative size ratio:
                Run implementation 0 to get accuracy of test set same/different classification
            
            # Implementation 1
            Train implementation 1 same/different classifier and 
            For each test pair size bin:
                run on test set to get accuracy of same/different classification

            # Implementation 2
            Train implementation 2 multiclass classifier.
            For each test pair size bin:
                Use feature vector to get same/different classification accuracy
    """

    k = 8
    k_fold = 0
    model_names = ['resnet101', 'vgg19', 'densenet169', 'cornet2']
    accuracies = {model_name:{'implementation_0':{}, 'implementation_1':{}, 'implementation_2':{}} for model_name in model_names}
    
    c_train_offset_values = [0,1,2,3,4]
    c_test_offset_values = [0,1,2,3,4]

    for model_name in model_names:
        idx_to_vec = get_idx_to_vec(model_name)
        for train_pairs, test_pairs in k_fold_sample(k, c_train_offset_values, c_test_offset_values):
            train_negative_pairs, train_positive_pairs = train_pairs[0], train_pairs[1]
            for i,implementation in [(0, implementation_0), (1,implementation_1)]:
                print("Model: {}, k_fold: {}, Implementation: {}".format(model_name, k_fold, i))

                # Implementation 0: Feature Vector Cosine Similarity Classifier
                # Implementation 1: SVM Classifier Pair Feature Vector Difference as input 
                # Implementation 2: MLP with multiclass output and Feature Vector Cosine Similarity Classifier for new points
                model = implementation.train(train_negative_pairs, train_positive_pairs, idx_to_vec)
                for c in range(len(c_test_offset_values)):
                    test_negative_pairs, test_positive_pairs = get_test_pairs(test_pairs[0], test_pairs[1], c, len(c_test_offset_values))
                    implementation_accuracy = implementation.get_classification_accuracy(model, test_negative_pairs, test_positive_pairs, idx_to_vec)
                    c_accuracy_list = accuracies[model_name]['implementation_{}'.format(i)].get(c, []) + [implementation_accuracy]
                    accuracies[model_name]['implementation_{}'.format(i)][c] = c_accuracy_list
                    print(c, implementation_accuracy)
        k_fold += 1

    return accuracies


if __name__ == '__main__':
    accuracies = main()
    print(accuracies)
    with open('./accuracies_k={}.json'.format(8), 'w') as f:
        json.dump(accuracies, f)

    # pd.set_option('display.max_colwidth', -1)
    # data = json.load(open('accuracies_k=3.json'))
    # df = pd.concat({k: pd.DataFrame(v,) for k, v in data.items()}, axis=0)
    # print(df.round(3))