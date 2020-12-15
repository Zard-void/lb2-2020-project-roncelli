import os
from profile_tools import prepare_test_from_profiles
from random import sample
from joblib import dump

WINDOW_SIZE = 17


if __name__ == '__main':
    profiles_ids = [sample[:-8] for sample in os.listdir('../data/test/profile')]
    tolerance = 0.02
    train_values = {1: 77742, 2: 48588, 3: 92091}
    train_values = {ss: train_values[ss] / sum(train_values.values()) for ss in train_values.keys()}
    while True:
        the_150 = sample(profiles_ids, k=200)
        test = prepare_test_from_profiles(the_150, '../data/test/profile/', WINDOW_SIZE, 0)
        structure = test['Class'].astype(str).str.cat(sep='')
        structure = [int(ss) for ss in structure]
        test_values = {ss: structure.count(ss) for ss in structure}
        test_values = {ss: test_values[ss] / sum(test_values.values()) for ss in test_values.keys()}
        for ss in train_values.keys():
            difference = abs(train_values[ss] - test_values[ss])
            if difference > tolerance:
                print(f'Difference between train and test of {ss} is {difference}, resampling')
                break
        else:
            print('All classes are within the tolerance threshold. Done!')
            break
    print('Dumping the test set to data/test/the_blind_w17.joblib')
    dump(test, '../data/test/the_blind_w17.joblib', compress=5)
