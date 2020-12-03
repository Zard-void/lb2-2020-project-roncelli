import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from profile_tools import prepare_training


def add_splits(dataset, ids_folder_path):
    folds = dict()
    for set_name in sorted(os.listdir(ids_folder_path)):
        with open(ids_folder_path + set_name) as cv_set:
            sample_ids = cv_set.read().splitlines()
            for sample_id in sample_ids:
                folds[sample_id] = int(set_name[-1])

    folds_df = pd.Series(np.nan, index=dataset.index)

    for index in tqdm(dataset.index, desc='Adding folds'):
        sample_id = index[0]
        folds_df.at[index] = folds[sample_id]

    dataset['Set'] = folds_df.astype('int64').astype('category')
    return dataset

if __name__ == '__main__':
    ID_FILE = '../data/training/training_ids.txt'
    with open(ID_FILE) as f:
        profile_ids = f.read().splitlines()

    profiles_folder = '../data/training/profile/'

    training_full = prepare_training(profile_ids, profiles_folder, 17, 0)
    training_full = add_splits(training_full, '../data/training/cv/')

    print('Saving Dataframe as tsv (this may take a while).')
    training_full.to_csv('../data/training/jpred_train.csv', sep='\t')
    print('Done!')
