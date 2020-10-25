import pandas as pd
import os
from tqdm import tqdm
import numpy as np


def split(ids_files_path, full_dataset_path):
    k_folds = dict()
    for file_name in sorted(os.listdir(ids_files_path)):
        print(file_name)
        with open(ids_files_path + file_name) as cv_fold:
            k_folds[file_name] = cv_fold.read().splitlines()
    print('Loading full dataset')
    training_full = pd.read_csv(full_dataset_path, sep='\t', index_col=[0,1], header=0)
    print('Creating k-folds')
    k_folds = {set: {'train': training_full[training_full.index.get_level_values(0).isin(k_folds[set])],
                     'test': training_full[training_full.index.get_level_values(0).isin(k_folds[set]) == False]}
               for set in k_folds.keys()
               }

    return k_folds


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
