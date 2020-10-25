import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from profile_tools import pad_profile, get_window, convert_secondary_to_int, check_profile
from CVsplit import add_splits

WINDOW_SIZE = 17
WINDOW_POSITION = [x for x in range(-(WINDOW_SIZE // 2), WINDOW_SIZE // 2 + 1)]
RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
X_train = list()
y_train = list()
indices = list()

profiles_folder = '../data/training/profile/'

for profile_file in tqdm(os.listdir(profiles_folder)):
    profile = pd.read_csv(profiles_folder + profile_file, sep='\t')
    if not check_profile(profile, profile_file):
        continue
    profile = pad_profile(profile, WINDOW_SIZE).to_numpy()
    for index in range(len(profile)):
        secondary = profile[index, 20]
        if secondary != 'X':
            window = get_window(profile, index, WINDOW_SIZE)
            if window.sum() == 0:
                print('The window indexed at', index, 'of', profile_file, 'is all 0. Discarding the window.')
                continue
            window = window.flatten().tolist()
            indices.append(('.'.join(profile_file.split('.')[:-1]), index - 8))
            X_train.append(window)
            secondary = convert_secondary_to_int(secondary)
            y_train.append(secondary)

columns = pd.MultiIndex.from_product([WINDOW_POSITION, RESIDUES], names=['Window position', 'Residue'])
indices = pd.MultiIndex.from_tuples(indices, names=['ID', 'Central index'])

X_train = pd.DataFrame(X_train, columns=columns, index=indices)
y_train = pd.Series(y_train, name='Class', index=indices)

#training_full = X_train.copy()
#training_full['Class'] = y_train.astype('category')

training_full = pd.concat([X_train, y_train], axis=1)

print('Generating dataframe')
print('Adding CV set id to samples')
training_full = add_splits(training_full, '../data/training/cv/')

print('Saving Dataframe as tsv (this may take a while).')

training_full.to_csv('../data/training/training_sage.csv')

print('Done!')
