import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC

def pad_profile(profile, window_size):
    """
    This function adds rows with zero value at the beginning and the end of each profile.

    :param profile: A sequence profile
    :type profile: :class:`pandas.DataFrame`
    :param window_size: the length of the window
    :type window_size: int
    :return: the padded profile
    :rtype: :class:`pandas.DataFrame`
    """

    columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
               'Structure']
    pad = np.zeros((((window_size - 1) // 2), 21))
    pad = pd.DataFrame(pad, columns=columns)
    pad['Structure'] = 'X'
    profile = pd.concat([pad, profile, pad])
    profile.reset_index(inplace=True, drop=True)
    return profile


def get_window(profile, index, window_size):
    """
    :param profile: the profile to extract the window from
    :param index: the central residue of the window
    :param window_size: the size of the window
    :return: the window
    :rtype: :class:`pandas.DataFrame`
    """

    before_offset = (window_size - 1) // 2
    after_offset = window_size // 2
    window = profile[index - before_offset:index + after_offset + 1, :20]
    return window

def convert_secondary_to_int(secondary):
    if secondary == 'C':
        secondary = 0
    elif secondary == 'H':
        secondary = 1
    elif secondary == 'E':
        secondary = 2
    else:
        raise Exception 'Secondary structure must be C, H or E. ' + print(secondary) + ' was provided.'
    return secondary
WINDOW_SIZE = 17

X_train = list()
y_train = list()
indices = list()

os.chdir('/home/stefano/PycharmProjects/lb2-2020-project-roncelli/data/training/profile')
for profile_file in tqdm(os.listdir()):
    profile = pd.read_csv(profile_file, sep='\t')
    profile = pad_profile(profile, window_size).to_numpy()
    for index in range(len(profile)):
        secondary = profile[index, 20]
        if secondary != 'X':
            window = get_window(profile, index, window_size)
            window = window.flatten().tolist()
            indices.append((''.join(profile_file.split('.')[:-1]), index - 8))
            X_train.append(window)
            secondary = convert_secondary_to_int(secondary)
            y_train.append(secondary)
WINDOW_POSITION = [x for x in range(-(window_size // 2), window_size // 2 + 1)]
RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

columns = pd.MultiIndex.from_product([WINDOW_POSITION, RESIDUES], names=['window', 'residue'])
indices = pd.MultiIndex.from_tuples(indices, names=['ID', 'position'])
X_train = pd.DataFrame(X_train, columns=columns, index=indices)
X_train.to_csv('../X_train.tsv',sep='\t')
