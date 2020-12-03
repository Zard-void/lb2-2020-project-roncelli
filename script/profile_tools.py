import numpy as np
import pandas as pd
import warnings
import os
from tqdm import tqdm

RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


class ProfileIsZero(Warning):
    """Raised when a sequence profile is all 0."""
    pass


class ColumnMismatch(Warning):
    """Raised when profile columns don't match the expected order."""
    pass


class SecondaryMismatch(Exception):
    """Raised when a secondary character doesn't match an expected possibility."""
    pass


def convert_secondary_to_int(secondary):
    if secondary == 'H':
        secondary = 1
    elif secondary == 'E':
        secondary = 2
    elif secondary == 'C':
        secondary = 3
    else:
        raise Exception('Secondary structure must be C, H or E. ' + secondary + ' was provided.')
    return secondary


def convert_int_to_secondary(secondary):
    if secondary == 1:
        secondary = 'H'
    if secondary == 2:
        secondary = 'E'
    if secondary == 3:
        secondary = 'C'
    return secondary


def check_window(window_size):
    if window_size % 2 == 0:
        raise ValueError('Window size must be an odd integer')
        exit(1)

def pad_profile(profile, window_size):
    """
    This function adds rows with zero values at the beginning and the end of each profile.

    :param profile: A sequence profile
    :param window_size: length of the window: number of total residues before and after the index

    :type profile: :class:`pandas.DataFrame`
    :return: the padded profile
    :rtype: :class:`pandas.DataFrame`
    """
    check_window(window_size)
    pad = np.zeros((window_size // 2, 20))
    pad = pd.DataFrame(pad, columns=RESIDUES)
    pad['Structure'] = 'X'
    profile = pd.concat([pad, profile, pad])
    profile.reset_index(inplace=True, drop=True)
    return profile


def get_window(profile, index, window_size):
    """
    :param profile: the profile to extract the window from
    :param index: the central residue of the window
    :param window_size: length of the window: number of total residues before and after the index
    :return: the window
    :rtype: :class:`pandas.DataFrame`
    """

    check_window(window_size)
    offset = window_size // 2
    window = profile[index - offset:index + offset + 1, :20]
    return window


def check_profile(profile, training_id):
    """
    Performs a series of check to ensure the profiles are suitable for training.
    :param profile:
    :param training_id:
    :return:
    """
    def check_columns():
        """
        Checks that the column order matches the expected order.

        """
        profile_columns = list(profile.columns[2:])
        if profile_columns != RESIDUES:
            warnings.warn(training_id + "The columns of the profile don't match the expected order. Skipping",
                          category=ColumnMismatch)
            return False
        return True

    def check_if_zero():
        """Check is the profile is a non-significant 0 profile.
        """
        if not profile.iloc[:, 2:].any(axis=None):
            warnings.warn(training_id + ": The profile is all 0. Skipping.", category=ProfileIsZero)
            return False
        return True
    return check_if_zero() and check_columns()


def infer_window_size(gor_model):
    """
    :param gor_model: A trained GOR model
    :type gor_model: pandas.DataFrame
    :return: the window size used to train the model
    :rtype int
    """
    n_classes = gor_model.index.levels[0].nunique()
    window_size = gor_model.shape[0] // n_classes
    return window_size


def prepare_query(profile_id, profiles_path, window_size):
    check_window(window_size)
    query = list()
    try:
        profile = pd.read_csv(os.path.join(profiles_path, profile_id + '.profile'), sep='\t')
    except FileNotFoundError:
        print('Profile file not found for', profile_id, '.')
    profile = pad_profile(profile, window_size).to_numpy()
    for index, window in enumerate(profile):
        secondary = profile[index, 20]
        if secondary != 'X':
            window = get_window(profile, index, window_size)
            window = window.flatten().tolist()
            query.append(window)
    query = np.array(query)
    return query


def prepare_training(ids, profiles_path, window_size, verbosity):
    if verbosity == 0:
        warnings.filterwarnings('ignore')
    with open(ids) as id_file:
        profile_ids = id_file.read().splitlines()
    X_train = list()
    y_train = list()
    indices = list()
    skipped_profiles = 0
    skipped_windows = 0
    offset = window_size // 2
    for profile_id in tqdm(profile_ids, desc='Parsing profiles'):
        try:
            profile = pd.read_csv(os.path.join(profiles_path, profile_id + '.profile'), sep='\t')
        except FileNotFoundError:
            print(f'Profile file not found for {profile_id}. Skipping this sample.')
            skipped_profiles += 1
            continue
        if not check_profile(profile, profile_id):
            skipped_profiles += 1
            continue
        profile = pad_profile(profile, window_size).to_numpy()
        for index in range(offset, profile.shape[0] - offset):
            secondary = profile[index, 20]
            window = get_window(profile, index, window_size)
            #if window.sum() == 0:
            #    if verbosity == 2:
            #        print('The window indexed at', index, 'of', profile_id, 'is all 0. Discarding the window.')
            #    skipped_windows += 1
            #    continue
            window = window.flatten().tolist()
            secondary = convert_secondary_to_int(secondary)
            X_train.append(window)
            y_train.append(secondary)
            indices.append((profile_id, index - offset))

    print('Constructing dataset')
    window_position = [x for x in range(-offset, offset + 1)]
    columns = pd.MultiIndex.from_product([window_position, RESIDUES], names=['Window position', 'Residue'])
    indices = pd.MultiIndex.from_tuples(indices, names=['ID', 'Central index'])
    X_train = pd.DataFrame(X_train, columns=columns, index=indices)
    y_train = pd.Series(y_train, name='Class', index=indices)
    training_full = X_train.copy()
    training_full['Class'] = y_train.astype('category')
    print('Skipped profiles:', skipped_profiles)
    #print('Skipped windows:', skipped_windows)
    return training_full