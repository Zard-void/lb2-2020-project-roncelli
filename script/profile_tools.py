import numpy as np
import pandas as pd
import math


def pad_profile(profile, window_size):
    """
    This function adds rows with zero value at the beginning and the end of each profile.

    :param profile: A sequence profile
    :type profile: :class:`pandas.DataFrame`
    :return: the padded profile
    :rtype: :class:`pandas.DataFrame`
    """

    if window_size % 2 == 0:
        raise ValueError('Window size must be an odd integer')
    pad = np.zeros((((window_size - 1) // 2), 20))
    pad = pd.DataFrame(pad, columns=RESIDUES)
    pad['Structure'] = 'X'
    profile = pd.concat([pad, profile, pad])
    profile.reset_index(inplace=True, drop=True)
    return profile


def get_window(profile, index, window_size):
    """
    :param profile: the profile to extract the window from
    :param index: the central residue of the window
    :return: the window
    :rtype: :class:`pandas.DataFrame`
    """
    if window_size % 2 == 0:
        raise ValueError('Window size must be an odd integer')
    offset = window_size // 2
    window = profile[index - offset:index + offset + 1, :20]
    return window


def convert_to_information(gor_model):
    gor_information = gor_model.copy(deep=True)
    for index in gor_information.index:
        for column in gor_information.columns:
            residue = column
            window_position = index[1]
            secondary = index[0]
            joint_residue_secondary = gor_information.loc[index, column]
            marginal_residue = gor_model.loc[('R', window_position), residue]
            marginal_secondary = gor_model.loc[(secondary, 0)].sum()
            gor_information.loc[index, column] = math.log2(joint_residue_secondary /
                                                           (marginal_residue * marginal_secondary))
    return gor_information


def check_columns(profile, training_id):
    profile_columns = list(profile.columns[2:])
    if profile_columns != RESIDUES:
        print(training_id, 'Column Mismatch. Skipping this same.')
        return False
    return True

def infer_window_size(gor_model):
    window_size = gor_model.shape[0] //3
    return window_size

RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


