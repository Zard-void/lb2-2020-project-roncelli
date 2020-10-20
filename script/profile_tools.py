import numpy as np
import pandas as pd
import math
import warnings


class ProfileIsZero(Warning):
    """Raised when a sequence profile is all 0"""
    pass


class ColumnMismath(Warning):
    """Raised when profile columns don't match the expected order"""
    pass


class GorModel(pd.DataFrame):
    def __init__(self):
        self.trained = False
    pass

def convert_secondary_to_int(secondary):
    if secondary == 'C':
        secondary = 0
    elif secondary == 'H':
        secondary = 1
    elif secondary == 'E':
        secondary = 2
    else:
        raise Exception('Secondary structure must be C, H or E. ' + secondary + ' was provided.')
    return secondary


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


def check_profile(profile, training_id):
    def check_columns():
        profile_columns = list(profile.columns[2:])
        if profile_columns != RESIDUES:
            warnings.warn(training_id + "The columns of the profile don't match the expected order. Skipping",
                          category=ColumnMismath)
            return False
        return True

    def check_if_zero():
        if not profile.iloc[:, 2:].any(axis=None):
            warnings.warn(training_id + ": The given training instance is all 0. Skipping.", category=ProfileIsZero)
            return False
        return True
    return check_if_zero() and check_columns()


def infer_window_size(gor_model):
    window_size = gor_model.shape[0] // 3
    return window_size


RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
