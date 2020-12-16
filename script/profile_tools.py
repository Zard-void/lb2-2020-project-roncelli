import numpy as np
import pandas as pd
import warnings
import os
from tqdm import tqdm
from collections.abc import Iterable
from sklearn.metrics import matthews_corrcoef, accuracy_score
from statistics import mean
from sultan.api import Sultan
from Bio.Blast.Applications import NcbipsiblastCommandline, NcbimakeblastdbCommandline
from tempfile import TemporaryDirectory, NamedTemporaryFile
from joblib import dump

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
        raise Exception(f'Secondary structure must be C, H or E: {secondary} was provided.')
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


def prepare_dataset_from_profiles(ids, profiles_path, window_size, verbosity):
    if verbosity == 0:
        warnings.filterwarnings('ignore')
    if isinstance(ids, Iterable) and not isinstance(ids, str):
        profile_ids = ids
    else:
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
            if window.sum() == 0:
                if verbosity == 2:
                    print('The window indexed at', index, 'of', profile_id, 'is all 0. Discarding the window.')
                skipped_windows += 1
                continue
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
    print('Skipped windows:', skipped_windows)
    return training_full


def prepare_test_from_profiles(ids, profiles_path, window_size, verbosity):
    if verbosity == 0:
        warnings.filterwarnings('ignore')
    if isinstance(ids, Iterable) and not isinstance(ids, str):
        profile_ids = ids
    else:
        with open(ids) as id_file:
            profile_ids = id_file.read().splitlines()
    X_train = list()
    y_train = list()
    indices = list()
    skipped_profiles = 0
    offset = window_size // 2
    number_profiles = 0
    for profile_id in profile_ids:
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
            window = window.flatten().tolist()
            secondary = convert_secondary_to_int(secondary)
            X_train.append(window)
            y_train.append(secondary)
            indices.append((profile_id, index - offset))
        number_profiles += 1
        if number_profiles == 150:
            break
    window_position = [x for x in range(-offset, offset + 1)]
    columns = pd.MultiIndex.from_product([window_position, RESIDUES], names=['Window position', 'Residue'])
    indices = pd.MultiIndex.from_tuples(indices, names=['ID', 'Central index'])
    X_train = pd.DataFrame(X_train, columns=columns, index=indices)
    y_train = pd.Series(y_train, name='Class', index=indices)
    training_full = X_train.copy()
    training_full['Class'] = y_train.astype('category')
    print('Skipped profiles:', skipped_profiles)
    return training_full


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


def average_mcc(y_true, y_pred, benchmark_mode=False):
    mcc_dict = dict()
    mcc_list = list()
    for secondary in np.unique([y_true, y_pred]):
        y_true_temp = [secondary if elem == secondary else 0 for elem in y_true]
        y_pred_temp = [secondary if elem == secondary else 0 for elem in y_pred]
        curr_mcc = matthews_corrcoef(y_true_temp, y_pred_temp)
        mcc_dict[secondary] = curr_mcc
        mcc_list.append(curr_mcc)
    mcc = mean(mcc_list)
    if benchmark_mode:
        mcc_dict['OvR'] = mcc
        return mcc_dict
    return mcc


def average_acc(y_true, y_pred, benchmark_mode=False):
    acc_dict = dict()
    acc_list = list()
    for secondary in np.unique([y_true, y_pred]):
        y_true_temp = [secondary if elem == secondary else 0 for elem in y_true]
        y_pred_temp = [secondary if elem == secondary else 0 for elem in y_pred]
        curr_acc = accuracy_score(y_true_temp, y_pred_temp)
        acc_dict[secondary] = curr_acc
        acc_list.append(curr_acc)
    acc = mean(acc_list)
    if benchmark_mode:
        acc_dict['OvR'] = acc
        return acc_dict
    return acc

def generate_profiles(in_dataframe, out_path):
    dataset = in_dataframe
    s = Sultan()

    print('Unpacking and generating Uniprot DB.')
    s.gunzip('-fk ../data/swiss-prot/uniprot_sprot.fasta.gz').run()
    cmd = NcbimakeblastdbCommandline(input_file='../data/swiss-prot/uniprot_sprot.fasta', dbtype='prot')
    cmd()
    if not os.path.exists(os.path.join(out_path, 'profile')):
        s.mkdir(os.path.join(out_path, 'profile')).run()

    with TemporaryDirectory() as psi_temp:
        for _, sample in tqdm(dataset.iterrows(), total=len(dataset), desc='Generating profiles'):
            with NamedTemporaryFile(mode='w') as blast_in:
                if isinstance(sample.name, tuple):
                    sample_id, chain = sample.name[0], sample.name[1]
                    out_name = f'{sample_id}_{chain}'
                else:
                    sample_id = sample.name
                    out_name = sample_id
                sequence, structure = sample[['Sequence', 'Structure']]
                structure = ' ' + structure
                print(f'>{out_name}', file=blast_in)
                print(sequence, file=blast_in)
                blast_in.seek(0)
                cmd = NcbipsiblastCommandline(query=blast_in.name,
                                              db='../data/swiss-prot/uniprot_sprot.fasta',
                                              evalue=0.01,
                                              num_iterations=3,
                                              out_ascii_pssm=f'{psi_temp}/{out_name}.pssm',
                                              num_descriptions=10000,
                                              num_alignments=10000,
                                            #  out=f'{psi_temp}{out_name}.alns.blast',
                                              num_threads=8)
                cmd()

                if not os.path.exists(os.path.join(psi_temp, out_name + '.pssm')):
                    tqdm.write(f'Unable to generate profile for {out_name}. No hits in the database.')
                    dataset.drop(index=sample.name, inplace=True)
                    continue
                with open(f'{psi_temp}/{out_name}.pssm', 'r') as pssm_file:
                    pssm_file.readline()
                    pssm_file.readline()
                    file_list = []
                    offset = False
                    position = 0
                    for line in pssm_file:
                        line = line.rstrip()
                        if not line:
                            break
                        line = line.split()
                        line.append(structure[position])
                        position += 1
                        if not offset:
                            for i in range(2):
                                line.insert(0, '')
                                offset = True
                        file_list.append(line)
                    df = pd.DataFrame(file_list)
                    df.drop((df.columns[col] for col in range(2, 22)), axis=1, inplace=True)
                    df.drop((df.columns[-3:-1]), axis=1, inplace=True)
                    df.drop((df.columns[0]), axis=1, inplace=True)
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df.rename(columns={df.columns[0]: "Sequence"}, inplace=True)
                    df.rename(columns={df.columns[-1]: "Structure"}, inplace=True)
                    df = df[['Structure'] + [col for col in df.columns if col != 'Structure']]
                    df.loc[:, 'A':'V'] = df.loc[:, 'A':'V'].astype(float).divide(100)
                    df.to_csv(f'{out_path}/profile/{out_name}.profile', sep='\t', index=False)
    print(f'Dumping clean test to data/test/full_test.joblib. Profiles are generated in {out_path}profile.')
    dump(dataset, '../data/test/full_test.joblib')