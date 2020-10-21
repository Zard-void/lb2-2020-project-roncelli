import argparse
import os
import numpy as np
import pandas as pd
from profile_tools import pad_profile, convert_to_information, get_window, check_profile
from tqdm import tqdm


def train(profiles_path, ids, window_size):
    os.chdir(profiles_path)
    gor_model = {secondary: np.zeros((window_size, 20)) for secondary in STRUCTURE_TYPE}
    print('Training')
    for training_id in tqdm(ids, desc='Training'):
        try:
            profile = pd.read_csv(training_id + '.profile', sep='\t')
        except FileNotFoundError:
            print('Profile file not found for', training_id, '. Skipping this sample.')
            continue
        if not check_profile(profile, training_id):
            continue
        profile = pad_profile(profile, window_size).to_numpy()
        for index in range(len(profile)):
            secondary = profile[index, 20]
            if secondary != 'X':
                window = get_window(profile, index, window_size)
                if window.sum() == 0:
                    print('The window indexed at', index, 'of', training_id, 'is all 0. Skipping the window.')
                    continue
                else:
                    gor_model[secondary] = np.add(gor_model[secondary], window)
                    gor_model['R'] = np.add(gor_model['R'], window)

    window_positions = [x for x in range(-(window_size // 2), window_size // 2 + 1)]
    gor_model = pd.concat([pd.DataFrame(v, index=window_positions, columns=RESIDUES) for v in gor_model.values()],
                          axis=0,
                          keys=gor_model.keys(),
                          names=['secondary', 'position'])

    print('Normalizing')
    gor_model = gor_model.divide(gor_model.loc[('R',)].sum(axis=1), axis=0, level=1)
    print('Converting to information matrix')
    gor_model = convert_to_information(gor_model)
    gor_model = gor_model.drop('R',)
    return gor_model


def main(profiles_path, ids, output, window_size):
    with open(ids) as id_file:
        training_ids = id_file.read().splitlines()
        gor_model = train(profiles_path, training_ids, window_size)
        print('Saving to ' + output)
        gor_model.to_csv(output, sep='\t')


parser = argparse.ArgumentParser()
parser.add_argument("--profiles_path", help='directory of data', type=os.path.abspath)
parser.add_argument('--output', help='directory of data', type=os.path.abspath)
parser.add_argument('--ids', help='List of ID')
parser.add_argument('--window_size', help='Length of the window for the training of the GOR model. Defaults to 17',
                    default=17, type=int)
args = parser.parse_args()


STRUCTURE_TYPE = ['H', 'C', 'E', 'R']
RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

main(args.profiles_path,
     args.ids,
     args.output,
     args.window_size
     )
