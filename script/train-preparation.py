import os
import pandas as pd
from tqdm import tqdm
from profile_tools import pad_profile, get_window, convert_secondary_to_int


WINDOW_SIZE = 17
WINDOW_POSITION = [x for x in range(-(WINDOW_SIZE // 2), WINDOW_SIZE // 2 + 1)]
RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
X_train = list()
y_train = list()
indices = list()

os.chdir('/home/stefano/PycharmProjects/lb2-2020-project-roncelli/data/training/profile')
for profile_file in tqdm(os.listdir()):
    profile = pd.read_csv(profile_file, sep='\t')
    profile = pad_profile(profile, WINDOW_SIZE).to_numpy()
    for index in range(len(profile)):
        secondary = profile[index, 20]
        if secondary != 'X':
            window = get_window(profile, index, WINDOW_SIZE)
            window = window.flatten().tolist()
            indices.append((''.join(profile_file.split('.')[:-1]), index - 8))
            X_train.append(window)
            secondary = convert_secondary_to_int(secondary)
            y_train.append(secondary)

columns = pd.MultiIndex.from_product([WINDOW_POSITION, RESIDUES], names=['window', 'residue'])
indices = pd.MultiIndex.from_tuples(indices, names=['ID', 'position'])
#X_train = pd.DataFrame(X_train, columns=columns, index=indices)
y_train = pd.Series(y_train)
y_train.to_csv('../y_train', sep='\t')
print('Saving Dataframe as tsv (this may take a while).')
#X_train.to_csv('../X_train.tsv', sep='\t')
print('Done!')
