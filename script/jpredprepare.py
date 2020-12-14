from profile_tools import prepare_dataset_from_profiles, add_splits
from joblib import dump


if __name__ == '__main__':
    ID_FILE = '../data/training/training_ids.txt'
    SPLIT_PATH = '../data/cv/'
    PROFILE_PATH = '../data/training/profile/'
    training_full = prepare_dataset_from_profiles(ID_FILE, PROFILE_PATH, 17, 0)
    training_full = add_splits(training_full, SPLIT_PATH)
    print('Dumping Dataframe (this may take a while).')
    dump(training_full, '../data/training/jpred.joblib', compress=5)
    print('Done!')
