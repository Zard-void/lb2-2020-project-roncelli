from gorpredictor import GORModel
import argparse
import os
from profile_tools import prepare_training, check_window


def main(profiles_path, ids, output, window_size, verbosity):
    check_window(window_size)

    train_dataset = prepare_training(ids, profiles_path, window_size, verbosity)
    X_train = train_dataset.drop(columns='Class').to_numpy()
    y_train = train_dataset['Class'].to_numpy().ravel()
    gor_predictor = GORModel(window_size)
    gor_predictor.fit(X_train, y_train)
    print(f'Saving to {output}')
    gor_predictor.information_tab_.to_csv(output, sep='\t')
    exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_path", help='Folder where the profiles are stored.', type=os.path.abspath)
    parser.add_argument('--output', help='Output path for the trained model.', type=os.path.abspath)
    parser.add_argument('--ids', help='Plain file with the training IDs.', type=os.path.abspath)
    parser.add_argument('--window_size', help='Length of the window used for training. Defaults to 17',
                        default=17, type=int)
    parser.add_argument('--verbosity', help='Verbosity of output. 0: only errors, 1: error and severe warnings, '
                                            '2: all errors and warnings.',
                        default=1,
                        type=int)
    args = parser.parse_args()
    main(args.profiles_path,
         args.ids,
         args.output,
         args.window_size,
         args.verbosity
         )
