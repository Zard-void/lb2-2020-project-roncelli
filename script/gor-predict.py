import pandas as pd
import os
from tqdm import tqdm
from profile_tools import infer_window_size, prepare_query
from gorpredictor import GORModel
import argparse


def main(ids, profiles_path, model_path):
    with open(ids) as id_file:
        profile_ids = id_file.read().splitlines()
    trained_model = pd.read_csv(model_path, sep='\t', index_col=[0, 1]).sort_index()
    window_size = infer_window_size(trained_model)
    gor_estimator = GORModel(window_size)
    gor_estimator.load_model(model_path)
    for profile_id in profile_ids:
        query = prepare_query(profile_id, profiles_path, window_size)
        ss_pred = gor_estimator.predict_single(query)
        print(f'>{profile_id}\n{ss_pred}')
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help='directory of the trained model. Must be a tab separated file.',
                        type=os.path.abspath)
    parser.add_argument("--profiles_path", help='directory of data', type=os.path.abspath)
    parser.add_argument('--ids', help='List of ID', type=os.path.abspath)
    args = parser.parse_args()
    main(args.ids,
         args.profiles_path,
         args.model_path
         )
