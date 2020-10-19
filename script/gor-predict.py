import pandas as pd
import os
from tqdm import tqdm
from profile_tools import pad_profile, get_window, infer_window_size
import argparse


window_size = 17
os.chdir('/home/stefano/PycharmProjects/lb2-2020-project-roncelli/data/training/')


def predict(gor_model, query):
    query = pd.read_csv(query, sep='\t')
    query = pad_profile(query, window_size).to_numpy()
    predicted_structure = list()
    for index in range(len(query)):
        secondary = query[index, 20]
        if secondary != 'X':
            window = get_window(query, index, window_size)
            probabilities = {secondary: (gor_model[secondary] * window).sum()
                             for secondary in gor_model.keys()}
            predicted_structure.append((max(probabilities, key=probabilities.get)))
    predicted_structure = ''.join(predicted_structure)
    return predicted_structure


trained_model = pd.read_csv('gor_model_normalized_SCRIPT.tsv', sep='\t', index_col=[0, 1]).sort_index()
window_size = infer_window_size(trained_model)
trained_model = {secondary: trained_model.xs((secondary, ), level=0).to_numpy()
                 for secondary in trained_model.index.levels[0]
                 }

print(window_size)
for item in tqdm(os.listdir('./profile')):
    with open('./profile/' + item) as handle:
        ss = predict(trained_model, handle)
