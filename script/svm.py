import pandas as pd
from joblib import dump, load
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
import re

from sklearn.metrics import matthews_corrcoef, make_scorer

import numpy as np


def average_mcc(y_true, y_pred):
    mcc_list = list()
    for secondary in ['0', '1', '2']:
        y_true_temp = np.array([secondary if str(elem) == secondary else 'O' for elem in y_true])
        y_pred_temp = np.array([secondary if str(elem) == secondary else 'O' for elem in y_pred])
        curr_mcc = matthews_corrcoef(y_true_temp, y_pred_temp)
        mcc_list.append(curr_mcc)
    mcc = np.average(np.array(mcc_list))
    return mcc


vanilla_svc = SVC(verbose=True, random_state=42)

dataset_path = '../data/training/training_full.tsv'

train = pd.read_csv(dataset_path, sep='\t', index_col=[0, 1], header=[0, 1])

train = training_dataset.sample(frac=1, random_state=42)
X_train = train[:, :340].to_numpy()
y_train = train['Class'].to_numpy().ravel()
ps = PredefinedSplit(train['Set']).split()
matthews_score = make_scorer(average_mcc)
scores = cross_validate(vanilla_svc, X_train, y_train,
                        cv=ps,
                        n_jobs=-1,
                        scoring=matthews_score,
                        verbose=10,
                        return_estimator=True,
                        return_train_score=True)

dump(scores, '../models/vanilla_svm.joblib')
