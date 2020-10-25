import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer

vanilla_svc = SVC(verbose=True)

dataset_path = '../data/training/training_full.tsv'


training_dataset = pd.read_csv(dataset_path, sep='\t', index_col=[0, 1], header=[0, 1])

train_20 = training_dataset.sample(frac=0.1)
X_train = train_20.iloc[:, :340].to_numpy()
y_train = train_20['Class'].to_numpy().ravel()
ps = PredefinedSplit(train_20['Set']).split()
matthews_score = make_scorer(matthews_corrcoef)
scores = cross_validate(vanilla_svc, X_train, y_train,
                        cv=ps,
                        n_jobs=-1,
                        scoring=matthews_score,
                        verbose=10,
                        return_estimator=True,
                        return_train_score=True)

#dump(scores, '../models/vanilla_csv.joblib')
