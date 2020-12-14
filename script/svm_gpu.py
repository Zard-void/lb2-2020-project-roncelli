import pandas as pd
from joblib import dump
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from thundersvm import SVC
from sklearn.metrics import make_scorer
from profile_tools import average_mcc

DATASET_PATH = '../jpred_train.csv'
RANDOM_STATE = 42
PARAMETERS = {
    'C': [2, 4],
    'kernel': ['rbf', 'linear', 'polynomial', 'sigmoid'],
    'gamma': [0.5, 2],
    'random_state': [RANDOM_STATE]
}

if __name__ == '__main__':
    svm = SVC(random_state=RANDOM_STATE)
    jpred = pd.read_csv(DATASET_PATH, sep='\t', header=[0, 1], index_col=[0, 1])
    X_train = jpred.iloc[:, :-2].to_numpy()
    y_train = jpred['Class'].to_numpy().ravel()
    ps = PredefinedSplit(jpred['Set']).split()
    matthews_score = make_scorer(average_mcc)
    clf = GridSearchCV(svm,
                       param_grid=PARAMETERS,
                       cv=ps,
                       n_jobs=1,
                       scoring=matthews_score,
                       verbose=10,
                       return_train_score=True,
                       refit=True)

    clf.fit(X_train, y_train)
    dump(clf.cv_results_, '../models/results.joblib', compress=5)
    dump(clf.best_estimator_, '../models/best_estimator.joblib', compress=5)
    dump(clf.best_score_, '../models/best_score.joblib', compress=5)
    dump(clf.scorer_, '../models/scorer.joblib', compress=5)
    print('Done!')
