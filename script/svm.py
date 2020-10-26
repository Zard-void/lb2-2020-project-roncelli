import pandas as pd
from joblib import dump
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, make_scorer
import numpy as np

def average_mcc1(y_true, y_pred):
    """
    A scorer function to be used for grid search evaluation.
    Returns the average matthews_corrcoef among the 3 classes of ss.
    """

    mcc_list = []
    trans_dict = {}

    for class_label in CLASS_LABELS:
        for my_label in CLASS_LABELS:
            if my_label == class_label:
                trans_dict[str(my_label)] = str(my_label)
            else:
                trans_dict[str(my_label)] = "O"

        y_true_temp = np.array([trans_dict[str(label)] for label in y_true])
        y_pred_temp = np.array([trans_dict[str(label)] for label in y_pred])
        curr_mcc = matthews_corrcoef(y_true_temp, y_pred_temp)
        mcc_list.append(curr_mcc)
    mcc_avg = np.average(np.array(mcc_list))

    return mcc_avg

def average_mcc(y_true, y_pred):
    mcc_list = list()
    for secondary in ['0', '1', '2']:
        y_true_temp = np.array([secondary if str(elem) == secondary else 'O' for elem in y_true])
        y_pred_temp = np.array([secondary if str(elem) == secondary else 'O' for elem in y_pred])
        curr_mcc = matthews_corrcoef(y_true_temp, y_pred_temp)
        mcc_list.append(curr_mcc)
    mcc = np.average(np.array(mcc_list))
    return mcc


CLASS_LABELS = [0, 1, 2]
DATASET_PATH = '../data/training/training_full.tsv'
RANDOM_STATE = 42
PARAMETERS = {
    'C': [2, 4],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': [0.5, 2],
    'random_state': [RANDOM_STATE]
}

clf = SVC(random_state=RANDOM_STATE)


train = pd.read_csv(DATASET_PATH, sep='\t', index_col=[0, 1], header=[0, 1])
#train = train.sample(frac=0.005, random_state=42)

X_train = train.iloc[:, :340].to_numpy()
y_train = train['Class'].to_numpy().ravel()
ps = PredefinedSplit(train['Set']).split()

matthews_score = make_scorer(average_mcc)

grid_search_clf = GridSearchCV(
    clf,
    param_grid=PARAMETERS,
    scoring=matthews_score,
    cv=ps,
    return_train_score=True,
    refit=True,
    n_jobs=-1,
    verbose=10
    )

grid_search_clf.fit(X_train, y_train)
dump(grid_search_clf.best_estimator_, '../models/svc_best_estimator.joblib')
dump(grid_search_clf.cv_results_, '../models/svc_cv_results.joblib')
dump(grid_search_clf.best_score_, '../models/svc_best_score.joblib')


