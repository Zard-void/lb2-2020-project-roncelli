from joblib import load
from sklearn.metrics import make_scorer, classification_report, accuracy_score
from profile_tools import average_mcc, average_acc
import pandas as pd
import numpy as np
from joblib import dump


svc = load('../models/best_estimator.joblib')
the_blind = load('../data/test/the_blind.joblib')
X_test = the_blind.iloc[:, :-1]
y_test = the_blind['Class'].to_numpy().ravel()
matth_coeff = make_scorer(average_mcc)
prediction_svc = svc.predict(X_test)
dump(prediction_svc, '../data/test/prediction_svm.joblib', compress=5)
predicted_svm = load('../data/test/prediction_svm.joblib')

report_gor = classification_report(y_true=y_test, y_pred=predicted_svm, target_names=['H', 'E', 'C'], labels=[1, 2, 3], output_dict=True)
report_gor = pd.DataFrame.from_dict(report_gor, orient='columns')
report_gor = report_gor.drop(index=['f1-score', 'support'], columns=['accuracy', 'weighted avg'])
report_gor.columns = ['H', 'E', 'C', 'OvR']

acc = average_acc(y_test, predicted_svm, benchmark_mode=True)
acc = pd.DataFrame.from_dict(acc, orient='index').T
acc.columns = ['H', 'E', 'C', 'OvR']
acc.index = ['Accuracy']

mcc = average_mcc(y_test, predicted_svm, benchmark_mode=True)
mcc = pd.DataFrame.from_dict(mcc, orient='index').T
mcc.columns = ['H', 'E', 'C', 'OvR']
mcc.index = ['Matthews Correlation Coefficient']

report_gor = report_gor.append(acc)
report_gor = report_gor.append(mcc)
report_gor['Ser'] = report_gor.iloc[:, 0:-1 ].sem(axis=1)

report_gor.to_latex(buf='../report/tables/svm_blind.txt', float_format="\\num{%f}", escape=False)
print(report_gor)
