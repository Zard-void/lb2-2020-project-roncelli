from joblib import load
from gorpredictor import GORModel
from sklearn.model_selection import cross_validate, PredefinedSplit
from sklearn.metrics import make_scorer, classification_report, accuracy_score
from profile_tools import average_mcc, average_acc
import pandas as pd


jpred = load('../data/training/jpred.joblib')
the_blind = load('../data/test/the_blind.joblib')

X_train = jpred.iloc[:, :-2]
y_train = jpred['Class']
ps = PredefinedSplit(jpred['Set'])
X_test = the_blind.iloc[:, :-1]
y_test = the_blind['Class']
matth_coeff = make_scorer(average_mcc)

gor = GORModel(17)
cv_gor = cross_validate(gor, X_train, y_train,  cv=ps, scoring=matth_coeff, return_train_score=True)
gor.fit(X_train, y_train)
predicted_gor = gor.predict(X_test)

report_gor = classification_report(y_true=y_test, y_pred=predicted_gor, target_names=['H', 'E', 'C'], labels=[1,2,3], output_dict=True)
report_gor = pd.DataFrame.from_dict(report_gor, orient='columns')
report_gor = report_gor.drop(index=['f1-score', 'support'], columns=['accuracy', 'weighted avg'])
report_gor.columns = ['H', 'E', 'C', 'OvR']

acc = average_acc(y_test, predicted_gor, benchmark_mode=True)
acc = pd.DataFrame.from_dict(acc, orient='index').T
acc.columns = ['H', 'E', 'C', 'OvR']
acc.index = ['Accuracy']

mcc = average_mcc(y_test, predicted_gor, benchmark_mode=True)
mcc = pd.DataFrame.from_dict(mcc, orient='index').T
mcc.columns = ['H', 'E', 'C', 'OvR']
mcc.index = ['Matthews Correlation Coefficient']

report_gor = report_gor.append(acc)
report_gor = report_gor.append(mcc)
report_gor['Ser'] = report_gor.iloc[:, 0:-1].sem(axis=1)

report_gor.to_latex(buf='../report/tables/gor_blind.txt', float_format="\\num{%f}", escape=False)






