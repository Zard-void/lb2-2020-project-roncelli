import pandas as pd
import numpy as np


class GorModel(pd.DataFrame):
    _metadata = ['istrained']

    def __init__(self, window_size, *args, **kw):
        super(GorModel, self).__init__(*args, **kw)
        self.window_size = window_size
        self.istrained = False

    @property
    def _constructor(self):
        return GorModel

    def fit(self, X, y):
        STRUCTURE_TYPE = [0, 1, 2, 3]

        model_as_dict = {secondary: np.zeros((self.window_size, 20)) for secondary in STRUCTURE_TYPE}

        X = X.to_numpy()
        y = y.to_numpy()
        for index in range(X.shape[0]):
            sample = X[index]
            secondary = y[index, 0]
            sample = sample.reshape(self.window_size, 20)
            model_as_dict[secondary] = np.add(model_as_dict[secondary], sample)
            model_as_dict[3] = np.add(model_as_dict[3], sample)


X_train = pd.read_csv('../data/training/X_train.tsv', sep='\t', index_col=[0, 1], header=[0, 1])
y_train = pd.read_csv('../data/training/y_train', sep='\t', index_col=0)
my_model = GorModel(window_size=17)

my_df = pd.DataFrame()

my_model.fit(X_train, y_train)
