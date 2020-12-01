import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def _normalize(raw_model):
    """Normalizes each observation in the input model. The new ranges will be between 0 and 1.

    Parameters
    ----------

    raw_model : pandas dataframe
        The model to normalize.

    Returns
    -------
    raw_model : pandas dataframe
        The normalized model.
    """
    return raw_model.divide(raw_model.loc[('R',)].sum(axis=1), axis=0, level=1)


def _convert_to_information(raw_model, ss_prob):
    """
    Constructs the information matrix base on the model in input.

    Parameters
    ----------
    raw_model : pandas dataframe
        The model to convert.

    ss_prob : dict
        The dictionary that reports the marginal probabilities of each secondary structure observed.

    Returns
    -------
    information : dict
        The trained model as an information matrix.
    """

    information = raw_model.copy(deep=True)
    for index in information.index:
        for column in information.columns:
            residue = column
            window_position = index[1]
            secondary = index[0]
            if secondary == 'R':
                continue
            joint_residue_secondary = information.at[index, column]
            marginal_residue = raw_model.loc[('R', window_position), residue]
            marginal_secondary = ss_prob[secondary]
            information.at[index, column] = math.log2(joint_residue_secondary /
                                                      (marginal_residue * marginal_secondary))
    information = {secondary: information.xs((secondary,), level=0).to_numpy()
                   for secondary in information.index.levels[0]
                   if secondary != 'R'}
    return information


class GORModel(BaseEstimator, ClassifierMixin):
    """Performs secondary structure prediction based on Garnier-Osguthorpe-Robson method.

    Parameters
    ----------
    window_size : {int}, default=17
        Specifies the length of the sliding window used for the prediction.
        Reflects the influence of distant residues in determining the secondary structure
        of the central residue. Must be an odd integer.

    Attributes
    ----------
    information_ : dict of nparrays of with length equal to the number of different secondary structures.
        The trained information matrix which will be used for the prediction
        of an unknown sequence
    """
    def __init__(self, window_size=17):
        self.window_size = window_size
        self.is_fitted_ = False

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
            n_features bust be a multiple of the window size
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
        """
        self.classes_ = unique_labels(y)
        X, y = check_X_y(X, y)
        ss_count = {label: 0 for label in self.classes_}
        total_count = 0
        raw_model = {secondary: np.zeros((self.window_size, 20)) for secondary in self.classes_}
        raw_model['R'] = np.zeros((self.window_size, 20))

        for sample_n in tqdm(range(len(X)), desc='Training'):
            window = X[sample_n]
            secondary = y[sample_n]
            window = np.reshape(window, (self.window_size, 20))
            ss_count[secondary] += 1
            total_count += 1
            raw_model[secondary] = np.add(raw_model[secondary], window)
            raw_model['R'] = np.add(raw_model['R'], window)

        ss_prob = {secondary: ss_count[secondary] / total_count for secondary in ss_count.keys()}
        window_positions = [x for x in range(-(self.window_size // 2), self.window_size // 2 + 1)]
        raw_model = pd.concat(
            [pd.DataFrame(v, index=window_positions, columns=RESIDUES) for v in raw_model.values()],
            axis=0,
            keys=raw_model.keys(),
            names=['secondary', 'position'])

        _normalize(raw_model)
        self.information_ = _convert_to_information(raw_model, ss_prob)
        self.information_tab_ = pd.concat(
            [pd.DataFrame(v, index=window_positions, columns=RESIDUES) for v in self.information_.values()],
            axis=0,
            keys=self.information_.keys(),
            names=['secondary', 'position'])
        self.is_fitted_ = True
        return self

    def load_model(self, model):
        if os.path.isfile(model):
            model = pd.read_csv(model, sep='\t', index_col=[0, 1]).sort_index()
        self.information_ = {secondary: model.xs((secondary,), level=0).to_numpy()
                             for secondary in model.index.levels[0]
                             }
        self.is_fitted_ = True

    def predict(self, X, batch_mode=True):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

        Returns
        -------
        predicted_structures : ndarray of shape (n_samples,)
        Class labels for samples in X.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        predicted_structure = list()
        if batch_mode:
            for sample_n in tqdm(range(X.shape[0]), desc='Predicting'):
                window = X[sample_n]
                window = np.reshape(window, (self.window_size, 20))
                probabilities = {secondary: (self.information_[secondary] * window).sum()
                                 for secondary in self.information_.keys()}
                predicted_structure.append(max(probabilities, key=probabilities.get))
            predicted_structure = np.array(predicted_structure)
        else:
            for sample_n in range(X.shape[0]):
                window = X[sample_n]
                window = np.reshape(window, (self.window_size, 20))
                probabilities = {secondary: (self.information_[secondary] * window).sum()
                                 for secondary in self.information_.keys()}
                predicted_structure.append(max(probabilities, key=probabilities.get))
        return predicted_structure
