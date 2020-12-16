import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


RESIDUES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


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

    for secondary in raw_model.keys():
        raw_model[secondary] /= raw_model['R'].sum(axis=1).reshape(-1, 1)
    for secondary in ss_prob.keys():
        raw_model[secondary] /= raw_model['R'] * ss_prob[secondary]
        raw_model[secondary] = np.log2(raw_model[secondary])
    raw_model.pop('R')
    return raw_model


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

        for window, secondary in tqdm(zip(X, y), desc='Training', total=len(X)):
            window = window.reshape(self.window_size, 20)
            ss_count[secondary] += 1
            total_count += 1
            raw_model[secondary] += window
            raw_model['R'] += window

        ss_prob = {secondary: ss_count[secondary] / total_count for secondary in ss_count.keys()}
        window_positions = [x for x in range(-(self.window_size // 2), self.window_size // 2 + 1)]

        self.information_ = _convert_to_information(raw_model, ss_prob)
        self.information_tab_ = pd.concat(
            [pd.DataFrame(v, index=window_positions, columns=RESIDUES) for v in self.information_.values()],
            axis=0,
            keys=self.information_.keys(),
            names=['Structure', 'Position'])
        self.is_fitted_ = True
        return self

    def load_model(self, model):
        """Loads a trained model.

        Parameters
        ----------
        model : {path or pd.Dataframe}
            The trained model. Can be either a path to a tab separated file or a pandas dataframe.
        """
        if os.path.isfile(model):
            model = pd.read_csv(model, sep='\t', index_col=[0, 1]).sort_index()
        self.information_ = {secondary: model.xs((secondary,), level=0).to_numpy()
                             for secondary in model.index.levels[0]
                             }
        self.is_fitted_ = True

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

        batch_mode : {bool}
            If false, the model expects as input a single protein. Useful for generating sequences of unknown proteins.
            If true, the model expects as input a two or more proteins in the same file.
            Useful for scoring on sets when the actual predicted sequence is not important.

        Returns
        -------
        predicted_structure : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        predicted_structure = list()
        for sample_n in tqdm(range(X.shape[0]), desc='Predicting'):
            window = X[sample_n]
            if window.sum() == 0:
                predicted_structure.append(3)
                continue
            window = np.reshape(window, (self.window_size, 20))
            probabilities = {secondary: (self.information_[secondary] * window).sum()
                             for secondary in self.information_.keys()}
            predicted_structure.append(max(probabilities, key=probabilities.get))
        predicted_structure = np.array(predicted_structure)
        return predicted_structure

    def predict_single(self, X):
        predicted_structure = list()
        for window in X:
            print(window.sum())
            if window.sum() == 0:
                predicted_structure.append(3)
                print('hey!')
                continue
            window = window.reshape(self.window_size, 20)
            probabilities = {secondary: (self.information_[secondary] * window).sum()
                             for secondary in self.information_.keys()}
            predicted_structure.append(str(max(probabilities, key=probabilities.get)))
        predicted_structure = ''.join(predicted_structure)
        return predicted_structure