"""
**SSVEP FB CCA Classifier**

Classifies SSVEPs using the FBCCA method.

"""

# Stock libraries
import numpy as np
from sklearn.cross_decomposition import CCA


# Import bci_essentials modules and methods
from ..signal_processing import (
    construct_filter_bank,
    implement_filter_bank,
    ssvep_templates
)
from .generic_classifier import GenericClassifier, Prediction
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

class SSVEPFBCCClassifier(GenericClassifier):
    """SSVEP FBCCA Classifier class.

    This class implements the SSVEP FBCCA classifier, which uses
    Canonical Correlation Analysis (CCA) to classify SSVEP signals.
    The classifier is trained using a filter bank approach.

    Attributes
    ----------
    sampling_freq : int
        Sampling frequency of the EEG data.
    target_freqs : list of `int`
        List of the target frequencies for SSVEP detection.
    n_harmonics : int
        Number of harmonics to use in the filter bank.
    n_filters : int
        Number of filters in the filter bank.
    filter_bank : list of `ndarray`
        List of filter coefficients for each target frequency.
    cca : `CCA`
        CCA object used for classification.
    templates : `ndarray`
        Template signals for each target frequency.

    """

    def sef_ssvep_clf_settings(self, fsample, target_freqs, n_harmonics=0, filter_bank=None, filter_order=4):
        """Initialize the SSVEP FBCCA Classifier.

        Parameters
        ----------
        fsample : float
            Sampling frequency of the EEG data [Hz].
        target_freqs : list of `float`
            List of the target frequencies for SSVEP detection [Hz].
        n_harmonics : int, optional
            Number of harmonics to use in the filter bank (default is 0).
        filter_bank :array-like **optional**
            Cutoff frequencies for the filter bank [Hz].
            Should be [n_filters, 2]. Where the first column is the low-cutoff
            frequency and the second column is the high-cutoff frequency.
        filter_order : int, optional
            Order of the filter (default is 4).

        Returns
        -------
        `None`
            
        """
        self.fsample = fsample
        self.filter_bank = filter_bank
        self.target_freqs = target_freqs
        self.n_harmonics = n_harmonics
        self.filter_bank = filter_bank
        self.filter_order = filter_order
        self.templates = None   # SSVEP signal templates
        self.cca_ncomponents = 1  # Number of components for CCA
        

        # Create filter bank
        if filter_bank:
            self.fb_coefficients = construct_filter_bank(
                self.filter_bank,
                self.fsample,
                self.filter_order
            )

        # Create templates for each target frequency
        self.templates = ssvep_templates(
            target_freqs, fsample, n_harmonics
        )
        

        # Initialize CCA objects for each target frequency
        # TODO n_components is a hyper-parameter that might need tuning
        self.ccas = []
        for _ in range(len(target_freqs)):
            self.ccas.append(CCA(n_components=self.cca_ncomponents))
        

    def fit(self):
        """Fit the model.

        This method is not used in the FBCCA classifier, as it does not require
        training.

        Returns
        -------
        `None`
            Models created used in `predict()`.

        """
        pass


    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            3D array where shape = (trials, channels, samples)

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels.  Probabilities
            are not available (empty list).

        """
        # Create signal templates for each target frequency
        # This is only done once to speed up the process
        if self.templates:
            self.templates = ssvep_templates(
                self.target_freqs, self.fsample, self.n_harmonics
            )

        # If the nsamples of X changes, we need to re-compute the templates
        if X.shape[-1] != self.templates.shape[-1]:
            self.templates = ssvep_templates(
                self.target_freqs, self.fsample, self.n_harmonics
            )

        # Filter the signal if necessary
        if self.filter_bank:
            X = implement_filter_bank(X, self.fb_coefficients)

        # Compute CCA correlations 
        correlations = np.zeros(len(self.ccas))
        for f, cca in enumerate(self.ccas):
            [X_c, Y_c] = cca.fit_transform(X.T, self.templates[f].T)

            correlations[f] = np.ndarray(
                [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(self.cca_ncomponents)])
            
        # Get the predicted labels and probabilities
        predicted_labels = np.argmax(correlations, axis=0)
        probabilities = correlations / correlations.sum(axis=0, keepdims=True)
       
        return Prediction(predicted_labels, probabilities)