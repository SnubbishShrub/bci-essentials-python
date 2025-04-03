"""
**SSVEP FB CCA Classifier**

Classifies SSVEPs using the FBCCA method.

"""

# Stock libraries
import numpy as np
from sklearn.cross_decomposition import CCA


# Import bci_essentials modules and methods
from ..signal_processing import (
    ssvep_templates,
    concatenate_trials,
    construct_filter_bank,
    implement_filter_bank,
)
from .generic_classifier import GenericClassifier, Prediction
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

class SsvepFbCcaClassifier(GenericClassifier):
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

    def set_ssvep_settings(self, fsample, target_freqs, n_samples=None, n_harmonics=0, filter_bank=None, filter_order=4, concatenate_trials=False):
        """Initialize the SSVEP FBCCA Classifier.

        Parameters
        ----------
        fsample : float
            Sampling frequency of the EEG data [Hz].
        target_freqs : list of `float`
            List of the target frequencies for SSVEP detection [Hz].
        n_samples : int, optional
            Number of samples in each trial, used to compute signal templates (default is None).
        n_harmonics : int, optional
            Number of harmonics to use in the filter bank (default is 0).
        filter_bank :array-like **optional**
            Cutoff frequencies for the filter bank [Hz].
            Should be [n_filters, 2]. Where the first column is the low-cutoff
            frequency and the second column is the high-cutoff frequency.
        filter_order : int, optional
            Order of the filter (default is 4).
        concatenate_trials : bool, optional
            Concatenates trials using a Hanning window (default is False).

        Returns
        -------
        `None`
            
        """
        self.fsample = fsample
        self.filter_bank = filter_bank
        self.target_freqs = target_freqs
        self.n_samples = n_samples
        self.n_harmonics = n_harmonics
        self.filter_bank = filter_bank
        self.filter_order = filter_order
        self.templates = None   # SSVEP signal templates
        self.cca_ncomponents = 1  # Number of components for CCA
        self.concatenate_trials = concatenate_trials        

        # Create filter bank
        if filter_bank:
            self.fb_coefficients = construct_filter_bank(
                self.filter_bank,
                self.fsample,
                self.filter_order
            )

        # Create templates for each target frequency
        self.templates = ssvep_templates(
            target_freqs=self.target_freqs,
            fsample=self.fsample,
            n_samples=self.n_samples,
            n_harmonics=self.n_harmonics
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
            Can be 2D [channels, samples] or 3D [trials, channels, samples] array.

        Returns
        -------
        prediction : Prediction
            Results of predict call containing the predicted class labels.  Probabilities
            are not available (empty list).

        """
        # Concatenate signal if necessary
        if self.concatenate_trials:
            X = concatenate_trials(X)

        # Create signal templates for each target frequency
        # This is only done once to speed up the process, or if the epoch size changes
        if ((self.templates is None) or (X.shape[-1] != self.templates.shape[-1])):
            self.templates = ssvep_templates(
                target_freqs=self.target_freqs,
                fsample=self.fsample,
                n_samples=X.shape[-1],
                n_harmonics=self.n_harmonics
            )
       
        # Filter bank the signal if necessary
        if self.filter_bank:
            X = implement_filter_bank(X, self.fb_coefficients)

        # Compute CCA correlations 
        correlations = np.zeros(len(self.ccas))
        for f, cca in enumerate(self.ccas):
            [X_c, Y_c] = cca.fit_transform(X.T, self.templates[f].T)

            component_correlations = [
                np.abs(np.corrcoef(X_c[:, c], Y_c[:, c])[0, 1])
                for c in range(self.cca_ncomponents)
            ]

            # Keep max correlation across components
            correlations[f] = np.max(component_correlations)
            
        # Get the predicted labels and probabilities
        predicted_labels = np.argmax(correlations)
        probabilities = correlations / correlations.sum()
       
        return Prediction(predicted_labels, probabilities)