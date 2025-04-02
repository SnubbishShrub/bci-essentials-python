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

    def sef_ssvep_clf_settings(self, sampling_freq, target_freqs, n_harmonics=1, n_filters=4):
        """Initialize the SSVEP FBCCA Classifier.

        Parameters
        ----------
        sampling_freq : int
            Sampling frequency of the EEG data.
        target_freqs : list of `int`
            List of the target frequencies for SSVEP detection.
        n_harmonics : int, optional
            Number of harmonics to use in the filter bank (default is 1).
        n_filters : int, optional
            Number of filters in the filter bank (default is 4).

        """
        self.sampling_freq = sampling_freq
        self.target_freqs = target_freqs
        self.n_harmonics = n_harmonics
        self.n_filters = n_filters

        # Create filter bank
        self.filter_bank = filter_bank.create_filter_bank(
            target_freqs, sampling_freq, n_harmonics, n_filters
        )

        # Initialize CCA object
        self.cca = CCA(n_components=1)

        # Create templates for each target frequency
        self.templates = ssvep_templates.create_templates(
            target_freqs, sampling_freq, n_harmonics
        )

        # Construct filter bank
        self.filter_bank = filter_bank.create_filter_bank(
            target_freqs, sampling_freq, n_harmonics, n_filters
        )

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
        # Get the shape of the input data
        n_trials, n_channels, n_samples = X.shape

        # Initialize the predictions array
        probabilities = np.zeros((n_trials, len(self.target_freqs)))

        # Apply filter bank to each trial
        for i in range(n_trials):
            filtered_data = implement_filter_bank(X[i], self.filter_bank)

            # Perform CCA for each target frequency
            for j in range(len(self.target_freqs)):
                self.cca.fit(filtered_data, self.templates[j])
                probabilities[i, j] = self.cca.score(filtered_data, self.templates[j])

        # Get the predicted class labels
        predicted_labels = np.argmax(probabilities, axis=1)

        # Create and return a Prediction object
        return Prediction(predicted_labels, probabilities)