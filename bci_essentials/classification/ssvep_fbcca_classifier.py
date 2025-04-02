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

class SSVEPFBCCClassifier:
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