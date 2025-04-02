"""
Signal processing tools for processing trials of EEG data.

The EEG data inputs can be 2D or 3D arrays.
- For single trials, inputs are of the shape `n_channels x n_samples`, where:
    - n_channels = number of channels
    - n_samples = number of samples
- For multiple trials, inputs are of the shape `n_trials x n_channels x n_samples`, where:
    - n_trials = number of trials
    - n_channels = number of channels
    - n_samples = number of samples

- Outputs are the same dimensions as input (trials, channels, samples)

"""

import numpy as np
from scipy import signal
import random
from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def bandpass(data, f_low, f_high, order, fsample):
    """Bandpass Filter.

    Filters out frequencies outside of the range f_low to f_high with a
    Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_low : float
        Lower corner frequency.
    f_high : float
        Upper corner frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    """
    Wn = [f_low / (fsample / 2), f_high / (fsample / 2)]
    b, a = signal.butter(order, Wn, btype="bandpass")

    try:
        n_trials, n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_trials, n_channels, n_samples), dtype=float)
        for trial in range(0, n_trials):
            current_trial = data[trial, :, :]
            new_data[trial, :, :] = signal.filtfilt(b, a, current_trial, padlen=0)

        return new_data

    except ValueError:
        n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_channels, n_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def lowpass(data, f_critical, order, fsample):
    """Lowpass Filter.

    Filters out frequencies above f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    """
    Wn = f_critical / (fsample / 2)
    b, a = signal.butter(order, Wn, btype="lowpass")

    try:
        n_trials, n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_trials, n_channels, n_samples), dtype=float)
        for trial in range(0, n_trials):
            for channel in range(0, n_channels):
                current_trial = data[trial, channel, :]
                new_data[trial, channel, :] = signal.filtfilt(
                    b, a, current_trial, padlen=0
                )

        return new_data

    except ValueError:
        n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_channels, n_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def highpass(data, f_critical, order, fsample):
    """Highpass Filter.

    Filters out frequencies below f_critical with a Butterworth filter of specific order.

    Wraps the scipy.signal.butter and scipy.signal.filtfilt methods.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_critical : float
        Critical (cutoff) frequency.
    order : int
        Order of the filter.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    """
    Wn = f_critical / (fsample / 2)
    b, a = signal.butter(order, Wn, btype="highpass")

    try:
        n_trials, n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_trials, n_channels, n_samples), dtype=float)
        for trial in range(0, n_trials):
            current_trial = data[trial, :, :]
            new_data[trial, :, :] = signal.filtfilt(b, a, current_trial, padlen=0)

        return new_data

    except ValueError:
        n_channels, n_samples = np.shape(data)

        new_data = np.ndarray(shape=(n_channels, n_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)

        return new_data


def notch(data, f_notch, Q, fsample):
    """Notch Filter.

    Notch filter for removing specific frequency components.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    f_notch : float
        Frequency of notch.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth bw relative to its
        center frequency, Q = w0/bw.
    fsample : float
        Sampling rate of signal.

    Returns
    -------
    new_data : numpy.ndarray
        Trials of filtered EEG data.
        3D array containing data with `float` type.

        shape = (`n_trials`,`n_channels`,`n_samples`)

    """

    b, a = signal.iirnotch(f_notch, Q, fsample)

    try:
        n_trials, n_channels, n_samples = np.shape(data)
        new_data = np.ndarray(shape=(n_trials, n_channels, n_samples), dtype=float)
        for trial in range(0, n_trials):
            current_trial = data[trial, :, :]
            new_data[trial, :, :] = signal.filtfilt(b, a, current_trial, padlen=0)
        return new_data

    except Exception:
        n_channels, n_samples = np.shape(data)
        new_data = np.ndarray(shape=(n_channels, n_samples), dtype=float)
        new_data = signal.filtfilt(b, a, data, padlen=0)
        return new_data

def construct_filter_bank(cutoff_freqs, fsample, filter_order):
    """Construct a filter bank for SSVEP detection.

    Parameters
    ----------
    target_freqs : array-like
        Target frequencies for SSVEP detection.
        Should be [n_filters, 2]. Where the first column is the low-cutoff
        frequency and the second column is the high-cutoff frequency.
    fsample : float
        Sampling rate of signal [Hz].
    filter_order : int
        Order of the filter.

    Returns
    -------
    filter_bank : list of `ndarray`
        List of filter coefficients for each target frequency.

    """
    # Pre-allocate filter bank array
    n_filters = len(cutoff_freqs)
    filter_bank = np.zeros((n_filters, 6, filter_order + 1))  # SOS format is 6 x (order + 1)
    
    # Normalize frequencies
    nyq = fsample / 2
    norm_freqs = cutoff_freqs / nyq
    
    # Create filter bank for all frequencies at once
    for f, (f_low, f_high) in enumerate(norm_freqs):
        filter_bank[f] = signal.butter(
            filter_order,
            [f_low, f_high],
            btype='band',
            output='sos'
            )
    
    return filter_bank

def implement_filter_bank(data, f_low, f_high, order, fsample):
    """ Filter Bank.
    """
    pass

def lico(X, y, expansion_factor=3, sum_num=2, shuffle=False):
    """Oversampling (linear combination oversampling (LiCO))

    Samples random linear combinations of existing epochs of X.

    This is broken, but I am also unsure if it deserves to be fixed. At the very least it probably belongs in a different file. -Brian

    Parameters
    ----------
    X : numpy.ndarray
        Trials of EEG data.
        3D array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples)
    y : numpy.ndarray
        Labels corresponding to X.
    expansion_factor : int, *optional*
        Number of times larger to make the output set over_X
        - Default is `3`.
    sum_num : int, *optional*
        Number of signals to be summed together
        - Default is `2`.

    Returns
    -------
    over_X : numpy.ndarray
        Oversampled X.
    over_y : numpy.ndarray
        Oversampled y.

    """
    true_X = X[y == 1]

    n_trials, n_channels, n_samples = true_X.shape
    logger.info("Shape of ERPs only: %s", true_X.shape)
    new_trial = n_trials * np.round(expansion_factor - 1)
    new_X = np.zeros([new_trial, n_channels, n_samples])
    for trial in range(n_trials):
        for j in range(sum_num):
            random_epoch = true_X[random.choice(range(n_trials)), :, :]
            new_X[trial, :, :] += random_epoch / sum_num

    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([new_trial]))

    return over_X, over_y


def ssvep_templates(
    target_freqs,
    srate=256.0,
    n_samples=256,
    n_harmonics=3,
):
    """Generate SSVEP templates with sine-cosine pairs for each target frequency.

    Parameters
    ----------
    target_freqs : array-like
        Target frequencies [Hz].
    srate : float
        Sampling rate of signal [Hz].
    n_samples : int
        Number of samples in each trial.
    n_harmonics : int, *optional*
        Number of harmonics.
        - Default is `3`.
    
    Returns
    -------
    templates : numpy.ndarray
        SSVEP templates with shape (n_targets, 2*n_harmonics, n_samples)
        where each target frequency has n_harmonics sine-cosine pairs.
        The harmonics are arranged as [sin(f), cos(f), sin(2f), cos(2f), ...].
    """

        # Create time vector
    t = np.arange(0, n_samples + 1, dtype=np.float32) / srate

    # Create frequency-harmonic combinations
    freqs = target_freqs[:, np.newaxis]  
    harmonics = np.arange(1, n_harmonics +1) 
    freq_harm = freqs * harmonics 

    # Compute phase terms using broadcasting
    phase = 2 * np.pi * t[np.newaxis, :, np.newaxis] * freq_harm[:, np.newaxis, :]
        
    # Interleave sin and cos templates
    n_targets = len(target_freqs)
    template_signal = np.empty((n_targets, n_samples, 2 * (n_harmonics + 1)), dtype=np.float32)
    template_signal[:, :, 0::2] = np.sin(phase)
    template_signal[:, :, 1::2] = np.cos(phase)