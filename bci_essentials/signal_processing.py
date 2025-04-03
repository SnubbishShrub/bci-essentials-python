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
    cutoff_freqs : array-like
        Cutoff frequencies for the filter bank [Hz].
        Should be [n_filters, 2]. Where the first column is the low-cutoff
        frequency and the second column is the high-cutoff frequency.
    fsample : float
        Sampling rate of signal [Hz].
    filter_order : int
        Order of the filter.

    Returns
    -------
    fb_coefficients : np.ndarray
        Array of  filter coefficients for each target frequency.

    """
    # Pre-allocate filter bank array
    n_filters = len(cutoff_freqs)
    fb_coefficients = np.zeros((n_filters, filter_order, 6))
    
    # Normalize frequencies
    nyq = fsample / 2
    norm_freqs = cutoff_freqs / nyq
    
    # Create filter bank for all frequencies at once
    for f, (f_low, f_high) in enumerate(norm_freqs):
        fb_coefficients[f, :, :] = signal.butter(
            filter_order,
            [f_low, f_high],
            btype='band',
            output='sos'
            )
    
    return fb_coefficients

def implement_filter_bank(data, fb_coefficients):
    """ Filter Bank.

    Parameters
    ----------
    data : numpy.ndarray
        Trials of EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    fsample : float
        Sampling rate of signal [Hz].
    filter_bank : list of `ndarray`
        List of filter coefficients for each target frequency.

    Returns
    -------
    filtered_data : numpy.ndarray
        Trials of filtered EEG data.
        3D (or 2D) array containing data with `float` type.

        shape = (n_trials, len(filter_bank), n_channels, n_samples)
        or ((len_filter_bank) n_channels, n_samples)

    """
    # Handle both 2D and 3D inputs
    is_3d = (data.ndim == 3)
    if not is_3d:
        data = data[np.newaxis, ...]
    
    [n_trials, n_channels, n_samples] = data.shape
    n_filters = fb_coefficients.shape[0]
    
    # Pre-allocate output array
    filtered_data = np.empty((n_trials, n_filters, n_channels, n_samples), dtype=np.float32)
    
    # Apply all filters to each trial
    for trial in range(n_trials):
        for f in range(n_filters):
            filtered_data[trial, f] = signal.sosfiltfilt(fb_coefficients[f,:,:], data[trial])

    # Return same dimensions as input + filter bank
    return filtered_data if is_3d else filtered_data[0]


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
    fsample=256.0,
    n_samples=None,
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
        Number of samples in each trial (default is `None`).
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

    if n_samples is None:
        logger.warning("SSVEP templates not computed because n_samples is None.")
        return None
    
    # Create time vector
    t = np.arange(0, n_samples, dtype=np.float32) / fsample

    # Create frequency-harmonic combinations
    freqs = target_freqs[:, np.newaxis, np.newaxis]  
    harmonics = np.arange(1, n_harmonics + 2)[np.newaxis, :, np.newaxis]

    # Compute phase terms using broadcasting
    phase = 2 * np.pi * freqs * harmonics * t[np.newaxis, np.newaxis, :]
        
    # Interleave sin and cos templates
    n_targets = len(target_freqs)
    template_signal = np.empty((n_targets, 2 * (n_harmonics + 1), n_samples), dtype=np.float32)
    template_signal[:, 0::2, :] = np.sin(phase)
    template_signal[:, 1::2, :] = np.cos(phase)

    return template_signal

def concatenate_trials(X):
    """Concatenate trials along the time axis using a Hanning window.

    Parameters
    ----------
    X : numpy.ndarray
        Trials of EEG data.
        3D array containing data with `float` type.

        shape = (n_trials, n_channels, n_samples)

    Returns
    -------
    concatenated_X : numpy.ndarray
        Concatenated trials of EEG data.
        2D array containing data with `float` type.

        shape = (n_channels, n_trials * n_samples)

    """
    
    # Get dimensions
    [_, n_channels, n_samples] = X.shape
    
    # Create and apply Hanning window to all trials at once
    window = np.hanning(n_samples)
    windowed_X = X * window[np.newaxis, np.newaxis, :]
    
    # Reshape to (n_channels, n_trials * n_samples)
    # transpose first to get trials sequential for each channel
    concatenated_X = windowed_X.transpose(1, 0, 2).reshape(n_channels, -1)

    return concatenated_X