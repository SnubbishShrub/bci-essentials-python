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
from imblearn.over_sampling import SMOTE

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
    # Get only positive class examples
    true_X = X[y == 1]
    
    n_trials, n_channels, n_samples = true_X.shape
    logger.info("Shape of ERPs only: %s", true_X.shape)
    
    # Calculate number of new trials needed
    n_new_trials = int(n_trials * (expansion_factor - 1))
    new_X = np.zeros([n_new_trials, n_channels, n_samples])
    
    # Generate synthetic trials
    for trial_idx in range(n_new_trials):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(sum_num), size=1)[0]
        # For each new trial, create a random combination of existing trials
        for j in range(sum_num):
            # Select a random positive example
            random_trial_idx = random.randint(0, n_trials - 1)
            random_epoch = true_X[random_trial_idx, :, :]
            # Add it to the new trial (scaled by 1/sum_num)
            new_X[trial_idx, :, :] += weights[j] * random_epoch

        # Add small noise for variability
        noise = np.random.normal(0, 0.01, size=new_X[trial_idx, :, :].shape)
        new_X[trial_idx, :, :] += noise

        # Normalize the new sample
        new_X[trial_idx, :, :] /= np.linalg.norm(new_X[trial_idx, :, :])
    
    # Combine original data with synthetic data
    over_X = np.append(X, new_X, axis=0)
    over_y = np.append(y, np.ones([n_new_trials]))
    
    # Shuffle the data if requested
    if shuffle:
        # Get indices and shuffle them
        indices = np.arange(len(over_y))
        np.random.shuffle(indices)
        # Reorder according to shuffled indices
        over_X = over_X[indices]
        over_y = over_y[indices]
    
    return over_X, over_y

def eeg_smote(X, y, expansion_factor=3, k_neighbors=5, shuffle=False):
    """Oversampling using SMOTE (Synthetic Minority Over-sampling Technique)
    
    Generates synthetic EEG trials for the minority class (typically target/P300 responses).
    
    Parameters
    ----------
    X : numpy.ndarray
        Trials of EEG data.
        3D array containing data with `float` type.
        shape = (n_trials, n_channels, n_samples)
    y : numpy.ndarray
        Labels corresponding to X.
    expansion_factor : float, *optional*
        Controls the amount of oversampling for the minority class.
        - Default is `3`.
    k_neighbors : int, *optional*
        Number of nearest neighbors to use for synthetic sample generation.
        - Default is `5`.
    shuffle : bool, *optional*
        Whether to shuffle the final combined dataset.
        - Default is `False`.
        
    Returns
    -------
    over_X : numpy.ndarray
        Oversampled X.
    over_y : numpy.ndarray
        Oversampled y.
    """

    # Get dimensions
    n_trials, n_channels, n_samples = X.shape
    
    # Calculate target number of minority class samples
    n_minority = sum(y == 1)
    sampling_strategy = min(expansion_factor, (len(y) - n_minority) / n_minority)
    
    # Reshape X to 2D for SMOTE (combine channels and samples)
    X_reshaped = X.reshape(n_trials, n_channels * n_samples)
    
    # Apply SMOTE
    try:
        # If not enough minority samples for k_neighbors, reduce k
        if n_minority <= k_neighbors:
            k_neighbors = max(1, n_minority - 1)
            logger.warning(f"Reduced k_neighbors to {k_neighbors} due to small minority class")
            
        # Configure and apply SMOTE
        smote = SMOTE(
            sampling_strategy="auto",
            k_neighbors=k_neighbors,
            random_state=42
        )
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape back to 3D
        X_resampled = X_resampled.reshape(-1, n_channels, n_samples)
        
        # Shuffle if requested
        if shuffle:
            indices = np.arange(len(y_resampled))
            np.random.shuffle(indices)
            X_resampled = X_resampled[indices]
            y_resampled = y_resampled[indices]
        
        logger.info(f"SMOTE expanded data from {len(y)} to {len(y_resampled)} samples")
        logger.info(f"New class balance: {sum(y_resampled == 1)}/{len(y_resampled)}")
        
        return X_resampled, y_resampled
        
    except ValueError as e:
        logger.error(f"SMOTE failed: {e}. Returning original data.")
        return X, y
    
def random_undersample(X, y, undersample_ratio=0.5, shuffle=True, random_seed=42):
    """Random undersampling for imbalanced datasets.
    
    Randomly removes samples from the majority class to achieve desired class balance.
    
    Parameters
    ----------
    X : numpy.ndarray
        Trials of EEG data.
        3D array containing data with `float` type.
        shape = (n_trials, n_channels, n_samples)
    y : numpy.ndarray
        Labels corresponding to X.
    undersample_ratio : float, *optional*
        Target ratio of minority:majority samples (0.5 means 1:2 ratio).
        Higher values mean less undersampling.
        - Default is `0.5`.
    shuffle : bool, *optional*
        Whether to shuffle the final dataset.
        - Default is `True`.
        
    Returns
    -------
    under_X : numpy.ndarray
        Undersampled X.
    under_y : numpy.ndarray
        Undersampled y.
    """
    # Get current class counts
    n_minority = sum(y == 1)
    n_majority = sum(y == 0)
    
    if n_minority == 0:
        logger.warning("No minority class samples. Returning original data.")
        return X, y
    
    # Calculate how many majority samples to keep
    # undersample_ratio = minority/majority, so majority = minority/undersample_ratio
    n_majority_keep = min(n_majority, int(n_minority / undersample_ratio))
    n_majority_remove = n_majority - n_majority_keep
    
    if n_majority_remove <= 0:
        logger.info("No undersampling needed. Current ratio already meets target.")
        return X, y
    
    # Get indices of majority class
    majority_indices = np.where(y == 0)[0]
    
    # Randomly select indices to remove
    np.random.seed(random_seed)  # For reproducibility
    remove_indices = np.random.choice(majority_indices, n_majority_remove, replace=False)
    
    # Get indices to keep
    keep_indices = np.array([i for i in range(len(y)) if i not in remove_indices])
    
    # Create undersampled dataset
    under_X = X[keep_indices]
    under_y = y[keep_indices]
    
    # Shuffle if requested
    if shuffle:
        indices = np.arange(len(under_y))
        np.random.shuffle(indices)
        under_X = under_X[indices]
        under_y = under_y[indices]
    
    logger.info(f"Undersampling removed {n_majority_remove} majority samples")
    logger.info(f"New class balance: {sum(under_y == 1)}/{len(under_y)} " 
                f"(ratio: {sum(under_y == 1)/(len(under_y)-sum(under_y == 1)):.3f})")
    
    return under_X, under_y
