import numpy as np
from ..utils.logger import Logger  # Logger wrapper
import time

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


# Will eventually move somewhere else
class DataTank:
    """
    DataTank class is for storing raw EEG, markers, epochs, and resting state data

    TODO: Add your desired flavour of save output from here.
    To be added:
    - MNE Raw
    - MNE Epochs
    - BIDS
    - XDF
    """

    def __init__(self, max_samples: int = 100000):
        """
        Initialize the DataTank.

        Parameters
        ----------
        max_samples : int
            The maximum number of samples to store in the DataTank. 
            Default is 100,000 (i.e., ~100 sec at 100 Hz).
        """

        # Buffer configuration
        self.max_samples = max_samples
               
        # Buffer state tracking
        self._eeg_write_index = 0
        self._eeg_samples_written = 0
        self._marker_write_index = 0
        self._markers_written = 0
        self._buffer_initialized = False
        
        # Metadata (will be set in set_source_data)
        self.headset_string = None
        self.fsample = None
        self.n_channels = None
        self.ch_types = None
        self.ch_units = None
        self.channel_labels = None

        # Keep track of the latest timestamp
        self.latest_eeg_timestamp = 0

        # Keep track of how many epochs have been sent
        self.epochs_sent = 0
        self.epochs = np.zeros((0, 0, 0))
        self.labels = np.zeros((0))
        
        self.__resting_state_data = None

        # Buffers will be initialized in set_source_data()
        self.__raw_eeg_buffer = None
        self.__raw_eeg_timestamps_buffer = None
        self.__raw_marker_buffer = None
        self.__raw_marker_timestamps_buffer = None

    def _add_to_circular_buffer(self, buffer, data, write_index, max_size):
        """
        Helper function to add data to any circular buffer with wraparound handling.
        
        Parameters
        ----------
        buffer : np.array
            The circular buffer to write to
        data : np.array
            The data to write (1D or 2D)
        write_index : int
            Current write position
        max_size : int
            Maximum buffer size
        
        Returns
        -------
        int
            Number of items written
        """
        if data.size == 0:
            return 0
        
        # Handle both 1D (timestamps, markers) and 2D (EEG) data
        n_items = data.shape[0]
        start_write_idx = write_index % max_size
        
        if start_write_idx + n_items <= max_size:
            # No wraparound - single assignment
            if data.ndim == 1:
                buffer[start_write_idx:start_write_idx + n_items] = data
            else:
                buffer[start_write_idx:start_write_idx + n_items, :] = data
        else:
            # Wraparound - split into two assignments
            first_chunk_size = max_size - start_write_idx
            second_chunk_size = n_items - first_chunk_size
            
            if data.ndim == 1:
                buffer[start_write_idx:] = data[:first_chunk_size]
                buffer[:second_chunk_size] = data[first_chunk_size:]
            else:
                buffer[start_write_idx:, :] = data[:first_chunk_size, :]
                buffer[:second_chunk_size, :] = data[first_chunk_size:, :]
        
        return n_items

    def _get_from_circular_buffer(self, buffer, timestamps_buffer, write_index, items_written, max_size):
        """
        Helper function to retrieve data from circular buffer in chronological order.
        
        Parameters
        ----------
        buffer : np.array
            The circular buffer to read from
        timestamps_buffer : np.array or None
            Associated timestamps buffer (can be None)
        write_index : int
            Current write position
        items_written : int
            Total items written to buffer
        max_size : int
            Maximum buffer size
        
        Returns
        -------
        tuple
            (data, timestamps) in chronological order, or just data if no timestamps
        """
        if items_written == 0:
            if buffer.ndim == 1:
                empty_data = np.array([])
            else:
                empty_data = np.zeros((0, buffer.shape[1]))
            
            if timestamps_buffer is not None:
                return empty_data, np.array([])
            else:
                return empty_data
        
        if items_written < max_size:
            # Buffer not full yet
            data = buffer[:items_written]
            timestamps = timestamps_buffer[:items_written] if timestamps_buffer is not None else None
        else:
            # Buffer full, reconstruct chronological order
            start_idx = write_index % max_size
            
            if buffer.ndim == 1:
                data = np.concatenate([buffer[start_idx:], buffer[:start_idx]])
            else:
                data = np.vstack([buffer[start_idx:, :], buffer[:start_idx, :]])
            
            if timestamps_buffer is not None:
                timestamps = np.concatenate([
                    timestamps_buffer[start_idx:], 
                    timestamps_buffer[:start_idx]
                ])
            else:
                timestamps = None
        
        if timestamps_buffer is not None:
            return data, timestamps
        else:
            return data
        
    def set_source_data(
        self, 
        headset_string: str,
        fsample: float,
        n_channels: int,
        ch_types: list[str],
        ch_units: list[str],
        channel_labels: list[str]

    ):
        """
        Set the source data for the DataTank so that this metadata can be saved with the data.

        Parameters
        ----------
        headset_string : str
            The name of the headset used to collect the data.
        fsample : float
            The sampling frequency of the data.
        n_channels : int
            The number of channels in the data.
        ch_types : list of str
            The type of each channel.
        ch_units : list of str
            The units of each channel.
        channel_labels : list of str
            The labels of each channel.

        Returns
        -------
        `None`
        """
        self.headset_string = headset_string
        self.fsample = fsample
        self.n_channels = n_channels
        self.ch_types = ch_types
        self.ch_units = ch_units
        self.channel_labels = channel_labels

        # Initialize pre-allocated circular buffers
        logger.info(f"Initializing EEG buffer: {self.max_samples} samples x {n_channels} channels")
        
        self.__raw_eeg_buffer = np.zeros((self.max_samples, n_channels), dtype=np.float32)
        self.__raw_eeg_timestamps_buffer = np.zeros(self.max_samples, dtype=np.float64)
        
        # Smaller buffer for markers
        max_markers = min(self.max_samples // 10, 10000)
        self.__raw_marker_buffer = np.zeros(max_markers, dtype='U100')
        self.__raw_marker_timestamps_buffer = np.zeros(max_markers, dtype=np.float64)
        
        # Reset indices
        self._eeg_write_index = 0
        self._eeg_samples_written = 0
        self._marker_write_index = 0
        self._markers_written = 0
        
        self._buffer_initialized = True
        logger.info("Pre-allocated buffers initialized successfully")

    def add_raw_eeg(self, new_raw_eeg, new_eeg_timestamps):
        """
        Add raw EEG data to the data tank.

        Parameters
        ----------
        new_raw_eeg : np.array
            The new raw EEG data to add.

        new_eeg_timestamps : np.array
            The timestamps of the new raw EEG data.

        Returns
        -------
        `None`
        """

        if new_raw_eeg.size == 0:
            return
        
        if not self._buffer_initialized:
            raise RuntimeError("DataTank buffers not initialized. Call set_source_data() first.")
        
        # Transpose if needed (input is n_channels x n_samples, buffer is n_samples x n_channels)
        if new_raw_eeg.shape[0] == self.n_channels:
            new_raw_eeg = new_raw_eeg.T
        
        # Use helper function for both EEG data and timestamps
        n_written = self._add_to_circular_buffer(
            self.__raw_eeg_buffer, new_raw_eeg, self._eeg_write_index, self.max_samples
        )
        self._add_to_circular_buffer(
            self.__raw_eeg_timestamps_buffer, new_eeg_timestamps, self._eeg_write_index, self.max_samples
        )
        
        # Update indices
        self._eeg_write_index += n_written
        self._eeg_samples_written += n_written
        self.latest_eeg_timestamp = new_eeg_timestamps[-1]

    def add_raw_markers(self, new_marker_strings, new_marker_timestamps):
        """
        Add raw markers to the data tank.

        Parameters
        ----------
        new_marker_strings : np.array
            The new marker strings to add.
        new_marker_timestamps : np.array
            The timestamps of the new marker strings.

        Returns
        -------
        `None`
        """

        if len(new_marker_strings) == 0:
            return
            
        if not self._buffer_initialized:
            raise RuntimeError("DataTank buffers not initialized. Call set_source_data() first.")
        
        max_markers = len(self.__raw_marker_buffer)
        
        # Use helper function for both markers and timestamps
        n_written = self._add_to_circular_buffer(
            self.__raw_marker_buffer, new_marker_strings, self._marker_write_index, max_markers
        )
        self._add_to_circular_buffer(
            self.__raw_marker_timestamps_buffer, new_marker_timestamps, self._marker_write_index, max_markers
        )
        
        # Update indices
        self._marker_write_index += n_written
        self._markers_written += n_written

    def get_raw_eeg(self):
        """
        Get the raw EEG data from the DataTank.

        Returns
        -------
        np.array
            The raw EEG data.
        np.array
            The timestamps of the raw EEG data.
        """
        if not self._buffer_initialized:
            raise RuntimeError("DataTank buffers not initialized. Call set_source_data() first.")

        eeg_data, timestamps = self._get_from_circular_buffer(
            self.__raw_eeg_buffer, self.__raw_eeg_timestamps_buffer,
            self._eeg_write_index, self._eeg_samples_written, self.max_samples
        )
        
        # Transpose back to (n_channels, n_samples) and handle empty case
        if eeg_data.size > 0:
            eeg_data = eeg_data.T
        else:
            eeg_data = np.zeros((self.n_channels, 0))
            timestamps = np.zeros((0))
        
        return eeg_data, timestamps

    def get_raw_markers(self):
        """
        Get the raw markers from the DataTank.

        Returns
        -------
        np.array
            The raw marker strings.
        np.array
            The timestamps of the raw marker strings.
        """
        if not self._buffer_initialized:
            raise RuntimeError("DataTank buffers not initialized. Call set_source_data() first.")
        
        return self._get_from_circular_buffer(
            self.__raw_marker_buffer, self.__raw_marker_timestamps_buffer,
            self._marker_write_index, self._markers_written, len(self.__raw_marker_buffer)
        )

    def add_epochs(self, X, y):
        """
        Add epochs to the data tank.

        Parameters
        ----------
        X : np.array
            The new epochs to add. Shape is (n_epochs, n_channels, n_samples).
        y : np.array
            The labels of the epochs. Shape is (n_epochs).

        Returns
        -------
        `None`

        """
        if len(X) == 0:
            return

        X = np.array(X)
        y = np.array(y)

        # Convert to lists for efficient appending, then back to numpy when needed
        if not hasattr(self, '_epochs_list'):
            # Initialize lists from existing arrays
            self._epochs_list = self.epochs.tolist() if self.epochs.size > 0 else []
            self._labels_list = self.labels.tolist() if self.labels.size > 0 else []

        # Simple shape check without creating temporary arrays
        if (len(self._epochs_list) > 0 and 
            len(X.shape) == 3 and 
            len(self._epochs_list[0]) != X.shape[1]):  # Check n_channels
            logger.warning("Epochs have different number of channels, skipping this data.")
            return

        # Append to lists (much faster than np.concatenate)
        for epoch, label in zip(X, y):
            self._epochs_list.append(epoch.tolist())
            self._labels_list.append(label)

        # Update numpy arrays periodically (every 10 epochs) or when accessed
        if len(self._epochs_list) % 10 == 0:
            self._sync_epochs()

    def _sync_epochs(self):
        """Convert epoch lists back to numpy arrays."""
        if hasattr(self, '_epochs_list') and len(self._epochs_list) > 0:
            self.epochs = np.array(self._epochs_list)
            self.labels = np.array(self._labels_list)

    def get_epochs(self, latest=False):
        """
        Get the epochs from the data tank.

        Parameters
        ----------
        latest : bool
            If `True`, only return the new data since the last call to this function.

        Returns
        -------
        np.array
            The epochs. Shape is (n_epochs, n_channels, n_samples).
        np.array
            The labels of the epochs. Shape is (n_epochs).

        """
        # Sync any pending epochs
        if hasattr(self, '_epochs_list'):
            self._sync_epochs()

        if self.epochs.size == 0:
            logger.warning("Data tank contains no epochs, returning empty arrays.")
            return np.array([]), np.array([])

        if latest:
            # Return only new epochs
            if self.epochs_sent >= len(self.epochs):
                return np.array([]), np.array([])
            
            new_epochs = self.epochs[self.epochs_sent:]
            new_labels = self.labels[self.epochs_sent:]
            self.epochs_sent = len(self.epochs)
            return new_epochs, new_labels
        else:
            # Return all epochs
            return self.epochs, self.labels

    def add_resting_state_data(self, resting_state_data):
        """
        Add resting state data to the data tank.

        Parameters
        ----------
        resting_state_data : dict
            Dictionary containing resting state data.

        Returns
        -------
        `None`

        """
        # Get the resting state data
        self.__resting_state_data = resting_state_data

    def get_resting_state_data(self):
        """
        Get the resting state data.

        Returns
        -------
        dict
            Dictionary containing resting state data.

        """
        return self.__resting_state_data

    def save_epochs_as_npz(self, file_name: str):
        """
        Saves EEG trials and labels as a numpy file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the EEG trials and labels to.

        Returns
        -------
        `None`

        """
        # Check if file ends with .npz, if not add it
        if file_name[-4:] != ".npz":
            file_name += ".npz"

        # Get the raw EEG trials and labels
        X = self.epochs
        y = self.labels

        # Save the raw EEG trials and labels as a numpy file
        np.savez(file_name, X=X, y=y)
