from mne_lsl.lsl import StreamInlet, StreamInfo, resolve_streams
from .sources import MarkerSource, EegSource
from ..utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)

__all__ = ["LslMarkerSource", "LslEegSource"]


class LslMarkerSource(MarkerSource):
    def __init__(self, stream: StreamInfo = None, timeout: float = 600):
        """Create a MarkerSource object that obtains markers from an LSL outlet

        Parameters
        ----------
        stream : StreamInfo, *optional*
            Provide stream to use for Markers, if not provided, stream will be discovered.
        timeout : float, *optional*
            How many seconds to wait for marker outlet stream to be discovered. 
            If no stream is discovered, an Exception is raised.
            By default init will wait 10 minutes.
        """
        try:
            if stream is None:
                stream = discover_first_stream("LSL_Marker_Strings", timeout=timeout)
            self.__inlet = StreamInlet(stream)
            self.__info = stream
        except Exception:
            raise Exception("LslMarkerSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name()

    def get_markers(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self.__inlet)

    def time_correction(self) -> float:
        return self.__inlet.time_correction()


class LslEegSource(EegSource):
    def __init__(self, stream: StreamInfo = None, timeout: float = 600):
        """Create a MarkerSource object that obtains EEG from an LSL outlet

        Parameters
        ----------
        stream : StreamInfo, *optional*
            Provide stream to use for EEG, if not provided, stream will be discovered.
        timeout : float, *optional*
            How many seconds to wait for marker outlet stream to be discovered. 
            If no stream is discovered, an Exception is raised.
            By default init will wait 10 minutes.
        """
        try:
            if stream is None:
                stream = discover_first_stream("EEG", timeout=timeout)
            self.__inlet = StreamInlet(stream)
            self.__info = stream
        except Exception:
            raise Exception("LslEegSource: could not create inlet")

    @property
    def name(self) -> str:
        return self.__info.name

    @property
    def fsample(self) -> float:
        return self.__info.sfreq

    @property
    def n_channels(self) -> int:
        return self.__info.n_channels

    @property
    def channel_types(self) -> list[str]:
        return [self.__info.stype] * self.n_channels

    @property
    def channel_units(self) -> list[str]:
        """Get channel units. Default to microvolts for EEG."""
        try:
            units = self.get_channel_properties('unit')
            # If no units found or empty strings, use default
            if not units or all(unit == "" for unit in units):
                logger.warning("No channel units found, defaulting to microvolts")
                units = ['µV'] * self.n_channels
            return units
        except Exception:
            logger.warning("Could not get channel units, defaulting to microvolts")
            return ['µV'] * self.n_channels

    @property
    def channel_labels(self) -> list[str]:
        """Get channel labels. 
        Uses ch_names if available, otherwise numbered channels.
        """
        if hasattr(self.__info, 'ch_names') and self.__info.ch_names:
            return list(self.__info.ch_names)
        return [f"Ch{i+1}" for i in range(self.n_channels)]

    def get_samples(self) -> tuple[list[list], list]:
        return pull_from_lsl_inlet(self.__inlet)

    def time_correction(self) -> float:
        return self.__inlet.time_correction()

    def get_channel_properties(self, property: str) -> list[str]:
        """Get channel properties from mne_lsl stream info.
        
        Parameters
        ----------
        property : str
            Property to get ('name', 'unit', 'type', etc)
        
        Returns
        -------
        list[str]
            List of property values for each channel
        """
        if property == 'name':
            return self.name
        elif property == "unit":
            return [''] * self.n_channels 
        elif property == "type":
            return self.channel_types
        elif property == "label":
            return self.channel_labels
        else:
            logger.warning(f"Property '{property}' not supported in mne_lsl")
            return [''] * self.n_channels


def discover_first_stream(type: str, timeout: float = 600) -> StreamInfo:
    """This helper returns the first stream of the specified type.

    If no stream is found, an exception is raised."""
    streams = resolve_streams(stype=type, timeout=timeout)
    return streams[0]


def pull_from_lsl_inlet(inlet: StreamInlet) -> tuple[list[list], list]:
    """StreamInlet.pull_chunk() may return None for samples.

    This helper prevents `None` from propagating by converting it into [[]].

    If None is detected, the timestamps list is also forced to [].
    """

    # read from inlet
    samples, timestamps = inlet.pull_chunk(timeout=0.1)

    # convert None or empty samples into empty lists
    if samples is None or len(samples) == 0:
        samples = [[]]
        timestamps = []

    # return tuple[list[list], list]
    return [samples, timestamps]
