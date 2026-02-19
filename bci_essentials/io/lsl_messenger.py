from mne_lsl.lsl import StreamInfo, StreamOutlet
from .messenger import Messenger
from ..classification.generic_classifier import Prediction
import numpy as np

__all__ = ["LslMessenger"]


class LslMessenger(Messenger):
    """A Messenger object for sending event messages to an LSL outlet."""

    def __init__(self):
        """Create an LslMessenger object.

        If the LSL outlet cannot be created, an exception is raised."""
        try:
            info = StreamInfo(
                name="PythonResponse",
                stype="BCI_Essentials_Predictions",
                n_channels=1,
                sfreq=0,  # 0 means irregular rate
                dtype="string",
                source_id="pyp30042",
            )
            self.__outlet = StreamOutlet(info)
            self.__outlet.push_sample(["This is the python response stream"])
        except Exception:
            raise Exception("LslMessenger: could not create outlet")

    def ping(self):
        self.__outlet.push_sample(["ping"])

    def marker_received(self, marker):
        self.__outlet.push_sample(["marker received : {}".format(marker)])

    def prediction(self, prediction: Prediction):
        prediction_message = self.format_prediction_message(prediction)
        self.__outlet.push_sample([prediction_message])

    def format_prediction_message(self, prediction: Prediction) -> str:
        prediction_message = "%s:%s"
        labels = prediction.labels
        probabilities = prediction.probabilities

        if len(labels) > 1:
            probabilities = [np.average(x) for x in np.matrix_transpose(probabilities)]
            labels = [int(np.argmax(probabilities))]

        label_string = str(labels[0])
        probabilities_string = self.format_probabilities_string(probabilities)

        return prediction_message % (label_string, probabilities_string)

    def format_probabilities_string(self, probabilities: list, precision: int = 4) -> str:
        format_string = "%.{}f".format(precision)

        if type(probabilities[0]) is np.ndarray:
            probabilities = probabilities[0]

        if len(probabilities) == 1:
            return format_string % probabilities[0]
        else:
            return "[%s]" % " ".join([format_string % p for p in probabilities])