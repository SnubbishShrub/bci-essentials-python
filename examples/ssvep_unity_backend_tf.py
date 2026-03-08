# Note: must use 2 second epochs

from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.classification.ssvep_basic_tf_classifier import (
    SsvepBasicTrainFreeClassifier,
)

target_frequencies = [6.0, 7.5, 8.2, 10.0]


# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = SsvepParadigm(live_update=True)
data_tank = DataTank()

# Define the classifier
classifier = SsvepBasicTrainFreeClassifier()

# Initialize the EEG Data
test_ssvep = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

# set train complete to true so that predictions will be allowed
classifier.set_ssvep_settings(eeg_source.fsample, target_frequencies)
test_ssvep.setup(online=True, train_complete=True)

# Run
test_ssvep.run()
