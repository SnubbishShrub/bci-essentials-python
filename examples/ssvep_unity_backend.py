from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.ssvep_basic_tf_classifier import (
    SsvepBasicTrainFreeClassifier
)

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = SsvepParadigm()
data_tank = DataTank()

# Define the classifier    
classifier = SsvepBasicTrainFreeClassifier()
classifier.set_ssvep_settings(
    sampling_freq=256.0, target_freqs= [7.692307, 10, 12.5, 14.28571])


# Initialize the data class
test_ssvep = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

test_ssvep.setup(
    online=True,
    train_complete= True,
)
test_ssvep.run()
