from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.ssvep_fbcca_classifier import (
    SsvepFbCcaClassifier,
)
import numpy as np
from bci_essentials.utils.logger import Logger  # Logger wrapper

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()

paradigm = SsvepParadigm()
data_tank = DataTank()

logger = Logger(name=__name__)  

# Settings
target_frequencies = np.array([6.25, 10, 11.11111, 14.28571])
n_harmonics = 3  # Number of harmonics to use for SSVEP classification
fb_cutoffs = np.array([[i, 27] for i in range(3, 25, 2)])   # Filter bank cut-off frequencies
filter_order = 6
nsamples = 5 * 256  # 5 seconds of data at 256 Hz
fsample = 256.0  # Hz
concatenate_trials = True  # Concatenate trials for training

classifier = SsvepFbCcaClassifier(subset=["O1", "Oz", "O2"])
classifier.set_ssvep_settings(
    fsample=256.0,
    n_harmonics=n_harmonics,  # Number of harmonics to use for SSVEP classification
    target_freqs=target_frequencies,
    n_samples=nsamples,
    filter_bank=fb_cutoffs,
    concatenate_trials=concatenate_trials,
)

# Initialize the data class
test_ssvep = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

test_ssvep.setup(
    online=True,
    train_complete= True,
    train_lock= True,
)
test_ssvep.run()
