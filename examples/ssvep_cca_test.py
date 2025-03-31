# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021
# Editted by Emily Schrag on 03/31/2025

import os
from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.utils.logger import Logger  # Logger wrapper
from bci_essentials.classification.ssvep_basic_tf_classifier import (
    SsvepBasicTrainFreeClassifier,
)

logger = Logger(name=__name__)

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "sub-Eli_ses-S003_task-T1_run-001_eeg.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)
paradigm = SsvepParadigm(live_update=False, iterative_training=False) # Setting live_upadate to False means it will classify each trial rather than each epoch (full 5 seconds rather than every second 5 time)
data_tank = DataTank()

# Define the classifier
classifier = SsvepBasicTrainFreeClassifier(subset=[])
classifier.set_ssvep_settings(sampling_freq=256.0, target_freqs=[7.692307, 10, 12.5, 14.28571])

test_ssvep = BciController(classifier, eeg_source, marker_source, paradigm, data_tank)

test_ssvep.setup(
    online=False,
    train_complete=True,  # Set to True to run the classifier on the entire dataset
)

test_ssvep.run()


