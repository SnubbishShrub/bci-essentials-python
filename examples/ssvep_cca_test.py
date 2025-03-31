# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021
# Editted by Emily Schrag on 03/31/2025

import os
from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.ssvep_paradigm import SsvepParadigm
from bci_essentials.data_tank.data_tank import DataTank

# Identify the file to simulate
# Filename assumes the data is within a subfolder called "data" located
# within the same folder as this script
filename = os.path.join("data", "sub-Eli_ses-S003_task-T1_run-001_eeg.xdf")
eeg_source = XdfEegSource(filename)
marker_source = XdfMarkerSource(filename)
paradigm = SsvepParadigm(live_update=False, iterative_training=False)
data_tank = DataTank()

