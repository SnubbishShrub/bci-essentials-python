from bci_essentials.io.lsl_sources import LslEegSource, LslMarkerSource
from bci_essentials.io.lsl_messenger import LslMessenger
from bci_essentials.bci_controller import BciController
from bci_essentials.paradigm.p300_paradigm import P300Paradigm
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.classification.erp_rg_classifier_hyperparamgridsearch import ErpRgClassifierHyperparamGridSearch

# create LSL sources, these will block until the outlets are present
eeg_source = LslEegSource()
marker_source = LslMarkerSource()
messenger = LslMessenger()
paradigm = P300Paradigm()
data_tank = DataTank()

# Set classifier settings ()
classifier = ErpRgClassifierHyperparamGridSearch()  # you can add a subset here

# Set some settings
classifier.set_p300_clf_settings(
    n_splits=5,
)

# Initialize the ERP
test_erp = BciController(
    classifier, eeg_source, marker_source, paradigm, data_tank, messenger
)

# Run main
test_erp.setup(
    online=True,
)
test_erp.run()
