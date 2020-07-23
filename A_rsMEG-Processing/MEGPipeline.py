import numpy as np
from DataManagerMEG import FileManager
from FrequencyFilter import BandPassFilter
from SignalEnvelope import SignalEnvelope


# Load data list with File Manager
M = FileManager()

for Subject in M.SubjectList:
    print('Processing Subject: ', Subject)
    # Loads Data of one Subject at a time
    SubjectData = M.loadSubjectFile(Subject)
    # Create Metastability Dictionary for each subject
    MetaDict = {}
    # Filter data
    CarrierFrequencies = list(np.arange(2, 48, 2)) + list(np.arange(64,92,2))
    for CarrierFreq in CarrierFrequencies:
        # Check if
        if M.exists(Subject, CarrierFreq):
            continue

        # Band Pass Filter Signal
        FilteredSignal = BandPassFilter(SubjectData, CarrierFreq)

        # Envelope FC
        Envelope = SignalEnvelope(FilteredSignal, LowPassFreq=0.2)
        LowFC = Envelope.getLowEnvCorr()
        FC = Envelope.getEnvCorr()

        # Save
        M.save(Envelope.LowEnv, Subject, CarrierFreq=CarrierFreq, Type='LowEnv')
        M.save(FC, Subject, CarrierFreq=CarrierFreq, Type='FC')
        M.save(LowFC, Subject, CarrierFreq=CarrierFreq, Type='LowFC')

        # Metastability and register into Dictionary
        Metastability = Envelope.getMetastability()
        MetaDict[CarrierFreq] = Metastability

    # Save Subject specific Metastability Dictionary
    M.save(MetaDict, Subject, Type='Metastability')

print('Processing done.')

