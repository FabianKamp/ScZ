import Z_config as config
from utils.FileManager import FileManager
from utils.SignalAnalysis import Signal, Envelope

# Load File Manager that handles file dependencies.
M = FileManager()

# Get list of subjects
SubjectList = M.getSubjectList()

if config.mode == 'lowpass-FC':
    # Initiate MataDict
    MetaDict = {}

# Iterate over all subjects
for Subject in SubjectList:
    print(f'Processing Subject: {Subject}')

    # Loads Data of one Subject at a time
    SubjectData = M.loadData(Subject)
    # Convert to Signal type
    SubjectSignal = Signal(SubjectData['Signal'], fsample=SubjectData['SampleFreq'])
    # Downsample Signal
    ResampleNum = SubjectSignal.getResampleNum(TargetFreq=config.DownFreq) #Calculate number of resampled datapoints
    SubjectSignal.downsampleSignal(ResampleNum)

    if config.mode == 'lowpass-FC':
        # Create Sub Dictionary for each subject
        MetaDict[Subject] = {}

    # Filter data
    for FreqBand, Limits in config.FrequencyBands.items():
        print('Processing: ', FreqBand)
        # Check if
        if M.exists(suffix=config.mode, SubjectNum=Subject, CarrierFreq=FreqBand):
            continue

        if config.mode == 'lowpass-FC':
            # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, LowPass=True)

            # Compute Metastability and register into Dictionary
            SubjectEnvelope = SubjectSignal.getLowPassEnvelope(Limits)
            Metastability = Envelope(SubjectEnvelope).getMetastability()
            MetaDict[Subject][FreqBand] = Metastability

        elif config.mode == 'FC':
            # Get orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, LowPass=False)
        else:
            raise Exception('Mode not found. Change mode to lowpass-FC or FC in config file.')

        # Save
        M.saveNumpy(FC, suffix=config.mode, SubjectNum=Subject, CarrierFreq=FreqBand)

# Save Subject specific Metastability Dictionary
if config.mode == 'lowpass-FC':
    M.saveDataFrame(MetaDict, suffix='Metastability')

print('Preprocessing done.')