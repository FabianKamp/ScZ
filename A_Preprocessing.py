import Z_config as config
from utils.FileManager import MEGManager
import network as net
from utils.SignalAnalysis import Signal, Envelope

# Load File Manager that handles file dependencies.
M = MEGManager()

# Get list of subjects
if config.SubjectList:
    SubjectList = config.SubjectList
else:
    SubjectList = M.getSubjectList()

# Iterate over all subjects
for Subject in SubjectList:
    print(f'Processing Subject: {Subject}')

    # Loads Data and Sampling Frequency of one Subject at a time
    Signal, fsample = M.loadSignal(Subject)
    # Convert to Signal type
    SubjectSignal = Signal(Signal, fsample=fsample)
    # Downsample Signal
    ResampleNum = SubjectSignal.getResampleNum(TargetFreq=config.DownFreq) #Calculate number of resampled datapoints
    SubjectSignal.downsampleSignal(ResampleNum)

    # Filter data
    for FreqBand, Limits in config.FrequencyBands.items():
        print('Processing: ', FreqBand)
        # Check if
        if M.exists(SubjectNum=Subject, CarrierFreq=FreqBand, suffix='FC'):
            continue

        if config.mode == 'lowpass':
            # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, LowPass=True)
        elif config.mode == 'no-lowpass':
            # Get orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, LowPass=False)
        else:
            raise Exception('Mode not found. Change mode to lowpass or no-lowpass in config file.')

        # Save
        M.saveFC(FC, SubjectNum=Subject, CarrierFreq=FreqBand)

        # Create Minimum spannint Tree
        MST = net.network(FC).MST()
        M.saveMST(MST, SubjectNum=Subject, CarrierFreq=FreqBand)

print('Preprocessing done.')