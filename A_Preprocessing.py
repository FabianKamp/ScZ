import Z_config as config
from utils.FileManager import MEGManager
import network as net
from utils.SignalAnalysis import Signal, Envelope

# Load File Manager, handle file dependencies
M = MEGManager()

# Get list of subjects if defined or load all subjects
if config.SubjectList:
    SubjectList = config.SubjectList
else:
    SubjectList = M.getSubjectList()

# Iterate over all subjects
for Subject in SubjectList:
    print(f'Processing Subject: {Subject}')
    Signal, fsample = M.loadSignal(Subject)
    # Convert to Signal
    SubjectSignal = Signal(Signal, fsample=fsample)
    # Downsample Signal
    ResampleNum = SubjectSignal.getResampleNum(TargetFreq=config.DownFreq) #Calculate number of resampled datapoints
    SubjectSignal.downsampleSignal(ResampleNum)

    # Filter data
    for FreqBand, Limits in config.FrequencyBands.items():
        print('Processing: ', FreqBand)
        # Check if
        if M.exists(Sub=Subject, CarrierFreq=FreqBand, suffix='FC'):
            continue

        # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
        if __name__ == "__main__":
            FC = SubjectSignal.getOrthFC(Limits, processes=2, LowPass=(config.mode=='lowpass'))

        else:
            raise Exception('Mode not found. Change mode to lowpass or no-lowpass in config file.')

        # Save
        M.saveFC(FC, Sub=Subject, CarrierFreq=FreqBand)

        # Create Minimum spannint Tree
        MST = net.network(FC).MST()
        M.saveMST(MST, Sub=Subject, CarrierFreq=FreqBand)

print('Preprocessing done.')