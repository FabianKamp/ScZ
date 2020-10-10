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

def preprocessing(M, SubjectList):
    # Iterate over all subjects
    for Subject in SubjectList:
        print(f'Processing Subject: {Subject}')
        Data, fsample = M.loadSignal(Subject)
        # Convert to Signal
        SubjectSignal = Signal(Data, fsample=fsample)
        # Downsample Signal
        ResampleNum = SubjectSignal.getResampleNum(TargetFreq=config.DownFreq) #Calculate number of resampled datapoints
        SubjectSignal.downsampleSignal(ResampleNum)

        # Filter data
        for FreqBand, Limits in config.FrequencyBands.items():
            print('Processing: ', FreqBand)
            # Check if
            if M.exists('FC', SubjectNum=Subject, CarrierFreq=FreqBand):
                continue

            # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, processes=2, LowPass=(config.mode=='lowpass'))

            # Save
            M.saveFC(FC, SubjectNum=Subject, CarrierFreq=FreqBand)

            # Create Minimum spannint Tree
            MST = net.network(FC).MST()
            M.saveMST(MST, SubjectNum=Subject, CarrierFreq=FreqBand)
    print('Preprocessing done.')

if __name__ == "__main__":
    preprocessing(M, SubjectList)