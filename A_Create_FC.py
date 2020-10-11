import Z_config as config
from utils.FileManager import MEGManager
import network as net
from utils.SignalAnalysis import Signal, Envelope
from time import time

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
                print(f'FC of {Subject} Freq {FreqBand} exists.')
                continue

            # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getOrthFC(Limits, processes=5)

            # Save
            FileName = M.createFileName(suffix='FC.npy', Sub=SubjectNum, Freq=CarrierFreq)
            FilePath = M.createFilePath(M.FcDir, SubjectNum, FileName)
            np.save(FilePath, Data)

    print('Preprocessing done.')

if __name__ == "__main__":
    start = time()
    preprocessing(M, SubjectList)
    end = time()
    print('Time: ', end-start)