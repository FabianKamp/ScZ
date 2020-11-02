import Z_config as config
from utils.FileManager import MEGManager
import network as net
from utils.SignalAnalysis import Signal
from time import time
import numpy as np

def preprocessing():
    """
    Function to perform preprocessing, generates FC matrix for each subject taking the configurations of the config file
    """
    M = MEGManager()
    # Iterate over all subjects
    for Subject in M.SubjectList:
        print(f'Processing Subject: {Subject}')
        Data, fsample = M.loadMatFile(Subject)        
        # Convert to Signal
        SubjectSignal = Signal(Data, fsample=fsample)        
        # Downsample Signal
        SubjectSignal.downsampleSignal(TargetFreq=M.DownFreq)

        # Filter data
        for FreqBand, Limits in M.FrequencyBands.items():
            print('Processing: ', FreqBand)
            # Check if
            if M.exists(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand):
                print(f'Overwriting FC of {Subject} Freq {FreqBand}.')

            # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
            FC = SubjectSignal.getFC(Limits, processes=5)

            # Save
            FileName = M.createFileName(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand)
            FilePath = M.createFilePath(M.FcDir, Subject, FileName)
            np.save(FilePath, FC)
    print('Preprocessing done.')

if __name__ == "__main__":
    start = time()
    preprocessing()
    end = time()
    print('Time: ', end-start)