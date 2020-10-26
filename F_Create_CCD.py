import Z_config as config
from utils.FileManager import MEGManager
import network as net
from utils.SignalAnalysis import Signal, Envelope
from time import time
import numpy as np
import itertools

# Load File Manager, handle file dependencies

def preprocessing(M):
    M = MEGManager()
    # Iterate over all subjects
    for (FreqBand, Limits), (Group, SubjectList) in itertools.product(config.FrequencyBands.items(), M.GroupIDs.items()): 
        # init CCD
        downfreq = 30
        n_timepoints = 60*5*downfreq
        CCD = np.zeros(n_timepoints, n_timepoints).astype('float32')
        
        for Subject in SubjectList:
            print(f'Processing Subject: {Subject}')
            Data, fsample = M.loadSignal(Subject)     

            # Convert to Signal
            SubjectSignal = Signal(Data, fsample=fsample)
            # Downsample Signal
            SubjectSignal.downsampleSignal(TargetFreq=downfreq)
            # Compute CCD              
            sCCD = SubjectSignal.getCCD(Limits)
            # Create group CCD
            max_idx = min(sCCD.shape[0], CCD.shape[0])
            sCCD = sCCD[:max_idx,:max_idx]; CCD = CCD[:max_idx, :max_idx]
            CCD += sCCD

        CCD /= len(SubjectList)

        # Save
        FileName = M.createFileName(suffix='CCD', filetype='.npy', no_conn=True, Group=Group, Freq=FreqBand)
        FilePath = M.createFilePath(M.CCDDir, Group, FileName)
        np.save(FilePath, CCD)

    print('Preprocessing done.')