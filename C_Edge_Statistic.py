import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net

# Load File Manager, handle file dependencies
M = MEGManager()
GroupDict = {'Control':M.ControlIDs, 'FEP':M.FEPIDs}

def get_fc(FreqBand, Subject):
    assert M.exists(suffix='FC.npy', Sub=Subject, Freq=FreqBand), f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing'
    FileName = M.createFileName(suffix='FC.npy', Sub=Subject, Freq=FreqBand)  
    FilePath = M.createFileName(M.FcDir, Subject, FileName)
    Fc = np.load(FilePath)  
    return Fc      

def get_edge_stats(M):        
    for FreqBand, Limits in config.FrequencyBands.items():
        print(f'Processing Freq {FreqBand}.')
        for Group, IDs in GroupDict.items() 
            CombinedData = np.empty((94,94,len(IDs)))        
            for n, Subject in enumerate(M.FEPIDs):
                CombinedData[n,:,:] = get_fc(FreqBand, Subject)
            FileName = M.createFileName(suffix='FC_mean.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFileName(M.GroupedFunctCon, Group, FileName)
            np.save(FilePath, CombinedData)

    FileName=M.createFileName(suffix='FC_mean.npy', )

