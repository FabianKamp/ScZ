import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time

# Load File Manager, handle file dependencies
M = MEGManager()
# Create Group Dictionary
GroupDict = {'Control':M.ControlIDs, 'FEP':M.FEPIDs}

def get_fc(FreqBand, Subject):
    assert M.exists(suffix='FC.npy', Sub=Subject, Freq=FreqBand), f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing'
    FileName = M.createFileName(suffix='FC.npy', Sub=Subject, Freq=FreqBand)  
    FilePath = M.createFilePath(M.FcDir, Subject, FileName)
    Fc = np.load(FilePath)  
    return Fc      

def get_edge_stats(M, GroupDict):        
    for FreqBand, Limits in config.FrequencyBands.items():
        print(f'Processing Freq {FreqBand}.')
        AllFCs = []
        Subjects = []
        for Group, IDs in GroupDict.items(): 
            GroupFCs = []  
            for n, Subject in enumerate(M.FEPIDs):
                # Appending Subject FC to Group FC list
                GroupFCs.append(get_fc(FreqBand, Subject))
                Subjects.append(Subject)
            AllFCs.extend(GroupFCs)
            GroupFCs = np.stack(GroupFCs)

            # Save stacked data
            FileName = M.createFileName(suffix='stacked-FCs.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'Data', Group, FileName)
            np.save(FilePath, GroupFCs)
            
            # Create Mean and Std FC
            MeanFC = np.mean(GroupFCs, axis=0)
            StdFC = np.std(GroupFCs, axis=0)
            
            # Save mean and std matrix
            FileName = M.createFileName(suffix='Mean-FC.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', Group, FileName)
            np.save(FilePath, MeanFC)
            
            FileName = M.createFileName(suffix='Std-FC.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'StDev', Group, FileName)
            np.save(FilePath, StdFC)

            # Calculate the mean and std edge value for each sub and save to array
            MeanEdge = np.mean(GroupFCs, axis=(1,2))
            StdEdge = np.std(GroupFCs, axis=(1,2))

            # Save mean and std edge values
            FileName = M.createFileName(suffix='Mean-Edge.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', Group, FileName)
            np.save(FilePath, MeanEdge)
            
            FileName = M.createFileName(suffix='Std-Edge.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'StDev', Group, FileName)
            np.save(FilePath, StdEdge)

        # Save stacked data
        FileName = M.createFileName(suffix='stacked-FCs.npy', Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'Data', FileName)
        np.save(FilePath, AllFCs)

    print('Processing finished.')

if __name__ == "__main__":
    start = time()
    get_edge_stats(M, GroupDict)
    end = time()
    print('Time: ', end-start)
