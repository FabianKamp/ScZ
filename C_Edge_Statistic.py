import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import itertools
from bct.nbs import nbs_bct

# Load File Manager, handle file dependencies
M = MEGManager()
# Create Group Dictionary
GroupDict = {'Control':M.ControlIDs, 'FEP':M.FEPIDs}

Stacking = False

def stack_fcs(M, GroupDict):  
    print('Started stacking.')
    for FreqBand, Limits in config.FrequencyBands.items():
        print(f'Processing Freq {FreqBand}.')
        AllFCs = []
        Subjects = []
        for Group, SubjectList in GroupDict.items(): 
            GroupFCs = []
            for Subject in SubjectList:  
                # Appending Subject FC to Group FC list
                FC = np.load(M.find(suffix='FC.npy', Sub=Subject, Freq=FreqBand))
                GroupFCs.append(get_fc(FreqBand, Subject))
                Subjects.append(Subject)
            AllFCs.extend(GroupFCs)
            GroupFCs = np.stack(GroupFCs)

            # Save stacked data
            FileName = M.createFileName(suffix='stacked-FCs.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'Data', Group, FileName)
            np.save(FilePath, GroupFCs)
            
        # Save stacked data
        FileName = M.createFileName(suffix='stacked-FCs.npy', Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'Data', FileName)
        np.save(FilePath, AllFCs)
    print('Finished stacking.')

def calc_descr_stats(M, GroupDict):
    print('Started calculating descr stats.')
    for Group, FreqBand in itertools.product(GroupDict.keys(), config.FrequencyBands.keys()):
        # Find File and load file
        GroupFCs = np.load(M.find(suffix='stacked-FCs.npy', Group=Group, Freq=FreqBand))
        
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
    print('Finished calculating descr stats.')

def calc_nbs(M, GroupDict):
    # Set seed to make results reproducible
    np.random.seed(0)
    for FreqBand in config.FrequencyBands.keys():
        print(f'Processing {FreqBand}.')
        GroupFCList=[]
        for Group in GroupDict.keys():
            # Find File and load file 
            GroupFCs = np.load(M.find(suffix='stacked-FCs.npy', Group=Group, Freq=FreqBand))
            # Transform matrix for nbs 
            tGroupFCs = np.moveaxis(GroupFCs, 0, -1)
            GroupFCList.append(tGroupFCs)
        pval, adj, null = nbs_bct(GroupFCList[0], GroupFCList[1], thresh=0.1, k=1000)
        print(pval.shape, adj.shape, null.shape)        
        #Save results to file
        FileName = M.createFileName(suffix='NBS-pval.npy', Freq=FreqBand)
        FilePath = M.createFilePath(M.GroupStatsFC, 'NetBasedStats', FileName)
        np.save(FilePath, pval)
        FileName = M.createFileName(suffix='Component-Adj.npy', Freq=FreqBand)
        FilePath = M.createFilePath(M.GroupStatsFC, 'NetBasedStats', FileName)
        np.save(FilePath, adj)

if __name__ == "__main__":
    start = time()
    if Stacking: 
        stack_fcs(M,GroupDict)
    #calc_descr_stats(M, GroupDict)
    calc_nbs(M,GroupDict)
    end = time()
    print('Time: ', end-start)
