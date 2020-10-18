import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import itertools
from bct.nbs import nbs_bct
import pandas as pd

# Load File Manager, handle file dependencies
M = MEGManager()
Stacking = True

def stack_fcs():  
    print('Started stacking.')
    for FreqBand, Limits in config.FrequencyBands.items():
        print(f'Processing Freq {FreqBand}.')
        AllFCs = []
        Subjects = []
        for Group, SubjectList in M.GroupIDs.items(): 
            GroupFCs = []
            for Subject in SubjectList:  
                # Appending Subject FC to Group FC list
                FC = np.load(M.find(suffix='FC.npy', Sub=Subject, Freq=FreqBand))
                GroupFCs.append(FC)
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

def calc_descr_stats():
    print('Started calculating descr stats.')
    for Group, FreqBand in itertools.product(M.GroupIDs.keys(), config.FrequencyBands.keys()):
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

def create_mean_edge_df():
    print('Creating mean edge dataframe.')
    DataDict = {'Subject':[], 'Group':[]}
    DataDict.update({key:[] for key in config.FrequencyBands.keys()})
    # iterate over all freqs and groups
    for Group, Subjects in M.GroupIDs.items():
        DataDict['Group'].extend([Group]*len(Subjects))
        DataDict['Subject'].extend(Subjects)
        for FreqBand in config.FrequencyBands.keys():
            Data = np.load(M.find(suffix='Mean-Edge.npy', Group=Group, Freq=FreqBand))
            DataDict[FreqBand].extend(Data.tolist())
    DataFrame = pd.DataFrame(DataDict)
    FileName = M.createFileName(suffix='Subject-Mean_Edge-Weights.pkl')
    FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', FileName)
    DataFrame.to_pickle(FilePath)
    print('Mean edge dataframe created.')

def calc_nbs():
    # Set seed to make results reproducible
    np.random.seed(0)
    for FreqBand in config.FrequencyBands.keys():
        print(f'Processing {FreqBand}.')
        GroupFCList=[]
        for Group in M.GroupIDs.keys():
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
        stack_fcs()
    calc_descr_stats()
    create_mean_edge_df()
    #calc_nbs(M)
    end = time()
    print('Time: ', end-start)
