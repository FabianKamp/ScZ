import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import itertools
from bct.nbs import nbs_bct
import pandas as pd
import scipy

# Load File Manager, handle file dependencies
M = MEGManager()
Stacking = True

def stack_fcs():
    """
    Stacks all FC matrices of each group into one matrix.
    """  
    print('Started stacking.')
    for FreqBand, Limits in config.FrequencyBands.items():
        print(f'Processing Freq {FreqBand}.')
        AllFCs = []
        Subjects = []
        for Group, SubjectList in M.GroupIDs.items(): 
            GroupFCs = []
            for Subject in SubjectList:  
                # Appending Subject FC to Group FC list
                FC = np.load(M.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
                GroupFCs.append(FC)
                Subjects.append(Subject)
            AllFCs.extend(GroupFCs)
            GroupFCs = np.stack(GroupFCs)

            # Save stacked data
            FileName = M.createFileName(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = M.createFilePath(M.GroupStatsFC, 'Data', Group, FileName)
            np.save(FilePath, GroupFCs)
            
        # Save stacked data
        FileName = M.createFileName(suffix='stacked-FCs',filetype='.npy', Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'Data', FileName)
        np.save(FilePath, AllFCs)
    print('Finished stacking.')

def calc_descr_stats():
    print('Started calculating descr stats.')
    for Group, FreqBand in itertools.product(M.GroupIDs.keys(), config.FrequencyBands.keys()):
        # Find File and load file
        GroupFCs = np.load(M.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
        
        # Create Mean and Std FC
        MeanFC = np.mean(GroupFCs, axis=0)
        StdFC = np.std(GroupFCs, axis=0)
        
        # Save mean and std matrix
        FileName = M.createFileName(suffix='Mean-FC',filetype='.npy', Group=Group, Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', Group, FileName)
        np.save(FilePath, MeanFC)
        
        FileName = M.createFileName(suffix='Std-FC',filetype='.npy', Group=Group, Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'StDev', Group, FileName)
        np.save(FilePath, StdFC)

        # Calculate the mean and std edge value for each sub and save to array
        MeanEdge = np.mean(GroupFCs, axis=(1,2))
        StdEdge = np.std(GroupFCs, axis=(1,2))

        # Save mean and std edge values
        FileName = M.createFileName(suffix='Mean-Edge',filetype='.npy', Group=Group, Freq=FreqBand)  
        FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', Group, FileName)
        np.save(FilePath, MeanEdge)
        
        FileName = M.createFileName(suffix='Std-Edge',filetype='.npy', Group=Group, Freq=FreqBand)  
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
            Data = np.load(M.find(suffix='Mean-Edge',filetype='.npy', Group=Group, Freq=FreqBand))
            DataDict[FreqBand].extend(Data.tolist())
    DataFrame = pd.DataFrame(DataDict)
    FileName = M.createFileName(suffix='Subject-Mean_Edge-Weights',filetype='.pkl', Freq=config.Frequencies)
    FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', FileName)
    DataFrame.to_pickle(FilePath)
    print('Mean edge dataframe created.')

def create_GBC_df(): 
    M = MEGManager()
    GBCDict = {Region:[] for Region in M.RegionNames}
    GBCDict.update({'Subject':[], 'Frequency':[], 'Group':[], 'Avg. GBC':[]})
    
    for FreqBand, (Group, SubjectList) in itertools.product(config.FrequencyBands.keys(), M.GroupIDs.items()):
        print(f'Processing Freq {FreqBand}, Group {Group}')
        for Subject in SubjectList:  
            # Load FC
            FC = np.load(M.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
            # Calc GBC
            np.fill_diagonal(FC, 0)
            GBC = np.sum(FC, axis=-1) / FC.shape[0]
            mGBC = np.mean(GBC)
            for Region, GBCi in zip(M.RegionNames, GBC):
                GBCDict[Region].append(GBCi)
            GBCDict['Group'].append(Group); GBCDict['Subject'].append(Subject)
            GBCDict['Frequency'].append(FreqBand); GBCDict['Avg. GBC'].append(mGBC)
    
    df = pd.DataFrame(GBCDict)
    FileName = M.createFileName(suffix='GBC',filetype='.pkl', Freq=config.Frequencies)
    FilePath = M.createFilePath(M.GroupStatsFC, 'GBC', FileName)
    df.to_pickle(FilePath)

def calc_nbs():
    """
    Caculates the NBS of Frequency Ranges defined in config FrequencyBands. 
    Before running this the FC matrices of each group have to be stacked.
    Saves p-val, components and null-samples to NetBasedStats.
    """
    # Set seed to make results reproducible
    np.random.seed(0)
    for FreqBand in config.FrequencyBands.keys():
        print(f'Processing {FreqBand}.')
        GroupFCList=[]
        for Group in M.GroupIDs.keys():
            # Find File and load file 
            GroupFCs = np.load(M.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
            # Transform matrix for nbs (NxNxP with P being the number of subjects per group)
            tGroupFCs = np.moveaxis(GroupFCs, 0, -1)
            GroupFCList.append(tGroupFCs)
        # Set thresholds to iterate over with NBS algorithm
        thresholds = [2.056]
        for thresh in thresholds:
            print(f'Threshold: {thresh}')
            pval, adj, null = nbs_bct(GroupFCList[0], GroupFCList[1], thresh=2, k=1000)      
            #Save results to file
            FileName = M.createFileName(suffix='NBS-pval',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            FilePath = M.createFilePath(M.GroupStatsFC, 'NetBasedStats', FileName)
            np.save(FilePath, pval)
            FileName = M.createFileName(suffix='Component-Adj',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            FilePath = M.createFilePath(M.GroupStatsFC, 'NetBasedStats', FileName)
            np.save(FilePath, adj)
            FileName = M.createFileName(suffix='Null-Sample',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            FilePath = M.createFilePath(M.GroupStatsFC, 'NetBasedStats', FileName)
            np.save(FilePath, null)

def test_avg_GBC(): 
    import dabest
    df_long = pd.read_pickle(M.find(suffix='GBC',filetype='.pkl', Freq=config.Frequencies))
    df_wide = pd.pivot_table(df_long, index=['Group', 'Subject'], columns='Frequency', values='Avg. GBC').reset_index()
    data = dabest.load(df_wide, idx=("Control", "FEP"), x='Group', y=32)
    print(type(data.mean_diff.statistical_tests))

def test_region_GBC():
    from mne.stats import bonferroni_correction, fdr_correction
    df = pd.read_pickle(M.find(suffix='GBC',filetype='.pkl', Freq=config.Frequencies))
    # Dictionary to put all t-values, p-values
    tdict={Freq:[] for Freq in config.FrequencyBands}
    pdict={Freq:[] for Freq in config.FrequencyBands}
    testdict={Freq:[] for Freq in config.FrequencyBands}
    
    for Region in M.RegionNames: 
        df_pivot = df.pivot_table(index=['Subject', 'Group'], columns='Frequency', values=Region).reset_index()
        df_control = df_pivot[df_pivot['Group']=='Control']
        df_fep = df_pivot[df_pivot['Group']=='FEP']
        for Freq in config.FrequencyBands.keys(): 
            # Test for equal variance
            _, pval = scipy.stats.levene(df_fep[Freq], df_control[Freq])
            if np.abs(pval) < 0.1:  
                print('Variance not equal, performed welch test.')
                t, pval = scipy.stats.ttest_ind(df_fep[Freq], df_control[Freq], equal_var=False)
                testdict[Freq].append('welch-test')
            else: 
                t, pval = scipy.stats.ttest_ind(df_fep[Freq], df_control[Freq], equal_var=True)
                testdict[Freq].append('t-test')
            tdict[Freq].append(t); pdict[Freq].append(pval)
    tdf = pd.DataFrame(tdict, index=M.RegionNames)
    pdf = pd.DataFrame(pdict, index=M.RegionNames)

    for Freq in config.FrequencyBands.keys(): 
        print(pdf[Freq])
        rej_bon, p_bon = bonferroni_correction(pdf[Freq], alpha=0.05)
        rej_fdr, p_fdr = fdr_correction(pdf[Freq], alpha=0.05, method='indep')
        #print(p_bon[:10])
        #print(p_fdr[:10])
    
if __name__ == "__main__":
    start = time()
    stack_fcs()
    calc_descr_stats()
    create_mean_edge_df()
    create_GBC_df()
    #test_region_GBC()
    calc_nbs()
    end = time()
    print('Time: ', end-start)
