import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import itertools
from bct.nbs import nbs_bct
import pandas as pd
import scipy
from multiprocessing import Pool

class analyzer(MEGManager): 
    """
    Class to compute edge statistics of Functional Connectivity Matrix
    """
    def stack_fcs(self):
        """
        Stacks all FC matrices of each group into one matrix.
        """  
        print('Started stacking.')
        for FreqBand, Limits in self.FrequencyBands.items():
            print(f'Processing Freq {FreqBand}.')
            AllFCs = []
            Subjects = []
            for Group, SubjectList in self.GroupIDs.items(): 
                GroupFCs = []
                for Subject in SubjectList:  
                    # Appending Subject FC to Group FC list
                    FC = np.load(self.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
                    GroupFCs.append(FC)
                    Subjects.append(Subject)
                AllFCs.extend(GroupFCs)
                GroupFCs = np.stack(GroupFCs)

                # Save stacked data
                FileName = self.createFileName(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand)  
                FilePath = self.createFilePath(self.GroupStatsFC, 'Data', Group, FileName)
                np.save(FilePath, GroupFCs)
                
            # Save stacked data
            FileName = self.createFileName(suffix='stacked-FCs',filetype='.npy', Freq=FreqBand)  
            FilePath = self.createFilePath(self.GroupStatsFC, 'Data', FileName)
            np.save(FilePath, AllFCs)
        print('Finished stacking.')

    def calc_descr_stats(self):
        """
        Calculates mean, sd over groups and over subjects
        """
        print('Started calculating descr stats.')
        for Group, FreqBand in itertools.product(self.GroupIDs.keys(), self.FrequencyBands.keys()):
            # Find File and load file
            GroupFCs = np.load(self.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
            
            # Create Mean and Std FC
            MeanFC = np.mean(GroupFCs, axis=0)
            StdFC = np.std(GroupFCs, axis=0)
            
            # Save mean and std matrix
            FileName = self.createFileName(suffix='Mean-FC',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = self.createFilePath(self.GroupStatsFC, 'Mean', Group, FileName)
            np.save(FilePath, MeanFC)
            
            FileName = self.createFileName(suffix='Std-FC',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = self.createFilePath(self.GroupStatsFC, 'StDev', Group, FileName)
            np.save(FilePath, StdFC)

            # Calculate the mean and std edge value for each sub and save to array
            MeanEdge = np.mean(GroupFCs, axis=(1,2))
            StdEdge = np.std(GroupFCs, axis=(1,2))

            # Save mean and std edge values
            FileName = self.createFileName(suffix='Mean-Edge',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = self.createFilePath(self.GroupStatsFC, 'Mean', Group, FileName)
            np.save(FilePath, MeanEdge)
            
            FileName = self.createFileName(suffix='Std-Edge',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = self.createFilePath(self.GroupStatsFC, 'StDev', Group, FileName)
            np.save(FilePath, StdEdge)
        print('Finished calculating descr stats.')

    def create_mean_edge_df(self):
        """
        Creates DataFrame with mean edge values for plotting.
        """
        print('Creating mean edge dataframe.')
        DataDict = {'Subject':[], 'Group':[]}
        DataDict.update({key:[] for key in self.FrequencyBands.keys()})
        # iterate over all freqs and groups
        for Group, Subjects in self.GroupIDs.items():
            DataDict['Group'].extend([Group]*len(Subjects))
            DataDict['Subject'].extend(Subjects)
            for FreqBand in self.FrequencyBands.keys():
                Data = np.load(self.find(suffix='Mean-Edge',filetype='.npy', Group=Group, Freq=FreqBand))
                DataDict[FreqBand].extend(Data.tolist())
        DataFrame = pd.DataFrame(DataDict)
        FileName = self.createFileName(suffix='Subject-Mean_Edge-Weights',filetype='.pkl', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.GroupStatsFC, 'Mean', FileName)
        DataFrame.to_pickle(FilePath)
        print('Mean edge dataframe created.')

    def create_GBC_df(self):
        """
        Calculates GBC and saves it to Dataframe
        """ 
        GBCDict = {Region:[] for Region in self.RegionNames}
        GBCDict.update({'Subject':[], 'Frequency':[], 'Group':[], 'Avg. GBC':[]})
        
        for FreqBand, (Group, SubjectList) in itertools.product(self.FrequencyBands.keys(), self.GroupIDs.items()):
            print(f'Processing Freq {FreqBand}, Group {Group}')
            for Subject in SubjectList:  
                # Load FC
                FC = np.load(self.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
                # Calc GBC
                np.fill_diagonal(FC, 0)
                GBC = np.sum(FC, axis=-1) / FC.shape[0]
                mGBC = np.mean(GBC)
                for Region, GBCi in zip(self.RegionNames, GBC):
                    GBCDict[Region].append(GBCi)
                GBCDict['Group'].append(Group); GBCDict['Subject'].append(Subject)
                GBCDict['Frequency'].append(FreqBand); GBCDict['Avg. GBC'].append(mGBC)
        
        df = pd.DataFrame(GBCDict)
        FileName = self.createFileName(suffix='GBC',filetype='.pkl', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.GroupStatsFC, 'GBC', FileName)
        df.to_pickle(FilePath)

    def calc_nbs(self):
        """
        Caculates the NBS of Frequency Ranges defined in config FrequencyBands. 
        Before running this the FC matrices of each group have to be stacked.
        Saves p-val, components and null-samples to NetBasedStats. 
        NBS is calculated in parallel over Frequency bands.
        """
        print('Calculating NBS values')
        # Set seed to make results reproducible
        np.random.seed(0)
        processes = np.min([len(self.FrequencyBands),15])
        with Pool(processes) as p: 
            dfList = p.map(self._parallel_nbs, self.FrequencyBands.keys())
        pvaldf = pd.concat(dfList)
        FileName = self.createFileName(suffix='NBS-P-Values', filetype='.npy', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.GroupStatsFC, 'NetBasedStats', FileName)
        pd.to_pickle(pvaldf, FilePath)
        
    def _parallel_nbs(self, FreqBand): 
        """
        Calculates nbs for one Frequency band. Is called in calc_nbs. 
        """
        print(f'Processing {FreqBand}.')
        ResultDict = {'Freq':[], 'Threshold':[], 'P-Val':[], 'Component-File':[], 'Index':[]}
        GroupFCList=[]
        for Group in self.GroupIDs.keys():
            # Find File and load file 
            GroupFCs = np.load(self.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
            # Transform matrix for nbs (NxNxP with P being the number of subjects per group)
            tGroupFCs = np.moveaxis(GroupFCs, 0, -1)
            GroupFCList.append(tGroupFCs)
        
        # Set thresholds to iterate over with NBS algorithm
        thresholds = [2.056]
        
        i = 0
        for thresh in thresholds:
            print(f'Threshold: {thresh}')
            # Set Component File Path
            CompFileName = self.createFileName(suffix='Component-Adj',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            CompFilePath = self.createFilePath(self.GroupStatsFC, 'NetBasedStats', CompFileName)
            pval, adj, null = nbs_bct(GroupFCList[0], GroupFCList[1], thresh=2, k=1000)
            
            for idx, p in enumerate(pval): 
                ResultDict['Freq'].append(FreqBand); ResultDict['Threshold'].append(thresh)
                ResultDict['P-Val'].append(p); ResultDict['Component-File'].append(CompFilePath)
                ResultDict['Index'].append(idx)
                i += 1

            # Save Null-Sample and Component to File
            NullFileName = self.createFileName(suffix='Null-Sample',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            NullFilePath = self.createFilePath(self.GroupStatsFC, 'NetBasedStats', NullFileName)
            np.save(NullFilePath, null)
            np.save(CompFilePath, adj)

        # Return dataframe 
        df = pd.DataFrame(ResultDict, index=range(i))
        return df

    def test_avg_GBC(self): 
        import dabest
        df_long = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
        df_wide = pd.pivot_table(df_long, index=['Group', 'Subject'], columns='Frequency', values='Avg. GBC').reset_index()
        data = dabest.load(df_wide, idx=("Control", "FEP"), x='Group', y=32)
        print(type(data.mean_diff.statistical_tests))

    def test_region_GBC(self):
        """
        Compute regionwise t-test between global connectivity values
        """
        from mne.stats import bonferroni_correction, fdr_correction
        df = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
        # Dictionary to put all t-values, p-values
        tdict={Freq:[] for Freq in self.FrequencyBands}
        pdict={Freq:[] for Freq in self.FrequencyBands}
        testdict={Freq:[] for Freq in self.FrequencyBands}
        
        for Region in self.RegionNames: 
            df_pivot = df.pivot_table(index=['Subject', 'Group'], columns='Frequency', values=Region).reset_index()
            df_control = df_pivot[df_pivot['Group']=='Control']
            df_fep = df_pivot[df_pivot['Group']=='FEP']
            for Freq in self.FrequencyBands.keys(): 
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
        tdf = pd.DataFrame(tdict, index=self.RegionNames)
        pdf = pd.DataFrame(pdict, index=self.RegionNames)

        for Freq in self.FrequencyBands.keys(): 
            print(pdf[Freq])
            rej_bon, p_bon = bonferroni_correction(pdf[Freq], alpha=0.05)
            rej_fdr, p_fdr = fdr_correction(pdf[Freq], alpha=0.05, method='indep')
            #print(p_bon[:10])
            #print(p_fdr[:10])
        
    def comp_net_measures(self):
        """
        Function to compute network measures for each subject, results are safed into pd.DataFrame
        """
        print('Started calculating graph measures')
        # Load List of Data with FileManager
        dfList = []
        for Group, SubjectList in self.GroupIDs.items():
            print(f'Processing Group {Group}.')
            with Pool(processes=15) as p:
                result = p.starmap(self._parallel_net_measures, [(idx, Group, Subject, FreqBand) for idx, (Subject, FreqBand) in 
                enumerate(itertools.product(SubjectList, self.FrequencyBands.keys()))])
            dfList.extend(result)
        DataFrame = pd.concat(dfList,ignore_index=True)    
        # save DataFrame to File
        FileName = self.createFileName(suffix='Graph-Measures-'+self.net_version, filetype='.pkl')
        FilePath = self.createFilePath(self.NetMeasuresDir, FileName)
        DataFrame.to_pickle(FilePath)
        print('Finished calculating graph measures')  
    
    def _parallel_net_measures(self, idx, Group, Subject, FreqBand):
        """
        Computes Graph measures for one subject over all FrequencyBands
        """
        print(f'Processing {Subject}, {FreqBand} Band')
        # Init Result Dict
        ResultDict = {}
        ResultDict['Subject']=Subject
        ResultDict['Group']=Group
        ResultDict['Frequency']=FreqBand
        
        # Network Version
        version = self.net_version
        # Load FC matrix
        Data = np.load(self.find(suffix=version, filetype='.npy', Sub=Subject, Freq=FreqBand))

        # Remove negative edges from Network and set Diagonal to 0
        Data[Data<0] = 0
        np.fill_diagonal(Data, 0)
        network = net.network(Data, np.arange(Data.shape[-1]))

        # Calls network methods, appends result to Dict
        for Measure, FuncName in self.GraphMeasures.items():
            ResultDict[Measure]=getattr(network, FuncName)()
        
        df = pd.DataFrame(ResultDict, index=[idx])
        return df


if __name__ == "__main__":
    start = time()
    analyz = analyzer()
    analyz.stack_fcs()
    analyz.calc_descr_stats()
    analyz.create_mean_edge_df()
    analyz.create_GBC_df()
    #analyz.test_region_GBC()
    analyz.calc_nbs()
    analyz.comp_net_measures()
    end = time()
    print('Time: ', end-start)
