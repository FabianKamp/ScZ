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
from V_Visualization import visualization

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
            Subjects = []
            for Group, SubjectList in self.GroupIDs.items(): 
                GroupFCs = []
                for Subject in SubjectList:  
                    # Appending Subject FC to Group FC list
                    FC = np.load(self.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
                    GroupFCs.append(FC)
                    Subjects.append(Subject)
                GroupFCs = np.stack(GroupFCs)

                # Save stacked data
                FileName = self.createFileName(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand)  
                FilePath = self.createFilePath(self.EdgeStatsDir, 'Stacked-Data', Group, FileName)
                np.save(FilePath, GroupFCs)
        print('Finished stacking.')

    def calc_group_mean_fc(self):
        """
        Calculates mean, sd over groups and over subjects
        """
        print('Started calculating descr stats.')
        for Group, FreqBand in itertools.product(self.GroupIDs.keys(), self.FrequencyBands.keys()):
            # Find File and load file
            GroupFCs = np.load(self.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
            
            # Create Mean FC
            MeanFC = np.mean(GroupFCs, axis=0)
            
            # Save mean matrix
            FileName = self.createFileName(suffix='Group-Mean-FC',filetype='.npy', Group=Group, Freq=FreqBand)  
            FilePath = self.createFilePath(self.EdgeStatsDir, 'Group-Mean-FC', Group, FileName)
            np.save(FilePath, MeanFC)

        print('Finished calculating descr stats.')

    def calc_mean_edge(self):
        """
        Creates DataFrame with mean edge values for plotting.
        """
        print('Creating mean edge dataframe.')
        DataDict = {'Subject':[], 'Group':[]}
        DataDict.update({key:[] for key in self.FrequencyBands.keys()})
        
        # iterate over all subjects and frequencies and save mean edge value to DataDict
        for Group, Subjects in self.GroupIDs.items():
            for Subject in Subjects:
                DataDict['Group'].append(Group); DataDict['Subject'].append(Subject)
                for FreqBand in self.FrequencyBands.keys():
                    # load FC and get mean
                    FC = np.load(self.find(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand))
                    edge_mean = np.mean(FC)                    
                    DataDict[FreqBand].append(edge_mean)

        DataFrame = pd.DataFrame(DataDict)
        # Save to 
        FileName = self.createFileName(suffix='Mean-Edge-Weights',filetype='.pkl', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.EdgeStatsDir, 'Mean-Edge-Weights', FileName)
        DataFrame.to_pickle(FilePath)
        
        print('Mean edge dataframe created.')

    def calc_GBC(self):
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
        FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', FileName)
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
        FileName = self.createFileName(suffix='NBS-P-Values', filetype='.pkl', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'P-Values', FileName)
        pd.to_pickle(pvaldf, FilePath)
        
    def _parallel_nbs(self, FreqBand): 
        """
        Calculates nbs for one Frequency band. Is called in calc_nbs. 
        :return dataframe with pvals of 
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
        
        i = 0 # Counter to set index of dataframe 
        for thresh in thresholds:
            print(f'Threshold: {thresh}')
            # Set Component File Path
            CompFileName = self.createFileName(suffix='Component-Adj', filetype='.npy', Freq=FreqBand, Thresh=thresh)
            CompFilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'Components', CompFileName)

            pval, adj, null = nbs_bct(GroupFCList[0], GroupFCList[1], thresh=2, k=1000)
            
            for idx, p in enumerate(pval): 
                ResultDict['Freq'].append(FreqBand); ResultDict['Threshold'].append(thresh)
                ResultDict['P-Val'].append(p); ResultDict['Component-File'].append(CompFilePath)
                ResultDict['Index'].append(idx)
                i += 1

            # Save Null-Sample and Component to File
            NullFileName = self.createFileName(suffix='Null-Sample',filetype='.npy', Freq=FreqBand, Thresh=thresh)
            NullFilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'Null-Sample', NullFileName)
            
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
        
    def calc_net_measures(self):
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
        FilePath = self.createFilePath(self.NetMeasuresDir, self.net_version, FileName)
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
    viz = visualization()
    
    analyz.stack_fcs()
    viz.plot_edge_dist()
    
    analyz.calc_group_mean_fc()
    viz.plot_group_mean_fc()
    
    analyz.calc_mean_edge()
    viz.plot_mean_edge()
    viz.plot_cross_hemi_corr()
    
    analyz.calc_GBC()
    viz.plot_avg_GBC()
    
    analyz.calc_nbs()
    
    analyz.calc_net_measures()
    viz.plot_net_measures()
    end = time()
    print('Time: ', end-start)
