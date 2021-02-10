import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import time
import itertools
from bct.nbs import nbs_bct
import pandas as pd
import scipy
from multiprocessing import Pool
from V_Visualization import visualization
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import dabest
from mne.stats import bonferroni_correction, fdr_correction

class analyzer(MEGManager): 
	"""
	Class to compute edge statistics of Functional Connectivity Matrix
	"""
	def __init__(self):
		# Load visualization
		super().__init__()
		self.viz = visualization()
	
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
	
	def excl_outliers(self, method='IQR'):
		"""
		Calculates outliers based on mean edge value using the +/- 3*IQR as threshold.
		"""
		mean_edge_df = pd.read_pickle(self.find(suffix='Mean-Edge-Weights',filetype='.pkl', Freq=self.Frequencies))
		for Group in ['Control','FEP']:
			df_split = mean_edge_df[mean_edge_df['Group']==Group]
			if method=='IQR':
				# Get inter quartile range for each group
				Q1 = df_split.quantile(0.25)
				Q3 = df_split.quantile(0.75)
				IQR = Q3 - Q1
				mask = (df_split > Q3 + 3 * IQR)|(df_split < Q1 - 3 * IQR)
				extreme = df_split[mask.any(axis=1)]
				extreme = set(extreme['Subject'])
			elif method=='Z-scores': 
				# Using Z-Scores to identify outliers				
				mask = np.abs(scipy.stats.zscore(df_split[self.FrequencyBands.keys()], axis=0)) > 3
				extreme = df_split[mask.any(axis=-1)]
				extreme = set(extreme['Subject'])
			
			#Remove Extreme Values from Subject Lists
			if len(extreme)>0:
				print(f'Removing Subjects: {extreme} from Group: {Group}')
				self.SubjectList = list(set(self.SubjectList) - extreme)
				self.GroupIDs[Group] = list(set(self.GroupIDs[Group])-extreme)
		
		# Time to read outliers
		time.sleep(5)

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
		thresholds = [3.0, 2.5, 2.0]   
		with Pool(20) as p: 
			dfList = p.starmap(self._parallel_nbs, itertools.product(self.FrequencyBands.keys(), thresholds))
		pvaldf = pd.concat(dfList)
		FileName = self.createFileName(suffix='NBS-P-Values', filetype='.csv', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'P-Values', FileName)
		pvaldf.to_csv(FilePath)
		
	def _parallel_nbs(self, FreqBand, thresh): 
		"""
		Calculates nbs for one Frequency band. Is called in calc_nbs. 
		:return dataframe with pvals of 
		"""
		print(f'Processing {FreqBand}, Thresh: {thresh}')
		ResultDict = {'Freq':[], 'Threshold':[], 'P-Val':[], 'Component-File':[], 'Index':[]}
		GroupFCList=[]
		for Group in self.GroupIDs.keys():
			# Find File and load file 
			GroupFCs = np.load(self.find(suffix='stacked-FCs',filetype='.npy', Group=Group, Freq=FreqBand))
			# Transform matrix for nbs (NxNxP with P being the number of subjects per group)
			tGroupFCs = np.moveaxis(GroupFCs, 0, -1)
			GroupFCList.append(tGroupFCs)   

		# Set Component File Path
		CompFileName = self.createFileName(suffix='Component-Adj', filetype='.npy', Freq=FreqBand, Thresh=thresh)
		CompFilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'Components', CompFileName)

		pval, adj, null = nbs_bct(GroupFCList[0], GroupFCList[1], thresh=thresh, k=1000)
		print('Adjacency Shape: ', adj.shape)
		
		for idx, p in enumerate(pval): 
			ResultDict['Freq'].append(FreqBand); ResultDict['Threshold'].append(thresh)
			ResultDict['P-Val'].append(p); ResultDict['Component-File'].append(CompFilePath)
			ResultDict['Index'].append(idx+1)

		# Save Null-Sample and Component to File
		NullFileName = self.createFileName(suffix='Null-Sample',filetype='.npy', Freq=FreqBand, Thresh=thresh)
		NullFilePath = self.createFilePath(self.EdgeStatsDir, 'NetBasedStats', 'Null-Sample', NullFileName)
		
		np.save(NullFilePath, null)
		np.save(CompFilePath, adj)

		# Return dataframe 
		df = pd.DataFrame(ResultDict, index=ResultDict['Index'])
		return df

	def dabest_avg_GBC(self): 
		"""
		Function to calculate effect size and t/p value for average GBC. 
		"""
		df_long = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
		df_wide = pd.pivot_table(df_long, index=['Group', 'Subject'], columns='Frequency', values='Avg. GBC').reset_index()
		res_list = []
		for Freq in self.FrequencyBands.keys():
			analysis = dabest.load(df_wide, idx=("Control", "FEP"), x='Group', y=Freq, ci=90)
			results = analysis.mean_diff.results
			results.insert(loc=0, column='Frequency', value=Freq)
			res_list.append(results)
		result_df = pd.concat(res_list)

		# Save Pickle
		FileName = self.createFileName(suffix='Mean-GBC-DABEST', filetype='.pkl', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', 'Stats', FileName)
		result_df.to_pickle(FilePath)

		# Save CSV
		FileName = self.createFileName(suffix='Mean-GBC-DABEST', filetype='.csv', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', 'Stats', FileName)
		result_df.to_csv(FilePath)

	def test_region_GBC(self):
		"""
		Compute regionwise t-test between global connectivity values
		"""
		from mne.stats import bonferroni_correction, fdr_correction
		df = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
		# Result Dictionary
		testdict={'Frequency':[], 'Region':[], 't-value':[], 'p-value':[], 'welch-t-value':[], 'welch-p-value':[], 'levene-p-value':[]}
		print('Started Statstical Test.')      
		for Region in self.RegionNames:
			print(f'Testing {Region}') 
			df_pivot = df.pivot_table(index=['Subject', 'Group'], columns='Frequency', values=Region).reset_index()
			df_control = df_pivot[df_pivot['Group']=='Control']
			df_fep = df_pivot[df_pivot['Group']=='FEP']
			for Freq in self.FrequencyBands.keys():
				testdict['Frequency'].append(Freq)
				testdict['Region'].append(Region)				
				# Test for equal variance, levene test
				_, pval = scipy.stats.levene(df_fep[Freq], df_control[Freq])
				testdict['levene-p-value'].append(pval)
				# welch test if variances are not equal
				t, pval = scipy.stats.ttest_ind(df_fep[Freq], df_control[Freq], equal_var=False)
				testdict['welch-t-value'].append(t)
				testdict['welch-p-value'].append(pval)
				# Standard t-test
				t, pval = scipy.stats.ttest_ind(df_fep[Freq], df_control[Freq], equal_var=True)           
				testdict['t-value'].append(t)
				testdict['p-value'].append(pval)
		
		# Transform to DataFrame
		df = pd.DataFrame(testdict)

		print('Bonferroni Correction.')	
		# Calculate Bonferroni and FDR correction
		# Set up columns
		df['Bonferroni'] = df['FDR'] = np.NaN


		for Freq in self.FrequencyBands.keys(): 
			df_split = df[df['Frequency'] == Freq]
			_, p_bon = bonferroni_correction(df_split['p-value'], alpha=0.05)
			_, p_fdr = fdr_correction(df_split['p-value'], alpha=0.05, method='indep')
			df.loc[df['Frequency'] == Freq, 'Bonferroni'] = p_bon
			df.loc[df['Frequency'] == Freq, 'FDR'] = p_fdr

		# Save Results
		FileName = self.createFileName(suffix='GBC-Region-T-Test',filetype='.pkl', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', FileName)
		df.to_pickle(FilePath)
	
	def fdr_correction(self, pvals, alpha):
		"""
		False Detection rate correction for multiple comparison.
		:return array of rejected test
		""" 
		pvals = np.array(pvals)
		m = len(pvals)
		# Get ranks
		ranks = np.zeros(pvals.shape)
		ranks[pvals.argsort()] = np.arange(m)		
		# true/false list if null hypothesis is rejected 
		rejected = np.array([pval<=(rank+1)/m*alpha for pval, rank in zip(pvals,ranks)])
		if not np.any(rejected):
			return rejected       
		# largest rank for which null hypothesis is rejected
		largest_k = np.max(ranks[rejected])
		# set all ranks smaller than largest_k to True
		rejected = ranks<=largest_k
		return rejected

	def bonferroni_correction(self, pvals, alpha):
		"""
		Bonferroni correction for multiple comparison
		:return array of rejected test, array of corrected p-values
		""" 
		pvals = np.array(pvals) * len(pvals)
		pvals[pvals>=1] = 1.0
		rejected = pvals<alpha    
		return rejected, pvals
	
	def dabest_region_GBC(self):
		"""
		Compute regionwise t-test between global connectivity values
		"""
		self.GBC_df = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
		# Result Dictionary
		dabest_list = []  
		print('Started Statstical Test.') 
		for Freq in self.FrequencyBands.keys():
			# Compute DABEST in Parallel
			with Pool(25) as p: 
				freq_list = p.starmap(self._parallel_region_dabest, zip(self.RegionNames, [Freq]*len(self.RegionNames)))			
			
			freq_df = pd.concat(freq_list)
			freq_df['Frequency'] = Freq

			# Correct Bootstrapped p-values for multiple comparisons
			t_bonferroni_result, t_bonferroni_pval = self.bonferroni_correction(freq_df['pvalue_students_t'], alpha=0.05)
			t_fdr_result = self.fdr_correction(freq_df['pvalue_students_t'], alpha=0.05)
			freq_df['t_bonferroni_result'] = t_bonferroni_result
			freq_df['t_bonferroni_pval'] = t_bonferroni_pval
			freq_df['t_fdr_result'] = t_fdr_result

			welch_bonferroni_result, welch_bonferroni_pval = self.bonferroni_correction(freq_df['pvalue_welch'], alpha=0.05)
			welch_fdr_result = self.fdr_correction(freq_df['pvalue_welch'], alpha=0.05)
			freq_df['welch_bonferroni_result'] = welch_bonferroni_result
			freq_df['welch_bonferroni_pval'] = welch_bonferroni_pval
			freq_df['welch_fdr_result'] = welch_fdr_result
			
			mann_whit_bonferroni_result, mann_whit_bonferroni_pval = self.bonferroni_correction(freq_df['pvalue_mann_whitney'], alpha=0.05)
			mann_whit_fdr_result = self.fdr_correction(freq_df['pvalue_mann_whitney'], alpha=0.05)
			freq_df['mann_whit_bonferroni_result'] = mann_whit_bonferroni_result
			freq_df['mann_whit_bonferroni_pval'] = mann_whit_bonferroni_pval
			freq_df['mann_whit_fdr_result'] = mann_whit_fdr_result
			
			# Append to list of results
			dabest_list.append(freq_df)

		# Dabest Dataframe
		dabest_df = pd.concat(dabest_list)
		# Save DABEST Results
		FileName = self.createFileName(suffix='GBC-Region-DABEST',filetype='.pkl', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', 'Stats', FileName)
		dabest_df.to_pickle(FilePath)
	
	def _parallel_region_dabest(self, Region, Freq):
		"""
		Compute Regionwise differences.
		"""
		print(f'DABEST on region {Region}, Frequency: {Freq}')
		df_pivot = self.GBC_df.pivot(index=['Subject', 'Group'], columns='Frequency', values=Region).reset_index()
		# Bootstrap test with DABEST
		analysis = dabest.load(df_pivot, idx=("Control", "FEP"), x='Group', y=Freq, ci=90)
		results = analysis.mean_diff.results
		# Levene Test
		_, pval = scipy.stats.levene(df_pivot.loc[df_pivot['Group']=='Control', Freq], df_pivot.loc[df_pivot['Group']=='Control', Freq])
		results['levene-p-value'] = pval
		# Insert Region Name in result df
		results.insert(loc=0, column='Region', value=Region)
		return results

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
	
	def dabest_net_measures(self):
		"""
		Computes Statistics on Graph Measures
		""" 
		self.Net_df = pd.read_pickle(self.find(suffix='Graph-Measures-'+self.net_version, filetype='.pkl'))
		# Result Dictionary
		dabest_list = []  
		print('Started Graph Measure Stats.') 
		for Freq in self.FrequencyBands.keys():
			with Pool(10) as p: 
				freq_list = p.starmap(self._parallel_net_dabest, zip(self.GraphMeasures.keys(), [Freq]*len(self.GraphMeasures.keys())))			
			
			freq_df = pd.concat(freq_list)
			freq_df['Frequency'] = Freq
			dabest_list.append(freq_df)
			
			# Correct Bootstrapped p-values
			_, t_bon_corrected = bonferroni_correction(freq_df['pvalue_students_t'], alpha=0.05)
			_, t_fdr_corrected = fdr_correction(freq_df['pvalue_students_t'], alpha=0.05, method='indep')
			freq_df['t_bon_corrected'] = t_bon_corrected
			freq_df['t_fdr_corrected'] = t_fdr_corrected
			
			_, welch_bon_corrected = bonferroni_correction(freq_df['pvalue_welch'], alpha=0.05)
			_, welch_fdr_corrected = fdr_correction(freq_df['pvalue_welch'], alpha=0.05, method='indep')
			freq_df['welch_bon_corrected'] = welch_bon_corrected
			freq_df['welch_fdr_corrected'] = welch_fdr_corrected

			_, mann_whit_bon_corrected = bonferroni_correction(freq_df['pvalue_mann_whitney'], alpha=0.05)
			_, mann_whit_fdr_corrected = fdr_correction(freq_df['pvalue_mann_whitney'], alpha=0.05, method='indep')
			freq_df['mann_whit_bon_corrected'] = mann_whit_bon_corrected
			freq_df['mann_whit_fdr_corrected'] = mann_whit_fdr_corrected

		# Dabest Dataframe
		dabest_df = pd.concat(dabest_list)	
		# save DataFrame to File
		FileName = self.createFileName(suffix='Graph-Measures-DABEST-'+self.net_version, filetype='.pkl')
		FilePath = self.createFilePath(self.NetMeasuresDir, self.net_version, FileName)
		dabest_df.to_pickle(FilePath)
		print('Graph Measure Statistics done.')
		pass

	def _parallel_net_dabest(self, Measure, Freq):
		"""
		Apply Dabest on Graph Measure, is called in dabest_net_measures.
		"""
		print(f'DABEST on Graph Measure {Measure}, Frequency: {Freq}')
		df_pivot = self.Net_df.pivot(index=['Subject', 'Group'], columns='Frequency', values=Measure).reset_index()
		# Bootstrap test with DABEST
		analysis = dabest.load(df_pivot, idx=("Control", "FEP"), x='Group', y=Freq, ci=90)
		results = analysis.mean_diff.results
		# Levene Test
		_, pval = scipy.stats.levene(df_pivot.loc[df_pivot['Group']=='Control', Freq], df_pivot.loc[df_pivot['Group']=='FEP', Freq])
		results['levene-p-value'] = pval
		# Insert Region Name in result df
		results.insert(loc=0, column='Measure', value=Measure)
		return results
	



