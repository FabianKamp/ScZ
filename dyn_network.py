from utils.FileManager import MEGManager
from SignalAnalysis import Signal
from time import time
import numpy as np
import itertools
from multiprocessing import Pool
import pandas as pd
import scipy.stats

# Load File Manager, handle file dependencies
class connectivity_dynamics(MEGManager):
	"""
	Class to calculate the connectivity dynamics of MEG Data.
	"""
	def calc_ccd(self):
		"""
		Calculates the average CCD for each Group and Frequency as defined in the configuration file. 
		Computes CCD in parallel calling _parallel_ccd
		"""
		print('Computing CCD.')
		# For now only use Alpha Band
		FreqBands = ['Alpha']
		FreqGroupSub = [(FreqBand, Group, Subject) for FreqBand in FreqBands for Group in self.GroupIDs.keys() for Subject in self.GroupIDs[Group]]		
		# Iterate in parallel over freqs, groups and subjects
		with Pool(processes=20) as p:
			p.starmap(self._parallel_ccd, FreqGroupSub)
		print('CCD done.')

	def _parallel_ccd(self, FreqBand, Group, Subject): 
		"""
		Computes average Group CCD for the input Frequency Band
		"""
		print(f'Processing Subject {Subject}, Group {Group}, Freq {FreqBand}')
		# Set Frequency Band Limits
		Limits = self.FrequencyBands[FreqBand]
		# Initiate CCD and set Downsampling Frequency
		DownFreq = 5 
		# Load raw Data
		Data, fsample = self.loadMatFile(Subject) 
		# Exclude areas
		exclude = list(range(40,46)) + list(range(74,82))
		Data = np.delete(Data, exclude, axis=0)
		# Take only first 3 Minutes
		Data = Data[:,:int(3.0*60*fsample)]
		# Convert to Signal
		SubjectSignal = Signal(Data, fsample=fsample)
		# Compute CCD              
		sCCD = SubjectSignal.getCCD(Limits, DownFreq=DownFreq) 
		# Save Subject CCD
		FileName = self.createFileName(suffix='CCD-Reduced', filetype='.npy', no_conn=True, Sub=Subject, Group=Group, Freq=FreqBand)
		FilePath = self.createFilePath(self.CCDDir, 'data', Group, FileName)
		np.save(FilePath, sCCD)
	
	def calc_ccd_hist(self):
		"""
		Calculates the average group histogram of the CCD matrices. 
		Excludes Subjects outside mean +/- 3*iqr with respect to the mean CCD value.
		"""
		print('Calculating histograms')
		# For now only calculate Alpha Band
		FreqBands = ['Alpha']
		# Detect outliers 
		outliers = []
		for FreqBand, Group in itertools.product(FreqBands, self.GroupIDs.keys()):
			CCD_mean = []
			SubjectList = self.GroupIDs[Group]
			for Subject in SubjectList:
				sCCD = np.load(self.find(suffix='CCD-Reduced', filetype='.npy', no_conn=True, Sub=Subject, Group=Group, Freq=FreqBand))
				CCD_mean.append(np.mean(sCCD))
			# outlier = np.abs(scipy.stats.zscore(CCD_mean))>3
			# Get outliers using IQR value
			outlier = (CCD_mean > np.mean(CCD_mean) + scipy.stats.iqr(CCD_mean)*3) | (CCD_mean < np.mean(CCD_mean) - scipy.stats.iqr(CCD_mean)*3)
			outliers.extend([sub for sub, out in zip(SubjectList, outlier) if out])
		
		# Delete duplicates in outliers
		outliers = set(outliers) 
		print('Excluding ', outliers)  

		# Define Edges and Midpoints of histogram bins
		self.bin_edges = np.arange(0,1,0.001)
		self.bins = self.bin_edges[:-1] + np.diff(self.bin_edges)/2
		gamma_dfs = []
		for FreqBand, Group in itertools.product(FreqBands, self.GroupIDs.keys()):
			print(f'Processing Group {Group}, FreqBand {FreqBand}')
			SubjectList = self.GroupIDs[Group]
			iterlist = [(FreqBand, Group, Subject) for Subject in set(SubjectList)-outliers]
			with Pool(20) as p: 
				results = p.starmap(self._parallel_ccd_hist, iterlist)				
			hists, gamma = list(zip(*results))
			# Calculate group mean histogramm
			group_hist = np.sum(np.stack(hists, axis=-1), axis=-1)/len(SubjectList)
			# repeat each bin by number in histogram, i.e. convert histogram to distribution
			group_hist = np.repeat(self.bins, group_hist.astype('int'))
			# Save Avg CCD Hist
			FileName = self.createFileName(suffix='Avg-CCD-Hist', filetype='.npy', no_conn=True, Group=Group, Freq=FreqBand)
			FilePath = self.createFilePath(self.CCDDir, 'hist', FileName)
			np.save(FilePath, group_hist.astype('float32'))
			# Save Gamma params to dataframe
			df = pd.DataFrame(gamma, columns=['Alpha', 'Location', 'Scale'])
			df['Subject']=set(SubjectList)-outliers; df['Group']=Group; df['Frequency']=FreqBand
			gamma_dfs.append(df)
		# Save Gamma params to dataframe 
		gamma_df = pd.concat(gamma_dfs)
		FileName = self.createFileName(suffix='CCD-Reduced-Hist-Gamma-Params', filetype='.pkl', Freq=self.Frequencies, no_conn=True)
		FilePath = self.createFilePath(self.CCDDir, 'hist', FileName)
		gamma_df.to_pickle(FilePath)
	
	def _parallel_ccd_hist(self, FreqBand, Group, Subject):
		# load subject CCD
		CCD = np.load(self.find(suffix='CCD-Reduced', filetype='.npy', no_conn=True, Sub=Subject, Group=Group, Freq=FreqBand))
		# Get histogramm 
		hist_idx = np.triu_indices(CCD.shape[0], k=1)
		hist, _ = np.histogram(CCD[hist_idx], bins=self.bin_edges)				
		# Fit gamma function to histogramm 
		# repeat each bin by number in histogram, i.e. convert histogram to distribution
		hist_distr = np.repeat(self.bins, hist.astype('int'))
		alpha, loc, scale = scipy.stats.gamma.fit(hist_distr)
		return hist, [alpha, loc, scale]
	
	def calc_meta(self):
		"""
		Calculates Kuramoto parameter and metastability in parallel 
		for all Subjects and FrequencyBands.
		"""
		print('Metastability started.')
		# Compute Metastabilit in parallel
		iter_list = [(Group, Subject, Frequency) for Group, SubjectList in self.GroupIDs.items() for Subject in SubjectList 
					for Frequency in self.FrequencyBands.keys()]
		with Pool(20) as p:
			results = p.starmap(self._parallel_meta, iter_list)
		
		# Transfer into panda Dataframe
		Group, Subject, Frequency, Metastability = zip(*results)
		df = pd.DataFrame({'Group':Group, 'Subject':Subject, 'Frequency':Frequency, 'Metastability':Metastability})
		
		# Save Metastability Dataframe
		FileName = self.createFileName(suffix='Metastability', filetype='.pkl', Freq=self.Frequencies, no_conn=True)
		FilePath = self.createFilePath(self.MetaDir, FileName)
		df.to_pickle(FilePath)
		print('Metastability done.')

	def _parallel_meta(self, Group, Subject, Frequency):
		"""
		Computes Metastability and Kuramoto for given Subject and Frequency Band. 
		There might be errors when directory is created in parallel.
		""" 
		print(f'Processing: {Subject}, {Group}, {Frequency}')
		Data, fsample = self.loadMatFile(Subject)
		# Downsample Signal to 200 Hz
		signal = Signal(Data, fsample)
		signal.resampleSignal(TargetFreq=self.DownFreq) 
		# Calculate Kuramoto and Metastability of Envelope in Frequency Band
		Limits = self.FrequencyBands[Frequency]        
		Kuramoto, Metastability = signal.getMetastability(Limits)
		# Save Kuramoto
		FileName = self.createFileName(suffix='Kuramoto', filetype='.npy', Sub=Subject, Freq=Frequency, no_conn=True)
		FilePath = self.createFilePath(self.MetaDir, 'Kuramoto', Subject, FileName)
		np.save(FilePath, Kuramoto)
		# Return Metastability
		return Group, Subject, Frequency, Metastability


if __name__ == "__main__":
	start = time()
	cdy = connectivity_dynamics()
	cdy.calc_ccd()
	cdy.calc_ccd_hist()
	#cdy.calc_meta()
	print('Time: ', time()-start)
	