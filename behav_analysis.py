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
import dabest
from mne.stats import bonferroni_correction, fdr_correction

class analyzer(MEGManager):
	"""
	Class to perform behavioral analysis.
	"""
	def __init__(self):
		# Load visualization
		super().__init__()
		self.viz = visualization()

	def regr_gaf_avg_GBC(self):
		"""
		Calculate the correlation coefficient between avg. GBC and GAF score for each frequency. 
		Plots regression and saves pdf to GBC Plot directory.
		:return dict with
		"""
		# Load demographic df - contains GAF - and avg. GBC df
		df_demo = self.getDemographics()
		df_panss = self.getPANSS()
		df_demo = pd.merge(df_demo, df_panss, how='outer')
		df_GBC = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
		df = pd.concat([pd.merge(df_demo, df_GBC[df_GBC['Frequency']==Freq], on=['Subject','Group']) for Freq in self.FrequencyBands.keys()])

		FileName = self.createFileName(suffix='GBC-GAF-PANS',filetype='.pkl', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', FileName)
		df.to_pickle(FilePath)

		# Calculate Regression
		result_list = []		
		for Freq in self.FrequencyBands.keys(): 
			freq_df = df.loc[df['Frequency']==Freq]
			# compute scipy.lineregress output: slope, inter, corrval, pval, stderr
			result = scipy.stats.linregress(x=freq_df['Avg. GBC'], y=freq_df['GAF'])
			result_list.append([Freq] + list(result))
		
		# Save Regression Results
		regress_df = pd.DataFrame(result_list, columns=['Frequency', 'slope', 'intercept', 'corrval', 'pval', 'stderr'])
		FileName = self.createFileName(suffix='GAF-GBC-Regress', filetype='.pkl', Freq=self.Frequencies)
		FilePath = self.createFilePath(self.EdgeStatsDir, 'GBC', 'Stats', FileName)
		regress_df.to_pickle(FilePath)

if __name__ == "__main__":
	analyz = analyzer()
	analyz.regr_gaf_avg_GBC()