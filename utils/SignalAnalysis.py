import numpy as np
import Z_config as config
from mne.filter import filter_data, next_fast_len
from scipy.signal import hilbert
from multiprocessing import Pool
import timeit
from utils.ConnFunc import *

def orth_corr(ComplexSignal, SignalEnv, fsample, ConjdivEnv):
	"""
	Computes orthogonalized correlation of the envelope of the complex signal (nx1 dim array) and the signal envelope  (nxm dim array). 
	This function is called by signal.getOrthFC()
	:param ComplexSignal Complex 
	"""
	# Orthogonalize signal
	OrthSignal = (ComplexSignal * ConjdivEnv).imag
	OrthEnv = np.abs(OrthSignal)
	# Envelope Correlation
	if config.conn_mode=='orth-lowpass':
		# Low-Pass filter
		OrthEnv = filter_data(OrthEnv, fsample, 0, config.LowPassFreq, fir_window='hamming', verbose=False)
		SignalEnv = filter_data(SignalEnv, fsample, 0, config.LowPassFreq, fir_window='hamming', verbose=False)	
	corr_mat = pearson(OrthEnv, SignalEnv)	
	corr = np.diag(corr_mat)
	return corr

class Signal():
	"""This class handles the signal analysis - band pass filtering,
	amplitude extraction and low-pass filtering the amplitude 
	"""
	def __init__(self, mat, fsample=None):
		"""
		Initialize the signal object with a signal matrix
		:param mat: n (regions) x p (timepoints) numpy ndarray containing the signal
		"""
		assert isinstance(mat, np.ndarray), "Signal must be numpy array"
		self.Signal = mat
		self.fsample = fsample
		self.NumberRegions, self.TimePoints = mat.shape

	def __getitem__(self, index):
		return self.Signal[index]

	def getFrequencyBand(self, Limits):
		"""
		Band pass filters signal from each region
		:param Limits: int, specifies limits of the frequency band
		:return: filter Signal
		"""
		lowerfreq = Limits[0]
		upperfreq = Limits[1]
		filteredSignal = filter_data(self.Signal, self.fsample, lowerfreq, upperfreq,
									 fir_window='hamming', verbose=False)
		return filteredSignal

	def getEnvelope(self, Limits):
		filteredSignal = self.getFrequencyBand(Limits)
		n_fft = next_fast_len(self.TimePoints)
		complex_signal = hilbert(filteredSignal, N=n_fft, axis=-1)[..., :self.TimePoints]
		filteredEnvelope = np.abs(complex_signal)
		return filteredEnvelope

	def getResampleNum(self, TargetFreq):
		downsamplingFactor = TargetFreq/self.fsample
		return int(self.TimePoints * downsamplingFactor)

	def downsampleSignal(self, resample_num=None, TargetFreq=None):
		from scipy.signal import resample
		if TargetFreq is not None:
			resample_num = self.getResampleNum(TargetFreq)

		if self.Signal.shape[1] < resample_num:
			raise Exception('Target sample size must be smaller than original frequency')
		re_signal = np.empty((self.Signal.shape[0], resample_num))
		for row in range(self.Signal.shape[0]):
			re_signal[row, :] = resample(self.Signal[row, :], resample_num)
		
		# Save to signal
		self.Signal = re_signal

		# Reevaluate Sampling Frequency
		downsamplingFactor = resample_num / self.TimePoints
		self.fsample = self.fsample * downsamplingFactor
		self.NumberRegions, self.TimePoints = self.Signal.shape

	def getFC(self, Limits, pad=100, processes=1):
		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)

		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]
		ComplexSignal = ComplexSignal[:,pad:-pad]

		# Get signal envelope
		SignalEnv = np.abs(ComplexSignal)
		
		# If no conn_mode is specified, unorthogonalized FC is computed.
		if not config.conn_mode:
			FC = pearson(SignalEnv, SignalEnv)
			return FC

		SignalConj = ComplexSignal.conj()
		ConjdivEnv = SignalConj/SignalEnv 

		# Compute orthogonalization and correlation in parallel		
		with Pool(processes=processes) as p: 
			result = p.starmap(orth_corr, [(Complex, SignalEnv, self.fsample, ConjdivEnv) for Complex in ComplexSignal])
		FC = np.array(result)

		# Make the Corr Matrix symmetric
		FC = (FC.T + FC) / 2.
		return FC
	
	def getOrthEnvelope(self, Index, ReferenceIndex, FreqBand, pad=100, LowPass=False):
		"""
		Function to compute the Orthogonalized Envelope of the indexed signal with respect to a reference signal.
		Is used create a plot the orthogonalized Envelope.
		"""
		Limits = config.FrequencyBands[FreqBand]
		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)
		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]
		ComplexSignal = ComplexSignal[:,pad:-pad]

		# Get signal envelope and conjugate
		SignalEnv = np.abs(ComplexSignal)
		SignalConj = ComplexSignal.conj()

		OrthSignal = (ComplexSignal[Index] * (SignalConj[ReferenceIndex] / SignalEnv[ReferenceIndex])).imag
		OrthEnv = np.abs(OrthSignal)

		if LowPass:
			OrthEnv = filter_data(OrthEnv, self.fsample, 0, config.LowPassFreq, fir_window='hamming', verbose=False)
			ReferenceEnv = filter_data(SignalEnv[ReferenceIndex], self.fsample, 0, config.LowPassFreq, fir_window='hamming', verbose=False)
		else:
			ReferenceEnv = SignalEnv

		return OrthEnv, ReferenceEnv

	def getMetastability(self):
		CentrLowEnv = self.Signal - np.mean(self.Signal, axis=-1, keepdims=True) # substract mean from Low Envelope
		ComplexLowEnv = hilbert(CentrLowEnv, axis=-1) # converts low pass filtered envelope to complex signal
		LowEnvPhase = np.angle(ComplexLowEnv)

		ImPhase = LowEnvPhase * 1j  # Multiply with imaginary element
		SumPhase = np.sum(np.exp(ImPhase), axis=0)  # Sum over all areas
		self.Kuramoto = np.abs(SumPhase) / ImPhase.shape[0]  # Compute kuramoto parameter
		self.Metastability = np.std(self.Kuramoto, ddof=1)

		return self.Metastability

	def getCCD(self, Limits):
		""" Defines the Coherence Connectivity Dynamics following the description of Deco et. al 2017
		:param signal: numpy ndarray containing the signal envelope
		:return: numpy nd array containing the CCD matrix
		"""
		print('Calculating CCD.')
		# Get envelope and phase of envelope
		env = self.getEnvelope(self, Limits)
		analytic_signal = hilbert(env, axis=-1)[:,100:-100]
		phase = np.angle(analytic_signal)

		timepoints = env.shape[1]
		assert timepoints <= 15000, 'CCD matrix too big, downsample to lower frequency'

		# Compute phase difference
		nnodes = env.shape[0]
		phase_diff = [np.abs(phase[i, :] - phase[j, :] for i in range(nnodes - 1) for j in range(i + 1, nnodes)]
		phase_diff = np.stack(phase_diff)

		# Compute instantaneous coherence
		V = np.cos(phase_diff)

		# Compute Coherence Connectivity matrix
		CCD_mat = np.zeros((timepoints, timepoints))
		for t1 in range(timepoints):
			for t2 in range(t1, timepoints):
				prod = np.dot(V[:, t1], V[:, t2])  # Vector product
				magn = np.linalg.norm(V[:, t1]) * np.linalg.norm(V[:, t2])  # Vector magnitude
				CCD_mat[t1, t2] = prod / magn  # Normalize vector product and save in matrix
				CCD_mat[t2, t1] = CCD_mat[t1, t2]  # Make matrix symmetrical

		return CCD_mat.astype('float32')
