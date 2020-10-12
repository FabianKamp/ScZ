import numpy as np
import Z_config as config
from mne.filter import filter_data, next_fast_len
from scipy.signal import hilbert
from multiprocessing import Pool
import timeit
from utils.ConnFunc import *

def parallel_orth_corr(ComplexSignal, SignalEnv, fsample, ConjdivEnv):
	OrthSignal = (ComplexSignal * ConjdivEnv).imag
	OrthEnv = np.abs(OrthSignal)
	# Envelope Correlation
	if config.mode=='lowpass':
		# Low-Pass filter
		OrthEnv = filter_data(OrthEnv, fsample, 0, config.LowPassFreq, fir_window='hamming',
										verbose=False)
	corr = pearson3(OrthEnv, SignalEnv)	
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

	def getSignalConj(self, Limits):
		filteredSignal = self.getFrequencyBand(Limits)
		n_fft = next_fast_len(self.TimePoints)
		complex_signal = hilbert(filteredSignal, N=n_fft, axis=-1)[..., :self.TimePoints]
		SignalConj = complex_signal.conj()
		return SignalConj

	def getLowPassEnvelope(self, Limits, boundaryfrequency=config.LowPassFreq):
		filteredEnvelope = self.getEnvelope(Limits)
		LowPassEnv = filter_data(filteredEnvelope, self.fsample, 0, boundaryfrequency,
								 fir_window='hamming', verbose=False)
		return LowPassEnv

	def getResampleNum(self, TargetFreq):
		downsamplingFactor = TargetFreq/self.fsample
		return int(self.TimePoints * downsamplingFactor)

	def downsampleSignal(self, resample_num=None, TargetFreq=None):
		if TargetFreq is not None:
			resample_num = self.getResampleNum(TargetFreq)

		from scipy.signal import resample
		if self.Signal.shape[1] < resample_num:
			raise Exception('Target sample size must be smaller than original frequency')
		re_signal = np.empty((self.Signal.shape[0], resample_num))
		for row in range(self.Signal.shape[0]):
			re_signal[row, :] = resample(self.Signal[row, :], resample_num)

		self.Signal = re_signal

		# Reevaluate Sampling Frequency
		downsamplingFactor = resample_num / self.TimePoints
		self.fsample = self.fsample * downsamplingFactor
		self.NumberRegions, self.TimePoints = self.Signal.shape

	def getOrthFC(self, Limits, processes=1):
		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)

		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]

		# Get signal envelope and conjugate
		SignalEnv = np.abs(ComplexSignal)
		SignalConj = ComplexSignal.conj()

		# Low pass filter envelope
		if config.mode == 'lowpass':
			SignalEnv = filter_data(SignalEnv, self.fsample, 0, config.LowPassFreq, fir_window='hamming',
									  verbose=False)
		ConjdivEnv = SignalConj/SignalEnv 

		# Compute correlation in parallel		
		with Pool(processes=processes) as p: 
			result = p.starmap(parallel_orth_corr, [(Complex, SignalEnv, self.fsample, ConjdivEnv) for Complex in ComplexSignal])
		FC = np.array(result)

		# Make the Corr Matrix symmetric
		FC = (FC.T + FC) / 2.
		return FC
	
	def getOrthEnvelope(self, Index, ReferenceIndex, FreqBand, LowPass=True):
		"""
		Function to compute the Orthogonalized Envelope of the indexed signal with respect to a reference Signal.
		Uses same code as Orthogonalization part of the getOrthFC function but does not compute the correlation.
		Is used create a plot the orthogonalized Envelope.
		:param Index:
		:param ReferenceIndex:
		:param Limits:
		:param LowPass:
		:return:
		"""
		Limits = config.FrequencyBands[FreqBand]

		# Filter signal
		FilteredSignal = self.getFrequencyBand(Limits)

		# Get complex signal
		n_fft = next_fast_len(self.TimePoints)
		ComplexSignal = hilbert(FilteredSignal, N=n_fft, axis=-1)[:, :self.TimePoints]

		# Get signal envelope and conjugate
		SignalEnv = np.abs(ComplexSignal)
		SignalConj = ComplexSignal.conj()

		# Low pass filter envelope
		if LowPass:
			ReferenceEnv = filter_data(SignalEnv[ReferenceIndex], self.fsample, 0, config.LowPassFreq, fir_window='hamming',
								 verbose=False)
		else:
			ReferenceEnv = SignalEnv

		OrthSignal = (ComplexSignal[Index] * (SignalConj / SignalEnv)).imag
		OrthEnv = np.abs(OrthSignal)
		OrthEnv = OrthEnv[ReferenceIndex]

		if LowPass:
			OrthEnv = filter_data(OrthEnv, self.fsample, 0, config.LowPassFreq, fir_window='hamming',
								  verbose=False)

		return OrthEnv, ReferenceEnv




class Envelope(Signal):
	"""
	Class to compute the CCD and FC of the Envelope
	"""
	def __init__(self, Envelope):
		self.Signal = Envelope

	def getMetastability(self):
		CentrLowEnv = self.Signal - np.mean(self.Signal, axis=-1, keepdims=True) # substract mean from Low Envelope
		ComplexLowEnv = hilbert(CentrLowEnv, axis=-1) # converts low pass filtered envelope to complex signal
		LowEnvPhase = np.angle(ComplexLowEnv)

		ImPhase = LowEnvPhase * 1j  # Multiply with imaginary element
		SumPhase = np.sum(np.exp(ImPhase), axis=0)  # Sum over all areas
		self.Kuramoto = np.abs(SumPhase) / ImPhase.shape[0]  # Compute kuramoto parameter
		self.Metastability = np.std(self.Kuramoto, ddof=1)

		return self.Metastability

	def getCCD(self):
		""" Defines the Coherence Connectivity Dynamics following the description of Deco et. al 2017
		:param signal: numpy ndarray containing the signal envelope
		:return: numpy nd array containing the CCD matrix
		"""
		from scipy.signal import hilbert
		print('Calculating CCD.')

		analytic_signal = hilbert(self.Signal, axis=-1)
		phase = np.angle(analytic_signal)

		nnodes = phase.shape[0]
		timepoints = phase.shape[1]
		nsums = int((nnodes - 1) * (nnodes) / 2)  # Sum of all integers until nnodes-1
		abs_diff = np.zeros((nsums, timepoints))

		counter = 0
		for i in range(nnodes - 1):
			for j in range(i + 1, nnodes):
				abs_diff[counter, :] = np.abs(phase[i, :] - phase[j, :])
				counter += 1

		# Compute instantaneous coherence
		V = np.cos(abs_diff)
		del abs_diff

		# Compute Coherence Connectivity matrix
		CCD_mat = np.zeros((timepoints, timepoints))
		for t1 in range(timepoints):
			for t2 in range(t1, timepoints):
				prod = np.dot(V[:, t1], V[:, t2])  # Vector product
				magn = np.linalg.norm(V[:, t1]) * np.linalg.norm(V[:, t2])  # Vector magnitude
				CCD_mat[t1, t2] = prod / magn  # Normalize vector product and save in matrix
				CCD_mat[t2, t1] = CCD_mat[t1, t2]  # Make matrix symmetrical

		return CCD_mat.astype('float32')

	def getFC(self, mode='pearson'):
		if mode == 'pearson':
			M = self.Signal
			n = M.shape[0]
			P = np.array([np.corrcoef((M[i, :], M[j, :]))[0, 1] for i in range(n) for j in range(n)])
			return P.reshape(n, n)
		else:
			pass
