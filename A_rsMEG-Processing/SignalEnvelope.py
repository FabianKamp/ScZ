from mne.filter import next_fast_len
from mne.filter import filter_data
from scipy.signal import hilbert
import numpy as np

class SignalEnvelope():
    def __init__(self, FilteredSignal, LowPassFreq):
        self.Signal = FilteredSignal.FilteredSignal
        self.N, self.T = self.Signal.shape
        self.SampleFreq = FilteredSignal.SampleFreq
        self.LowPassFreq = LowPassFreq

        # Get complex signal
        n_fft = next_fast_len(self.T)
        self.ComplexSignal = hilbert(self.Signal, N=n_fft, axis=-1)[:, :self.T]

        # Get signal envelope
        self.SignalEnv = np.abs(self.ComplexSignal)
        # Low-Pass Filter
        self.LowEnv = filter_data(self.SignalEnv, self.SampleFreq, 0, self.LowPassFreq, fir_window='hamming', verbose=False)
        # Get complex conjugate
        self.SignalConj = self.ComplexSignal.conj()

    def getLowEnvCorr(self):
        LowEnv = self.LowEnv - np.mean(self.LowEnv, axis=-1, keepdims=True)

        LowEnvStd = np.linalg.norm(LowEnv, axis=-1)  # Calculate the standard deviation using np.linalg.norm which squares, sums and takes the root
        LowEnvStd[np.isclose(LowEnvStd, 0)] = 1  # Eliminates zeros which produce error during pearson correlation

        LowCorrMat = np.empty((self.N, self.N))
        for n, ComplexSignal in enumerate(self.ComplexSignal):
            OrthSignal = (ComplexSignal * (self.SignalConj/self.SignalEnv)).imag
            OrthEnv = np.abs(OrthSignal)

            # Low-Pass Envelope Correlation
            # Low-Pass filter
            LowOrthEnv = filter_data(OrthEnv, self.SampleFreq, 0, self.LowPassFreq, fir_window='hamming',
                                         verbose=False)
            LowOrthEnv -= np.mean(LowOrthEnv, axis=-1, keepdims=True)  # Centralize the data
            LowOrthEnvStd = np.linalg.norm(LowOrthEnv, axis=-1)  # Compute standard deviation
            LowOrthEnvStd[np.isclose(LowOrthEnvStd, 0)] = 1  # Sets std close to zero to 1

            # Correlate each row of the
            LowCorrMat[n] = np.diag(np.matmul(LowOrthEnv, LowEnv.T))
            LowCorrMat[n] /= LowEnvStd[n]
            LowCorrMat[n] /= LowOrthEnvStd

        # Make the Corr Matrix symmetric
        LowCorrMat = (LowCorrMat.T + LowCorrMat) / 2.
        LowCorrMat *= (1 / 0.577)  # Correct for underestimation

        self.LowCorrMat = LowCorrMat
        return self.LowCorrMat

    def getEnvCorr(self):
        SignalEnv = self.SignalEnv - np.mean(self.SignalEnv, axis=-1, keepdims=True)

        EnvStd = np.linalg.norm(SignalEnv, axis=-1)  # Calculate the standard deviation using np.linalg.norm which squares, sums and takes the root
        EnvStd[np.isclose(EnvStd, 0)] = 1  # Eliminates zeros which produce error during pearson correlation

        CorrMat = np.empty((self.N, self.N))
        for n, ComplexSignal in enumerate(self.ComplexSignal):
            OrthSignal = (ComplexSignal * (self.SignalConj/self.SignalEnv)).imag
            OrthEnv = np.abs(OrthSignal)

            # Envelope Correlation
            OrthEnv -= np.mean(OrthEnv, axis=-1, keepdims=True)  # Centralize the data
            OrthEnvStd = np.linalg.norm(OrthEnv, axis=-1)  # Compute standard deviation
            OrthEnvStd[np.isclose(OrthEnvStd, 0)] = 1  # Sets std close to zero to 1

            # Correlate each row of the
            CorrMat[n] = np.diag(np.matmul(OrthEnv, SignalEnv.T))
            CorrMat[n] /= EnvStd[n]
            CorrMat[n] /= OrthEnvStd

        # Make the Corr Matrix symmetric
        CorrMat = (CorrMat.T + CorrMat) / 2.
        CorrMat *= (1 / 0.577)  # Correct for underestimation

        self.CorrMat = CorrMat
        return self.CorrMat

    def getMetastability(self):
        CentrLowEnv = self.LowEnv - np.mean(self.LowEnv, axis=-1, keepdims=True) # substract mean from Low Envelope
        ComplexLowEnv = hilbert(CentrLowEnv, axis=-1) # converts low pass filtered envelope to complex signal
        LowEnvPhase = np.angle(ComplexLowEnv)

        ImPhase = LowEnvPhase * 1j  # Multiply with imaginary element
        SumPhase = np.sum(np.exp(ImPhase), axis=0)  # Sum over all areas
        self.Kuramoto = np.abs(SumPhase) / ImPhase.shape[0]  # Compute kuramoto parameter
        self.Metastability = np.std(self.Kuramoto, ddof=1)

        return self.Metastability




