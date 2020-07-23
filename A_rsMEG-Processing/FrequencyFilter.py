from mne.filter import filter_data
import numpy as np

class BandPassFilter():
    def __init__(self, SubjectData, CarrierFrequency):
        """
        Filters the data into carrier frequencies +/- 1 Hz frequency bands.
        :param SubjectData: Dictionary that contains signal and sampling frequency of one subject
        :param CarrierFrequency: Carrier frequency
        """
        self.Signal = SubjectData['Signal']
        self.SampleFreq = SubjectData['SampleFreq']
        self.N = self.Signal.shape[0]
        self.T = self.Signal.shape[1]
        self.CarrierFreq = CarrierFrequency

        if self.CarrierFreq <= 2:
            print('Carrier Frequency is 2 or smaller. Frequencyintervall 0.1-4 for Band-Pass Filter.')
            self.FreqBand = [0.1, 4.]
        else:
            self.FreqBand = [self.CarrierFreq-2, self.CarrierFreq+2]

        self.FilteredSignal = self._applyBPFilter()

    def _applyBPFilter(self):
        FilteredSignal = filter_data(self.Signal, self.SampleFreq, self.FreqBand[0], self.FreqBand[1], fir_window='hamming',
                                     verbose=False)
        return FilteredSignal

