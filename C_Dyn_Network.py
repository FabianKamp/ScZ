from utils.FileManager import MEGManager
from utils.SignalAnalysis import Signal
from time import time
import numpy as np
import itertools
from multiprocessing import Pool
import pandas as pd

# Load File Manager, handle file dependencies
class connectivity_dynamics(MEGManager):
    """
    Class to calculate the connectivity dynamics of MEG Data.
    """
    def calc_CCD(self):
        """
        Calculates the average CCD for each Group and Frequency as defined in the configuration file. 
        Computes CCD in parallel calling _parallel_ccd
        """
        print('Computing CCD.')
        # Iterate over Groups and Frequencies
        # CCD is computed in parallel
        FreqGroupList = [(FreqBand, Group) for FreqBand in self.FrequencyBands.keys() for Group in self.GroupIDs.keys()]
        processes = min(len(FreqGroupList), 10)
        with Pool(processes) as p:
            p.starmap(self._parallel_ccd, FreqGroupList)
        print('CCD done.')

    def _parallel_ccd(self, FreqBand, Group): 
        """
        Computes average Group CCD for the input Frequency Band
        """
        print(f'Processing Group {Group}, Freq {FreqBand}')
        # Set Frequency Band Limits
        Limits = self.FrequencyBands[FreqBand]
        # Initiate CCD and set Downsampling Frequency
        DownFreq = 5 
        n_timepoints = 60*5*DownFreq
        CCD = np.zeros((n_timepoints, n_timepoints)).astype('float32')
        
        # Iterate over all subjects and add each CCD to CCD
        SubjectList = self.GroupIDs[Group]
        
        for Subject in SubjectList:
            print('Processing Subject: ', Subject)
            Data, fsample = self.loadMatFile(Subject) 
            # Take only first Minute
            Data = Data[:,:60*fsample]
            # Convert to Signal
            SubjectSignal = Signal(Data, fsample=fsample)
            # Compute CCD              
            sCCD = SubjectSignal.getCCD(Limits, DownFreq=DownFreq) 
            # Save Subject CCD
            FileName = self.createFileName(suffix='CCD', filetype='.npy', no_conn=True, Group=Group, Sub=Subject, Freq=FreqBand)
            FilePath = self.createFilePath(self.CCDDir, Group, FileName)
            np.save(FilePath, sCCD)
            # Mean group CCD
            max_idx = min(sCCD.shape[0], CCD.shape[0])
            sCCD = sCCD[:max_idx,:max_idx]; CCD = CCD[:max_idx, :max_idx]
            CCD += sCCD
            print('CCD shape: ', CCD.shape)

        # Computes average CCD
        CCD /= len(SubjectList)

        # Save
        FileName = self.createFileName(suffix='Avg-CCD', filetype='.npy', no_conn=True, Group=Group, Freq=FreqBand)
        FilePath = self.createFilePath(self.CCDDir, Group, FileName)
        np.save(FilePath, CCD)
    
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
    #cdy.calc_CCD()
    cdy.calc_meta()
    print('Time: ', time()-start)
    