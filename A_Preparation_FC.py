from utils.FileManager import MEGManager
from time import time
import numpy as np
import network as net
from SignalAnalysis import Signal
import itertools

class preparation(MEGManager):
    """
    Class which handles the preparation of the FC matrices 
    """
    def calc_FC(self):
        """
        Function to perform preprocessing, generates FC matrix for each subject taking the configurations of the config file
        """
        # Iterate over all subjects
        for Subject in self.SubjectList:
            print(f'Processing Subject: {Subject}')
            Data, fsample = self.loadMatFile(Subject)        
            # Convert to Signal
            SubjectSignal = Signal(Data, fsample=fsample)        
            # Downsample Signal
            SubjectSignal.downsampleSignal(TargetFreq=self.DownFreq)

            # Filter data
            for FreqBand, Limits in self.FrequencyBands.items():
                print('Processing: ', FreqBand)
                # Check if
                if self.exists(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand):
                    print(f'Overwriting FC of {Subject} Freq {FreqBand}.')

                # Get Low-Pass orthogonalized Functional Connectivity Matrix of Frequency Band
                FC = SubjectSignal.getFC(Limits, processes=5)

                # Save
                FileName = self.createFileName(suffix='FC', filetype='.npy', Sub=Subject, Freq=FreqBand)
                FilePath = self.createFilePath(self.FcDir, Subject, FileName)
                np.save(FilePath, FC)
        print('Preprocessing done.')
    
    def z_trans_FC(self):
        """
        Function to z-transform fc matrices. Fc matrices must be calculated before calling this function
        """
        print('Z-Transforming FC matrices.')
        for Subject, FreqBand in itertools.product(self.SubjectList, self.FrequencyBands.keys()):                    
            # load FC matrix
            FC = np.load(self.find(suffix='FC',filetype='.npy', Sub=Subject, Freq=FreqBand))
            
            # Z transform 
            Mean = np.mean(FC)
            Std = np.std(FC)
            if np.isclose(Std,0): 
                Std = 1
            Zscores = (FC - Mean)/Std

            # Save Z scores
            FileName = self.createFileName(suffix='FC-z-scores',filetype='.npy', Sub=Subject, Freq=FreqBand)  
            FilePath = self.createFilePath(self.FcDir, Subject, FileName)
            np.save(FilePath, Zscores)
        print('Z-Transforming done.')
    
    def calc_mst(self):
        """
        Function to calculate minimum spanning tree of fc matrices. Fc matrices must be calculated before calling this function.
        """
        print('Minimum spanning Tree is beeing computed.')
        for Subject, FreqBand in itertools.product(self.SubjectList, self.FrequencyBands.keys()):                    
            # load FC matrix
            FC = np.load(self.find(suffix='FC',filetype='.npy', Sub=Subject, Freq=FreqBand))

            # Init network
            FC[FC<0] = 0
            np.fill_diagonal(FC, 0)
            Network = net.network(FC, np.arange(94))

            # Minimum Spanning Tree  
            mst = Network.MST()
            MstName = self.createFileName(suffix='MST',filetype='.npy', Sub=Subject, Freq=FreqBand)
            MstPath = self.createFilePath(self.MSTDir, Subject, MstName)
            np.save(MstPath, mst)
        print('Minimum spanning Tree computed.')
    
    def calc_bin(self): 
        """
        Function to calculate the binarized FC Matrices.
        :params in the range of 0-1 
        """
        print('Binarizing network.')
        for Subject, FreqBand in itertools.product(self.SubjectList, self.FrequencyBands.keys()):  
            for thresh in [0.1,0.2,0.3]:                  
                # load FC matrix
                FC = np.load(self.find(suffix='FC',filetype='.npy', Sub=Subject, Freq=FreqBand))

                # Init network
                FC[FC<0] = 0
                np.fill_diagonal(FC, 0)
                Network = net.network(FC, np.arange(94))

                # Binarizing
                binarized_FC = Network.binarize_net(thresh)
                FileName = self.createFileName(suffix=f'Binarized-FC-Thresh-{thresh}',filetype='.npy', Sub=Subject, Freq=FreqBand)
                FilePath = self.createFilePath(self.BinFcDir, f'Thresh-{thresh}', Subject, FileName)
                np.save(FilePath, binarized_FC)
        print('Binarizing done.')

if __name__ == "__main__":
    start = time()
    prep = preparation()
    #prep.calc_FC()
    #prep.calc_mst()
    prep.calc_bin()
    #prep.z_trans_FC()
    end = time()
    print('Time: ', end-start)