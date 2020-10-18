# Calculate Minimum Spanning Tree, Thresholded Network and Split Network
import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net
from time import time
import itertools

def process_fc():
    print('Processing started.')
    M = MEGManager()
    SubjectList = M.getSubjectList()
    
    for Subject, FreqBand in itertools.product(SubjectList, config.FrequencyBands.keys()):               
        if not M.exists(suffix='FC',filetype='.npy', Sub=Subject, Freq=FreqBand):
            print(f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing')
            continue            
        # load FC matrix
        FcName = M.createFileName(suffix='FC',filetype='.npy', Sub=Subject, Freq=FreqBand)
        FcPath = M.createFilePath(M.FcDir, Subject, FcName)
        FC = np.load(FcPath)

        # Z transform 
        Mean = np.mean(FC)
        Std = np.std(FC)
        if np.isclose(Std,0): 
            Std = 1
        Zscores = (FC - Mean)/Std

        # Save Z scores
        FileName = M.createFileName(suffix='FC_z-scores',filetype='.npy', Sub=Subject, Freq=FreqBand)  
        FilePath = M.createFilePath(M.FcDir, Subject, FileName)
        np.save(FilePath, Zscores)

        # Init network
        FC[FC<0] = 0
        np.fill_diagonal(FC, 0)
        Network = net.network(FC, np.arange(94))

        # Minimum Spanning Tree  
        mst = Network.MST()
        MstName = M.createFileName(suffix='FC_mst',filetype='.npy', Sub=Subject, Freq=FreqBand)
        MstPath = M.createFilePath(M.MSTDir, Subject, MstName)
        np.save(MstPath, mst)

        # TODO: Binarize Network    
    print('FC processing done.')

# run function
if __name__ == "__main__":
    start = time()
    process_fc()
    end = time()
    print('Time: ', end-start)