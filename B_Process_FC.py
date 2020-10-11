# Calculate Minimum Spanning Tree, Thresholded Network and Split Network
import numpy as np
from utils.FileManager import MEGManager
import Z_config as config
import network as net

# Load File Manager, handle file dependencies
M = MEGManager()

# Get list of subjects if defined or load all subjects
if config.SubjectList:
    SubjectList = config.SubjectList
else:
    SubjectList = M.getSubjectList()

def process_fc(M, SubjectList):
    for Subject in SubjectList:
        print(f'Processing Subject {Subject}.')
        for FreqBand, Limits in config.FrequencyBands.items():        
            if not M.exists(suffix='FC.npy', Sub=Subject, Freq=FreqBand):
                print(f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing')
                continue
            
            # load FC matrix
            FcName = M.createFileName(suffix='FC.npy', Sub=Subject, Freq=FreqBand)
            FcPath = M.createFilePath(M.FcDir, Subject, FcName)
            FC = np.load(FcPath)

            # Init network
            Network = net.network(FC)

            # Minimum Spanning Tree  
            mst = Network.MST()
            MstName = M.createFileName(suffix='MST.npy', Sub=Subject, Freq=FreqBand)
            MstPath = M.createFilePath(M.MSTDir, Subject, MstName)
            np.save(MstPath, mst)

            # Split into positive and negative network
            pos, neg = Network.split()
            PosName = M.createFileName(suffix='FC_pos.npy', Sub=Subject, Freq=FreqBand)
            PosPath = M.createFilePath(M.SplitFcDir, 'Positive', Subject, PosName)
            np.save(PosPath, pos)

            NegName = M.createFileName(suffix='FC_neg.npy', Sub=Subject, Freq=FreqBand)
            NegPath = M.createFilePath(M.SplitFcDir, 'Negative', Subject, NegName)
            np.save(NegPath, neg)
            
            config.binthresholds.sort()
            # Binarize Network        
            for thres in config.binthresholds:
                BinFc = Network.binarize(thres)
                BinFcName = M.createFileName(suffix='FC_bin-thres-'+str(thres)+'.npy', Sub=Subject, Freq=FreqBand)
                BinFcPath = M.createFilePath(M.BinFcDir, 'Bin_Thres-' + str(thres), Subject, BinFcName)
                np.save(BinFcPath, BinFc)
    
    print('FC processing done.')

# run function
if __name__ == "__main__":
    process_fc(M, SubjectList)
