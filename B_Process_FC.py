from utils.FileManager import MEGManager
import Z_config
import network as net

# Load File Manager, handle file dependencies
M = MEGManager()

# Get list of subjects if defined or load all subjects
if config.SubjectList:
    SubjectList = config.SubjectList
else:
    SubjectList = M.getSubjectList()

def process_fc(M, SubjectList)
    for Subject in SubjectList:
        print(f'Processing Subject {Subject}.')
        for FreqBand, Limits in config.FrequencyBands.items():        
            if not M.exists(suffix='FC.py', Sub=Subject, Freq=FreqBand):
                print(f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing')
                continue
            
            # load FC matrix
            FcName = M.createFileName(suffix='FC.npy', Sub=Subject, Freq=FreqBand)
            FcPath = M.createFilepath(M.FcDir, SubjectNum, FileName)
            Fc = np.load(FilePath)
            Network = net(FC)

            # Minimum Spanning Tree  
            mst = Network.MST()
            MstName = M.createFileName(suffix='MST.npy', Sub=Subject, Freq=FreqBand)
            MstPath = M.createFilepath(M.MstDir, SubjectNum, FileName)
            np.save(MstFilePath, mst)

            # Split into positive and negative network
            pos, neg = Network.split()
            PosName = M.createFileName(suffix='pos-FC.npy', Sub=Subject, Freq=FreqBand)
            PosPath = M.createFilepath(M.SplitFC, 'Positive', SubjectNum, FileName)
            np.save(PosPath, pos)

            NegName = M.createFileName(suffix='neg-FC.npy', Sub=Subject, Freq=FreqBand)
            NegPath = M.createFilepath(M.SplitFC, 'Negative', SubjectNum, FileName)
            np.save(NegPath, neg)

            # Binarize Network        
            for thres in config.binthresholds.sort():
                BinFc = Network.binarize(thres)
                BinFcName = M.createFileName(suffix='FC_thres'+str(thes)+'.npy', Sub=Subject, Freq=FreqBand)
                BinFcPath = M.createFilepath(M.BinFcDir, 'Thres-' + str(thres), SubjectNum, FileName)
                np.save(BinFcPath, BinFc)
    print('FC processing done.')

# run function
if __name__ == "__main__":
    process_fc(M, SubjectList)

