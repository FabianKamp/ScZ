import os
import glob
import numpy as np
import pickle

class FileManager():
    def __init__(self):
        self.ParentDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/B_Analysis'
        self.DataDir = os.path.join(self.ParentDir, 'D_FunctCon')
        self.ResultDir = os.path.join(self.ParentDir, 'G_NetworkMeasures', 'A_Subject-Analysis')
        self._getFCList()
        self._getLowFCList()

    def _getFCList(self):
        """Gets the subject numbers of the subjects for which FC matrix exists
        """
        FCFiles = glob.glob(os.path.join(self.DataDir, '/*/*_env-FC.npy'))
        FCList = [Path.split('/')[-1] for Path in FCFiles]
        FCList = [File.split('_')[0] for File in FCList]
        self.FCList = FCList

    def _getLowFCList(self):
        """Gets the subject numbers of the subjects for which FC matrix exists
        """
        LowFCFiles = glob.glob(os.path.join(self.DataDir, '/*/*_low-env-FC.npy'))
        LowFCList = [Path.split('/')[-1] for Path in LowFCFiles]
        LowFCList = [File.split('_')[0] for File in LowFCList]
        self.LowFCList = LowFCList
    
    def loadFC(self, SubjectNum, CarrierFreq, Type=''):
        FileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_' + Type +'env-FC.npy'
        FilePath = os.path.join(self.DataDir, SubjectNum, FileName)
        FC = np.load(FilePath)
        return FC
    
    def exists(self, SubjectNum, CarrierFreq, Type=''):
        FileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_' + Type + '-env-FC.npy'
        FilePath = os.path.join(self.DataDir, SubjectNum, FileName)
        if os.path.isfile(FilePath):
            return True
        else:
            print(f'FC or LowFC of {SubjectNum} and Carrier Frequency {CarrierFreq} missing.')
            return False

    def Resultexists(self, SubjectNum, Type=''):
        FileName = SubjectNum + '_' + Type + '-Network-Measures.p'
        FilePath = os.path.join(self.ResultDir, FileName)
        if os.path.isfile(FilePath):
            print(f'Subject {SubjectNum} result file already exists.')
            return True
        else:
            return False


    def save(self, SubjectNum, Data, Type=''):
        FileName = SubjectNum + '_' + Type + '-Network-Measures.p'
        FilePath = os.path.join(self.ResultDir, FileName)
        with open(FilePath, 'wb') as File:
            pickle.dump(Data, File, protocol=pickle.HIGHEST_PROTOCOL)

