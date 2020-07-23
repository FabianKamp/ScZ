import mat73
import os
import glob
import numpy as np

class FileManager():
    """Class to load the meg files of all subject that have not yet been
    processed.
    """
    def __init__(self):
        self.ParentDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/B_Analysis'
        self.DataDir = os.path.join(self.ParentDir, 'C_rsMEG-Data')
        self.FcDir = os.path.join(self.ParentDir, 'D_FunctCon')
        self.AmplEnvDir = os.path.join(self.ParentDir, 'E_AmplEnv')
        self.MetaDir = os.path.join(self.ParentDir, 'F_Metastability')
        self._getSubjectList()

    def _getSubjectList(self):
        """Gets the subject numbers of the subjects that have not been
        processed yet.
        """
        MEGFiles = glob.glob(os.path.join(self.DataDir, '*AAL94_norm.mat'))
        FileList = [Path.split('/')[-1] for Path in MEGFiles]
        SubjectList = [File.split('_')[0] for File in FileList]
        #self.SubjectList = list(set(allSubjects) - set(fcSubjects + amplenvSubjects))
        self.SubjectList = SubjectList
        self.FileList = FileList

    def loadSubjectFile(self, Subject):
        SubjectFile = os.path.join(self.DataDir, Subject+'_AAL94_norm.mat')
        DataFile = mat73.loadmat(SubjectFile)
        fsample = int(DataFile['AAL94_norm']['fsample'])
        signal = DataFile['AAL94_norm']['trial'][0]
        timepoints = DataFile['AAL94_norm']['time'][0]
        SubjectData = {'SampleFreq': fsample, 'Signal': signal.T, 'Timepoints': timepoints} #Transposes the signal
        return SubjectData

    def save(self, Data, SubjectNum, Type, CarrierFreq=None):
        if Type == 'FC':
            if CarrierFreq is None:
                raise Exception('Carrier Frequency not defined.')
            ResultDir = os.path.join(self.FcDir, SubjectNum)
            if not os.path.isdir(ResultDir):
                os.mkdir(ResultDir)
            FileName = SubjectNum + '_' + 'Carrier-Freq-' + str(CarrierFreq) + '_env-FC.npy'
            np.save(os.path.join(ResultDir, FileName), Data)

        if Type == 'LowFC':
            if CarrierFreq is None:
                raise Exception('Carrier Frequency not defined.')
            ResultDir = os.path.join(self.FcDir, SubjectNum)
            if not os.path.isdir(ResultDir):
                os.mkdir(ResultDir)
            FileName = SubjectNum + '_' + 'Carrier-Freq-' + str(CarrierFreq) + '_low-env-FC.npy'
            np.save(os.path.join(ResultDir, FileName), Data)

        if Type == 'LowEnv':
            if CarrierFreq is None:
                raise Exception('Carrier Frequency not defined.')
            ResultDir = os.path.join(self.AmplEnvDir, SubjectNum)
            if not os.path.isdir(ResultDir):
                os.mkdir(ResultDir)
            FileName = SubjectNum + '_' + 'Carrier-Freq-' + str(CarrierFreq) + '_low-env.npy'
            np.save(os.path.join(ResultDir, FileName), Data.astype('float32'))

        if Type == 'Metastability':
            import pickle

            ResultDir = os.path.join(self.MetaDir)
            if not os.path.isdir(ResultDir):
                os.mkdir(ResultDir)
            FileName = SubjectNum + '_' + 'Metastability.p'
            FilePath = os.path.join(ResultDir, FileName)
            with open(FilePath, 'wb') as File:
                pickle.dump(Data, File, protocol=pickle.HIGHEST_PROTOCOL)

    def exists(self, SubjectNum, CarrierFreq):
        FcDir = os.path.join(self.FcDir, SubjectNum, 'FC')
        FcFileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_env-FC.npy'
        FcPath = os.path.join(FcDir, FcFileName)
        if os.path.isfile(FcPath):
            print('Functional Connectivity matrix already exists. Processing aborted for subject ',
                  f'{SubjectNum}, Carrier Frequency {CarrierFreq}.')
            return True

        LowFcDir = os.path.join(self.FcDir, SubjectNum, 'LowFC')
        LowFcFileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_low-env-FC.npy'
        LowFcPath = os.path.join(LowFcDir, LowFcFileName)
        if os.path.isfile(LowFcPath):
            print('Low Functional Connectivity matrix already exists. Processing aborted for subject ',
                  f'{SubjectNum}, Carrier Frequency {CarrierFreq}.')
            return True

        EnvDir = os.path.join(self.AmplEnvDir, SubjectNum)
        EnvFileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_low-env.npy'
        EnvPath = os.path.join(EnvDir, EnvFileName)
        if os.path.isfile(EnvPath):
            print('Low Envelope already exists. Processing aborted for subject ',
                  f'{SubjectNum}, Carrier Frequency {CarrierFreq}.')
            return True




