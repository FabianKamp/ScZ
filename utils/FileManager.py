import mat73
import Z_config as config
import os
import glob
import numpy as np
import pandas as pd
from utils.SignalAnalysis import Signal, Envelope
import matplotlib.pyplot as plt

class FileManager():
    """Class to manage all file dependencies of this project.
    """
    def __init__(self):
        # Configure according to config File
        self.ParentDir = config.ParentDir
        self.DataDir = config.DataDir
        self.InfoFile = config.InfoFile
        # Get Group IDs from Info sheet
        self.ControlIDs = self.getGroupIDs('CON')
        self.FEPIDs = self.getGroupIDs('FEP')

    def _createFileName(self, suffix='', SubjectNum=None, CarrierFreq=None):
        """
        Function to create FileName string. The file name and location is inferred from the suffix.
        Creates directories if not existing.
        :param suffix: name suffix of file
        :param SubjectNum: subject ID
        :param CarrierFreq: Carrier Frequency of Signal if existing
        :param mkdir: boolean, creates new directories if true
        :return: FilePath string
        """
        # config.mode contains lowpass-FC or FC. Is added to suffix.
        if config.mode not in suffix:
            suffix = suffix + '_' + config.mode

        if SubjectNum is not None and CarrierFreq is not None:
            FileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_' + suffix
        elif SubjectNum is None and CarrierFreq is not None:
            FileName = 'Carrier-Freq-' + str(CarrierFreq) + '_' + suffix
        elif config.Standard:
            FileName = 'Standard-Freq-Bands_' + suffix
        else:
            FileName = suffix

        return FileName

    def exists(self, suffix, SubjectNum=None, CarrierFreq=None):
        FileName = self._createFileName(suffix, SubjectNum, CarrierFreq)
        if glob.glob(os.path.join(self.ParentDir, f'**/{FileName}.*'), recursive=True):
            print(f'{FileName} file exists. ',
                  '{SubjectNum}, Carrier Frequency {CarrierFreq}.')
            return True
        else:
            return False

    def getGroupIDs(self, Group):
        """
        Gets the IDs of the group. Handles the Info-File.
        This function only works with the Info-File supplied in the Info-Folder
        :param Group: str, Group that should be loaded
        :return: list of IDs
        """
        ExcelContent = pd.read_excel(self.InfoFile)
        pos = self._getLocation(ExcelContent, Group)
        IDs = ExcelContent.loc[pos[0] + 2:, pos[1]]
        IDs = list(IDs.dropna())
        return IDs

    def getGroup(self, SubjectNum):
        """
        Returns the Group that Subject ID belongs to
        :param SubjectNum:
        :return:
        """
        if SubjectNum in self.FEPIDs:
            Group = 'FEP'
        elif SubjectNum in self.ControlIDs:
            Group = 'Control'
        else:
            Group = None
            print(f'{SubjectNum} not found in {config.InfoFileName}')
        return Group

    def _getLocation(self, df, value, all=False):
        """
        Gets the first location (idx) of the value in the dataframe.
        :param df: panda.DataFrame
        :param value: str, searched string
        :param all: if true, returns all occurences of value
        :return: list of indices
        """
        poslist = []
        temp = df.isin([value]).any()
        columnlist = list(temp[temp].index)
        for column in columnlist:
            rows = df[column][df[column] == value].index
            for row in rows:
                poslist.append([row, column])
                if not all:
                    return poslist[0]
        return poslist

class MEGManager(FileManager):
    def __init__(self):
        super().__init__()
        # Create the Directory Paths
        self.FcDir = os.path.join(self.ParentDir, 'D_FunctCon')
        self.AmplEnvDir = os.path.join(self.ParentDir, 'E_AmplEnv')
        self.MetaDir = os.path.join(self.ParentDir, 'F_Metastability')
        self.SubjectAnalysisDir = os.path.join(self.ParentDir, 'G_GraphMeasures', 'A_SubjectAnalysis')
        self.NetMeasures = os.path.join(self.ParentDir, 'G_GraphMeasures', 'A_NetMeasures')

    def getSubjectList(self):
        """Gets the subject numbers of the MEG - Datafiles.
        """
        MEGFiles = glob.glob(os.path.join(self.DataDir, '*AAL94_norm.mat'))
        FileList = [Path.split('/')[-1] for Path in MEGFiles]
        SubjectList = [File.split('_')[0] for File in FileList]
        return SubjectList

    def getFCList(self):
        """Gets the subject numbers of the MEG-FC - Datafiles
        """
        FCFiles = glob.glob(os.path.join(self.FcDir, '*'))
        FCList = [Path.split('/')[-1] for Path in FCFiles]
        return FCList

    def loadSignal(self, Subject):
        """
        Loads the MEG - Signal of specified Subject
        :param Subject: Subject ID
        :return: Dictionary containing Signal and Sampling Frequency
        """
        SubjectFile = os.path.join(self.DataDir, Subject + '_AAL94_norm.mat')
        DataFile = mat73.loadmat(SubjectFile)
        fsample = int(DataFile['AAL94_norm']['fsample'])
        signal = DataFile['AAL94_norm']['trial'][0]
        SubjectData = {'SampleFreq': fsample, 'Signal': signal.T} #Transposes the signal
        return SubjectData

    def loadFC(self, SubjectNum, CarrierFreq, suffix=''):
        FileName = super()._createFileName(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        FilePath = os.path.join(self.FcDir, SubjectNum, FileName)
        FC = np.load(FilePath)
        return FC

    def loadGraphMeasures(self, suffix):
        FileName = super()._createFileName(suffix, SubjectNum=None)
        FilePath = os.path.join(self.NetMeasures, FileName)
        DataFrame = pd.read_pickle(FilePath)
        return DataFrame

    def saveFC(self, Data, SubjectNum, CarrierFreq, suffix=''):
        FileName = super()._createFileName(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        SubjectDir = os.path.join(self.FcDir, SubjectNum)
        if not os.path.isdir(SubjectDir):
            os.mkdir(SubjectDir)
        FilePath = os.path.join(SubjectDir, FileName + '.npy')
        np.save(FilePath, Data)

    def safeGraphMeasures(self, DataDict, suffix):
        df = self._createDataFrame(DataDict)
        FilePath = os.path.join(self.NetMeasures, self._createFileName(suffix) + '.pkl')
        df.to_pickle(FilePath)

    def safeMetastability(self, DataDict, suffix='Metastability'):
        df = self._createDataFrame(DataDict)
        FilePath = os.path.join(self.MetaDir, self._createFileName(suffix) + '.pkl')
        df.to_pickle(FilePath)

    def _createDataFrame(self, DataDict):
        for SubjectNum in DataDict.keys():
            Group = self.getGroup(SubjectNum)
            DataDict[SubjectNum].update({'Group':Group})
        df = pd.DataFrame.from_dict(DataDict, orient='index')
        return df

    def loadavgCCD(self, Group, FreqBand):
        if Group == 'Control':
            GroupIDs = self.ControlIDs
        elif Group == 'FEP':
            GroupIDs = self.FEPIDs
        else:
            raise Exception('Group not found')
        AvgCCD = None
        for n, Subject in enumerate(GroupIDs):
            # Load Subject Data
            MEGData = self.loadSignal(Subject)
            # Convert to Signal Class
            MEGSignal = Signal(MEGData['Signal'], fsample=MEGData['SampleFreq'])
            # Downsample Data
            ResNum = MEGSignal.getResampleNum(TargetFreq=config.DownFreq)
            MEGSignal.downsampleSignal(resample_num=ResNum)
            # Compute Envelope
            Limits = config.FrequencyBands[FreqBand]
            MEGEnvelope = MEGSignal.getLowPassEnvelope(Limits=Limits)
            # Compute CCD
            CCD = Envelope(MEGEnvelope).getCCD()
            # Compute Average CCD
            if AvgCCD is None:
                AvgCCD = CCD.copy()
            else:
                # Take smaller CCD shape if not equal
                if AvgCCD.shape != CCD.shape:
                    samples = min(AvgCCD.shape[1], CCD.shape[1])
                    AvgCCD = AvgCCD[:,:samples]
                    CCD = CCD[:,:samples]
                AvgCCD = (AvgCCD*n + CCD)/(n+1)

        return AvgCCD

class PlotManager(FileManager):
    def __init__(self):
        super().__init__()
        self.PlotDir = os.path.join(self.ParentDir, 'P_Plots')

    def safeEnvelopePlot(self, fig, suffix, SubjectNum, CarrierFreq):
        FileName = super()._createFileName(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        Directory = os.path.join(self.PlotDir, 'Orthogonalized-Envelope')
        FilePath = os.path.join(Directory, FileName + '.png')
        plt.savefig(fig, FilePath)

    def safeFCPlot(self, fig, suffix, SubjectNum, CarrierFreq):
        FileName = super()._createFileName(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        Directory = os.path.join(self.PlotDir, 'Functional-Connectivity')
        FilePath = os.path.join(Directory, FileName + '.png')
        plt.savefig(fig, FilePath)

    def safeMeanDiffPlot(self, fig, suffix):
        FileName = super()._createFileName(suffix)
        Directory = os.path.join(self.PlotDir, 'Graph-Measures')
        FilePath = os.path.join(Directory, FileName + '.png')
        plt.savefig(fig, FilePath)

    def safeAvgCCD(self, fig, suffix, CarrierFreq):
        FileName = super()._createFileName(suffix, CarrierFreq=CarrierFreq)
        Directory = os.path.join(self.PlotDir, 'Coherence-Connectivity-Dynamics')
        FilePath = os.path.join(Directory, FileName + '.png')
        plt.savefig(fig, FilePath)

class EvolutionManager(FileManager):
    """This class loads the DTI data from the SCZ-Dataset."""
    def __init__(self, Group=None):
        super().__init__()
        if Group==None:
            raise Exception('Please enter Group Name: HC, SCZ or SCZaff.')
        else:
            self.Group = Group
        self.DTIDir = config.DTIDir
        self.MEGDir = config.DataDir

        if not os.path.isdir(self.DTIDir) or not os.path.isdir(self.MEGDir):
            raise Exception('Data Directory does not exist.')

    def loadDTIDataset(self, normalize=True):
        """Loads subject data, normalizes and takes the average of connectivity and length matrices.
        Averaged Cmat and LengthMat is saved in self.Cmat and self.LengthMat."""
        self.SubData = {}
        self._loadDTIFiles()
        self.NumSubjects = len(self.SubData['Cmats'])

        if self.NumSubjects == 0:
            raise Exception('No connectivity matrices found.')

        # Normalize connectivitiy matrices
        if normalize:
            self._normalizeCmats()

        # Average connectivity and length matrices
        self.Cmat = self._getSCAverage(self.SubData['Cmats'])
        self.LengthMat = self._getSCAverage(self.SubData['LengthMats'])

    def _loadDTIFiles(self):
        """Function loads all of the subject data into self.SubData dictionary"""
        import scipy.io

        self.CMFiles = glob.glob(os.path.join(self.DTIDir + self.Group, '**/*CM.mat'), recursive=True)
        self.LENFiles = glob.glob(os.path.join(self.DTIDir + self.Group, '**/*LEN.mat'), recursive=True)

        self.SubData['Cmats'] = []
        self.SubData['LengthMats'] = []

        for cm, len in zip(self.CMFiles, self.LENFiles):
            CMFile = scipy.io.loadmat(cm)
            for key in CMFile:
                if isinstance(CMFile[key], np.ndarray):
                    self.SubData['Cmats'].append(CMFile[key])
                    break
            LENFile = scipy.io.loadmat(len)
            for key in LENFile:
                if isinstance(LENFile[key], np.ndarray):
                    self.SubData['LengthMats'].append(LENFile[key])
                    break

    def _normalizeCmats(self, method="max"):
        if method == "max":
            for c in range(self.NumSubjects):
                maximum = np.max(self.SubData['Cmats'][c])
                self.SubData['Cmats'][c]  = self.SubData['Cmats'][c] / maximum

    def _getSCAverage(self, Mats):
        mat = np.zeros(Mats[0].shape)
        for m in Mats:
            mat += m
        mat = mat/len(Mats)
        return mat






