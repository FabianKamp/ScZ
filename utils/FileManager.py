import mat73
import Z_config as config
import os, glob
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
        self.GroupIDs = {'Control': self.ControlIDs, 'FEP': self.FEPIDs}

    def createFileName(self, suffix, filetype, net_version=False, **kwargs):
        """
        Function to create FileName string. The file name and location is inferred from the suffix.
        Creates directories if not existing.
        :param suffix: name suffix of file
        :return: FilePath string
        """
        # if net_version is True than config.net_version is added automatically to suffix
        if net_version and config.net_version:
            suffix += '_' + config.net_version
        
        # config.mode contains orth-lowpass, orth, etc. Is automatically added to suffix.
        if config.conn_mode:
            suffix += '_' + config.conn_mode
                
        FileName = ''
        for key, item in kwargs.items():
            FileName += key + '-' + str(item) + '_'
        
        FileName += suffix + filetype
        return FileName

    def createFilePath(self, *args):
        Directory = ''
        for arg in args[:-1]:
            Directory = os.path.join(Directory, arg)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)

        FilePath = os.path.join(Directory, args[-1])
        return FilePath

    def exists(self, suffix, filetype, net_version=False, **kwargs):
        FileName = self.createFileName(suffix, filetype, net_version, **kwargs)
        if glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True):
            return True
        else:
            return False
    
    def find(self, suffix, filetype, net_version=False, **kwargs):
        FileName = self.createFileName(suffix, filetype, net_version, **kwargs)
        InnerPath = glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True)
        if len(InnerPath)>1:
            raise Exception('Multiple Files found.')
        if len(InnerPath)<1:
            raise Exception('No File found.')
        
        TotalPath = os.path.join(self.ParentDir, InnerPath[0])
        return TotalPath

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
        self.FcDir = os.path.join(self.ParentDir, 'FunctCon')
        self.GroupStatsFC = os.path.join(self.ParentDir, 'GroupStatsFunctCon')
        self.MSTDir = os.path.join(self.ParentDir, 'MinimalSpanningTree')
        self.BinFcDir = os.path.join(self.ParentDir, 'BinFunctCon')
        self.SplitFcDir = os.path.join(self.ParentDir, 'SplitFunctCon')
        self.MetaDir = os.path.join(self.ParentDir, 'Metastability')
        self.SubjectAnalysisDir = os.path.join(self.ParentDir, 'GraphMeasures', 'SubjectAnalysis')
        self.NetMeasuresDir = os.path.join(self.ParentDir, 'GraphMeasures')

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
        signal = DataFile['AAL94_norm']['trial'][0] # Signal has to be transposed
        return signal.T, fsample

    def loadGraphMeasures(self, suffix):
        FileName = super().createFileName(suffix=suffix)
        FilePath = os.path.join(self.NetMeasures, FileName + '.pkl')
        DataFrame = pd.read_pickle(FilePath)
        return DataFrame

    def loadMetastability(self, suffix='Metastability'):
        FileName = super().createFileName(suffix=suffix)
        FilePath = os.path.join(self.MetaDir, FileName + '.pkl')
        DataFrame = pd.read_pickle(FilePath)
        return DataFrame

    def saveGraphMeasures(self, DataDict, suffix):
        FileName = super().createFileName(suffix=suffix)
        FilePath = super().createFilePath(self.NetMeasures, FileName + '.pkl')
        if self.exists(FilePath):
            df = self._updateDataFrame(FilePath, DataDict)
        else:
            df = self.createDataFrame(DataDict)

        df.to_pickle(FilePath)

    def saveMetastability(self, DataDict, suffix='Metastability'):
        FileName = self.createFileName(suffix=suffix)
        FilePath = os.path.join(self.MetaDir, FileName + '.pkl')
        if self.exists(suffix):
            df = self._updateDataFrame(FilePath, DataDict)
        else:
            df = self.createDataFrame(DataDict)
        df.to_pickle(FilePath)

    def _createDataFrame(self, DataDict):
        """
        Creates DataFrame from DataDictionary using the index as orientation and
        adds Group
        :param DataDict: dictionary to convert into DataFrame
        :return: DataFrame
        """
        for SubjectNum in DataDict.keys():
            Group = self.getGroup(SubjectNum)
            DataDict[SubjectNum].update({'Group': Group})
        df = pd.DataFrame.from_dict(DataDict, orient='index')
        return df

    def _updateDataFrame(self, FilePath, DataDict):
        """
        Updates the DataFrame that is safed under the FilePath
        :param FilePath: FilePath of DataFrame
        :param DataDict: DataDictionary that is used to update DataFrame
        :return: Updated DataFrame
        """
        previous = pd.read_pickle(FilePath)
        previous = previous.to_dict('index')
        updated = previous.update(DataDict)
        df = self._createDataFrame(updated)
        return df

class PlotManager(MEGManager):
    def __init__(self):
        super().__init__()
        self.PlotDir = os.path.join(self.ParentDir, 'Plots')
        
    def saveEnvelopePlot(self, fig, SubjectNum, CarrierFreq, suffix):
        FileName = super().createFileName(suffix, Sub=SubjectNum, Freq=CarrierFreq)
        FilePath = super().createFilePath(self.PlotDir, 'Orthogonalized-Envelope', FileName + '.png')
        fig.savefig(FilePath)

    def saveFCPlot(self, fig, SubjectNum, CarrierFreq, suffix):
        FileName = super().createFileName(suffix, Sub=SubjectNum, Freq=CarrierFreq)
        FilePath = super().createFilePath(self.PlotDir, 'Functional-Connectivity', FileName + '.png')
        fig.savefig(FilePath)

    def saveMetaPlot(self, fig, suffix):
        FileName = super().createFileName(suffix)
        FilePath = super().createFilePath(self.PlotDir, 'Metastability', FileName + '.png')
        fig.savefig(FilePath)

    def saveMeanDiffPlot(self, fig, CarrierFreq, suffix):
        FileName = super().createFileName(suffix, Freq=CarrierFreq)
        FilePath = super().createFilePath(self.PlotDir, 'Graph-Measures', FileName + '.png')
        fig.savefig(FilePath)

    def saveAvgCCD(self, fig, CarrierFreq, suffix):
        FileName = super().createFileName(suffix, Freq=CarrierFreq)
        FilePath = super().createFilePath(self.PlotDir, 'Coherence-Connectivity-Dynamics', FileName + '.png')
        fig.savefig(FilePath)

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
            Signal, fsample = self.loadSignal(Subject)
            # Convert to Signal Class
            MEGSignal = Signal(Signal, fsample)
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
