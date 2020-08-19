import mat73
import Z_config as config
import os
import glob
import numpy as np
import pandas as pd

class FileManager():
    """Class to manage all file dependencies of this project.
    """
    def __init__(self):
        # Configure according to config File
        self.ParentDir = config.ParentDir
        self.DataDir = config.DataDir
        self.InfoFile = config.InfoFile
        # Create the Directory Paths
        self.FcDir = os.path.join(self.ParentDir, 'D_FunctCon')
        self.AmplEnvDir = os.path.join(self.ParentDir, 'E_AmplEnv')
        self.MetaDir = os.path.join(self.ParentDir, 'F_Metastability')
        self.SubjectAnalysisDir = os.path.join(self.ParentDir, 'G_GraphMeasures', 'A_SubjectAnalysis')
        self.NetMeasures = os.path.join(self.ParentDir, 'G_GraphMeasures', 'A_NetMeasures')
        self.PlotDir = os.path.join(self.ParentDir, 'G_GraphMeasures', 'B_Plots')
        # Get Group IDs from Info sheet
        self.ControlIDs = self.getGroupIDs('CON')
        self.FEPIDs = self.getGroupIDs('FEP')

    def _createFilePath(self, suffix, SubjectNum=None, CarrierFreq=None, mkdir=True):
        """
        Function to create Filepath string. The file name and location is inferred from the suffix.
        Creates directories if not existing.
        :param suffix: name suffix of file
        :param SubjectNum: subject ID
        :param CarrierFreq: Carrier Frequency of Signal if existing
        :param mkdir: boolean, creates new directories if true
        :return: FilePath string
        """

        # Create Directory Path
        if suffix == config.mode:
            ResultDir = os.path.join(self.FcDir, SubjectNum)
            FileName = SubjectNum + '_Carrier-Freq-' + str(CarrierFreq) + '_' + suffix + '.npy'
        elif 'Metastability' in suffix:
            ResultDir = self.MetaDir
            FileName = suffix + '.p'
        elif 'Plot' in suffix:
            Measure = suffix.split('_')[0]
            ResultDir = os.path.join(self.PlotDir, Measure)
            FileName = 'Carrier-Freq-' + str(CarrierFreq) + '_' + suffix + '.png'
        elif suffix.split('_')[-1] in config.GraphMeasures:
            ResultDir = self.NetMeasures
            FileName = suffix + '.p'
        else:
            raise Exception('Suffix not found.')

        # Create directory if it doesn't exist
        if mkdir:
            if not os.path.isdir(ResultDir):
                os.mkdir(ResultDir)

        # Join Directory Path and Filename
        FilePath = os.path.join(ResultDir, FileName)
        return FilePath

    def getSubjectList(self):
        """Gets the subject numbers of the subjects that have not been
        processed yet.
        """
        MEGFiles = glob.glob(os.path.join(self.DataDir, '*AAL94_norm.mat'))
        FileList = [Path.split('/')[-1] for Path in MEGFiles]
        SubjectList = [File.split('_')[0] for File in FileList]
        return SubjectList

    def getFCList(self):
        """Gets the subject numbers of the subjects for which FC matrix exists
        """
        FCFiles = glob.glob(os.path.join(self.FcDir, '*'))
        FCList = [Path.split('/')[-1] for Path in FCFiles]
        return FCList

    def loadData(self, Subject):
        SubjectFile = os.path.join(self.DataDir, Subject+'_AAL94_norm.mat')
        DataFile = mat73.loadmat(SubjectFile)
        fsample = int(DataFile['AAL94_norm']['fsample'])
        signal = DataFile['AAL94_norm']['trial'][0]
        SubjectData = {'SampleFreq': fsample, 'Signal': signal.T} #Transposes the signal
        return SubjectData

    def loadFC(self, suffix, SubjectNum, CarrierFreq):
        FilePath = self._createFilePath(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        FC = np.load(FilePath)
        return FC

    def loadGraphMeasures(self, suffix):
        FilePath = self._createFilePath(suffix, SubjectNum=None)
        DataFrame = pd.read_pickle(FilePath)
        return DataFrame

    def saveNumpy(self, Data, suffix, SubjectNum, CarrierFreq):
        FilePath = self._createFilePath(suffix, SubjectNum=SubjectNum, CarrierFreq=CarrierFreq)
        np.save(FilePath, Data)

    def saveDataFrame(self, DataDict, suffix):
        for SubjectNum in DataDict.keys():
            Group = self.getGroup(SubjectNum)
            DataDict[SubjectNum].update({'Group':Group})
        df = pd.DataFrame.from_dict(DataDict, orient='index')
        if config.Standard:
            suffix = 'Standard-Freq-Bands_' + suffix
        FilePath = self._createFilePath(suffix)
        # Save to pickle
        df.to_pickle(FilePath)

    def savePlot(self, Plot, suffix, CarrierFreq):
        FilePath = self._createFilePath(suffix, CarrierFreq=CarrierFreq)
        Plot.savefig(FilePath)

    def exists(self, suffix, SubjectNum, CarrierFreq=None):
        FilePath = self._createFilePath(suffix, SubjectNum, CarrierFreq, mkdir=False)
        if os.path.isfile(FilePath):
            print(f'{suffix} file exists. ',
                  '{SubjectNum}, Carrier Frequency {CarrierFreq}.')
            return True
        else:
            return False

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
            raise Exception('Subject not found in Info-file')

        return Group



