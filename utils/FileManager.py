import mat73
import Z_config as config
import os, glob, sys
import numpy as np
import pandas as pd
from utils.SignalAnalysis import Signal
import matplotlib.pyplot as plt

class FileManager():
    """
    Class to manage all file dependencies of this project.
    """
    def __init__(self):
        print('Loading File Manager.')
        # Configure according to config File
        self.ParentDir = config.ParentDir
        self.DataDir = config.DataDir
        self.InfoFile = config.InfoFile
        self.NetDir = config.NetDir
        self.DemoFile = config.DemoFile
        self.BehavFile = config.BehavFile

        # Get Group IDs from Info sheet
        ControlIDs = self.getGroupIDs('CON')
        FEPIDs = self.getGroupIDs('FEP')
        self.GroupIDs = {'Control': ControlIDs, 'FEP': FEPIDs}

        # AAL name file 
        self.RegionNames, self.RegionCodes = self.getRegionNames()
        self.RegionCoordinates = self.getRegionCoords()

        # Load Attributes from config file 
        self.FrequencyBands = config.FrequencyBands
        self.DownFreq = config.DownFreq
        self.GraphMeasures = config.GraphMeasures
        self.SubjectList = config.SubjectList
        self.Frequencies = config.Frequencies
        self.net_version = config.net_version

    def createFileName(self, suffix, filetype, **kwargs):
        """
        Function to create FileName string. The file name and location is inferred from the suffix.
        Creates directories if not existing.
        :param suffix: name suffix of file
        :return: FilePath string
        """     
        # config.mode contains orth-lowpass, orth, etc. Is automatically added to suffix.
        if config.conn_mode and ('no_conn', True) not in list(kwargs.items()):
            suffix += '_' + config.conn_mode
                
        FileName = ''
        for key, val in kwargs.items():
            if key != 'no_conn':
                FileName += key + '-' + str(val) + '_'
        
        FileName += suffix + filetype
        return FileName

    def createFilePath(self, *args):
        """
        Creates full FilePath, if makes directories if they don't exist
        """
        Directory = ''
        for arg in args[:-1]:
            Directory = os.path.join(Directory, arg)
            if not os.path.isdir(Directory):
                os.mkdir(Directory)

        FilePath = os.path.join(Directory, args[-1])
        return FilePath

    def exists(self, suffix, filetype, **kwargs):
        FileName = self.createFileName(suffix, filetype, **kwargs)
        if glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True):
            return True
        else:
            return False
    
    def find(self, suffix, filetype, **kwargs):
        """
        Finds File within file structure.
        """
        FileName = self.createFileName(suffix, filetype, **kwargs)
        InnerPath = glob.glob(os.path.join(self.ParentDir, f'**/{FileName}'), recursive=True)
        if len(InnerPath)>1:
            raise Exception(f'Multiple Files found: {InnerPath}')
        if len(InnerPath)<1:
            raise Exception(f'No File found: {FileName}')
        
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

    def getGroup(self, SubjectNum):
        """
        Returns the Group that Subject ID belongs to
        :param SubjectNum:
        :return:
        """
        if SubjectNum in self.GroupIDs['FEP']:
            Group = 'FEP'
        elif SubjectNum in self.GroupIDs['Control']:
            Group = 'Control'
        else:
            Group = None
            print(f'{SubjectNum} not found in {config.InfoFileName}')
        return Group
    
    def getDemographics(self): 
        """
        Loads Demographic Data
        :return pd.DataFrame
        """
        with pd.ExcelFile(self.DemoFile) as xls:
            df_control = pd.read_excel(xls, 'CON')
            df_fep = pd.read_excel(xls, 'FEP')
        # Cut of rows and set columns - Control Group
        columns = list(df_control.iloc[2,1:])
        df_control = df_control.iloc[3:,1:]
        df_control.columns=columns
        df_control['AP'] = np.NaN 
        df_control['Group'] = 'Control'
        df_control.reset_index(drop=True)
        # Cut of rows and set columns - FEP Group
        columns = list(df_fep.iloc[2,1:7])
        df_fep_demo = df_fep.iloc[3:,1:7]
        df_fep_demo.columns=columns
        df_fep_demo['Group'] = 'FEP'
        df_fep_demo.reset_index(drop=True)

        # Concat dataframes
        df_demo = pd.concat([df_control, df_fep_demo])
        df_demo.rename(columns={'subjnr':'Subject'}, inplace=True)
        df_demo.drop(columns='subjcode', inplace=True)
        df_demo.rename(columns={'sex':'Gender', 'age':'Age'}, inplace=True)
        df_demo['GAF'] = df_demo['GAF'].astype('float32')
        return df_demo
    
    def getPANSS(self):
        """
        Loads PANSS scores of the schizophrenic patients
        :return pd.DataFrame
        """
        with pd.ExcelFile(self.DemoFile) as xls:
            df_fep = pd.read_excel(xls, 'FEP')
        columns = df_fep.iloc[2,8:]
        df_panss = df_fep.iloc[3:, 8:]
        df_panss.columns = columns
        df_panss.reset_index(drop=True)
        df_panss.rename(columns={'subjnr':'Subject'}, inplace=True)
        # change data type
        df_panss[['POS','NEG', 'COG', 'EXC', 'DEP', 'TOTAL']] = df_panss[['POS','NEG', 'COG', 'EXC', 'DEP', 'TOTAL']].astype('float32')
        return df_panss
    
    def getRegionNames(self):
        AAL2File = config.AAL2NamesFile
        with open(AAL2File, 'r') as file:
            f=file.readlines()
        assert len(f) == 94, 'AAL Name File must contain 94 lines.'        
        labels=[line[:-1].split()[1] for line in f]
        codes =[line[:-1].split()[2] for line in f]
        codes = list(map(int,codes))
        return labels, codes
    
    def getRegionCoords(self): 
        import json
        with open(config.AAL2CoordsFile, 'r') as file: 
            CoordDict = json.load(file)
        return CoordDict
    
    def _getRegionCoordsfromAAL(self):
        coords = plotting.find_parcellation_cut_coords('aal2.nii.gz')[:94]
        with open('aal2.nii.txt', 'r') as file:
            f=file.readlines()

        labels=[line[:-1] for line in f][:94]
        data = {label:coord for label, coord in zip(labels, coords.tolist())}

        with open(config.AAL2CoordsFile, mode='w') as outfile:
            json.dump(data, outfile)
     
class MEGManager(FileManager):
    def __init__(self):
        super().__init__()
        # Create the Directory Paths
        self.FcDir = os.path.join(self.ParentDir, 'FunctCon')
        self.EdgeStatsDir = os.path.join(self.ParentDir, 'EdgeStats')
        self.MSTDir = os.path.join(self.ParentDir, 'MinimalSpanningTree')
        self.BinFcDir = os.path.join(self.ParentDir, 'BinFunctCon')
        self.SplitFcDir = os.path.join(self.ParentDir, 'SplitFunctCon')
        self.MetaDir = os.path.join(self.ParentDir, 'Metastability')
        self.CCDDir = os.path.join(self.ParentDir, 'CCD')
        self.NetMeasuresDir = os.path.join(self.ParentDir, 'GraphMeasures')
        self.PlotDir = os.path.join(self.ParentDir, 'Plots')
        if len(self.SubjectList) == 0:
            self.SubjectList = self.getSubjectList()

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
    
    def loadMatFile(self, Subject):
        """
        Loads the MEG - Signal of specified Subject from Mat file
        :param Subject: Subject ID
        :return: Dictionary containing Signal and Sampling Frequency
        """
        SubjectFile = os.path.join(self.DataDir, Subject + '_AAL94_norm.mat')
        DataFile = mat73.loadmat(SubjectFile)
        fsample = int(DataFile['AAL94_norm']['fsample'])
        signal = DataFile['AAL94_norm']['trial'][0] # Signal has to be transposed
        return signal.T, fsample
