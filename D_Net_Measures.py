from utils.FileManager import MEGManager
import Z_config as config
import network as net
import numpy as np
import itertools
import pandas as pd
from time import time
from multiprocessing import Pool

def parallel_net_measures(idx, Group, Subject, FreqBand):
    print(f'Processing {Subject}, {FreqBand} Band')
    M = MEGManager() 
    # Init Result Dict
    ResultDict = {}
    ResultDict['Subject']=Subject
    ResultDict['Group']=Group
    ResultDict['Frequency']=FreqBand
    
    # Network Version
    version = config.net_version

    # Load FC matrix
    Data = np.load(M.find(suffix=version, filetype='.npy', Sub=Subject, Freq=FreqBand))

    # Remove negative edges from Network and set Diagonal to 0
    Data[Data<0] = 0
    np.fill_diagonal(Data, 0)
    network = net.network(Data, np.arange(Data.shape[-1]))

    # Calls network methods, appends result to Dict
    for Measure, FuncName in config.GraphMeasures.items():
        ResultDict[Measure]=getattr(network, FuncName)()
    
    df = pd.DataFrame(ResultDict, index=[idx])
    return df

def comp_net_measures():
    """
    Function to compute network measures for each subject, results are safed into pd.DataFrame
    :params type, default '', set to different FC version: 'mst', 'thres', etc. 
    """
    print('Started calculating graph measures')
    # Load List of Data with FileManager
    M = MEGManager()
    dfList = []
    for Group, SubjectList in M.GroupIDs.items():
        print(f'Processing Group {Group}.')
        with Pool(processes=10) as p:
            result = p.starmap(parallel_net_measures, [(idx, Group, Subject, FreqBand) for idx, (Subject, FreqBand) in 
            enumerate(itertools.product(SubjectList, config.FrequencyBands.keys()))])
        dfList.extend(result)
    DataFrame = pd.concat(dfList,ignore_index=True)    
    # save DataFrame to File
    FileName = M.createFileName(suffix='Graph-Measures', filetype='.pkl', net_version=True)
    FilePath = M.createFilePath(M.NetMeasuresDir, FileName)
    DataFrame.to_pickle(FilePath)
    print('Finished calculating graph measures')    

if __name__ == "__main__":
    start = time()
    comp_net_measures()
    end = time()
    print('Time: ', end-start)
    
