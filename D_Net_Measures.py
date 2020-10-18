from utils.FileManager import MEGManager
import Z_config as config
import network as net
import numpy as np
import itertools
import pandas as pd
from time import time

def comp_net_measures(version=''):
    """
    Function to compute network measures for each subject, results are safed into pd.DataFrame
    :params type, default '', set to different FC version: 'mst', 'thres', etc. 
    """
    print('Started calculating graph measures')
    # Load List of Data with FileManager
    M = MEGManager()
    ResultDict = {}
    ResultDict.update({'Subject':[], 'Group':[], 'Frequency':[]})
    ResultDict.update({Measure:[] for Measure in config.GraphMeasures})

    for Group, SubjectList in M.GroupIDs.items():
        print(f'Processing Group {Group}.')
        for Sub, FreqBand in itertools.product(SubjectList, config.FrequencyBands.keys():
            ResultDict['Subject'].append(Sub); ResultDict['Frequency'].append(FreqBand)
            ResultDict['Group'].append(Group)
            # Load FC matrix
            if version:
                suffix = 'FC_' + version + '.npy'
            else:
                suffix = 'FC.npy'
            Data = np.load(M.find(suffix=suffix, Sub=Subject, Freq=FreqBand))
            network = net.network(Data, np.arange(Data.shape))
            
            # Calls network methods, appends result to Dict
            for Measure in config.GraphMeasures:
                ResultDict[Measure].append(getattr(network, Measure))
    DataFrame = pd.DataFrame(ResultDict)
    
    # save DataFrame to File
    if version:
        suffix = 'Graph-Measures_' + version + '.pkl'
    else:
        suffix = 'Graph-Measures.npy'
    FileName = M.createFileName(suffix=suffix)
    FilePath = M.createFilePath(M.NetMeasuresDir, 'Mean', FileName)
    DataFrame.to_pickle(FilePath)
    print('Finished calculating graph measures')    

if __name__ == "__main__":
    start = time()
    comp_net_measures()
    end = time()
    print('Time: ', end-start)
    










