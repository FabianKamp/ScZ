import NetworkFunctions.network as net
import numpy as np
from DataManagerNet import FileManager

# Load List of Data with FileManager
M = FileManager()
print(M.LowFCList)

for Subject in M.LowFCList:
    CarrierFrequencies = list(np.arange(2, 48, 2)) + list(np.arange(64, 92, 2))
    for CarrierFreq in CarrierFrequencies:
        LowFC = M.loadFC(Subject, CarrierFreq, Type='low')
        Network = net.network(LowFC)
        # Calculate Shortest Path
        ShortestPath = np.asarray(Network.shortestpath())
        MatDictionary = {'LowFC': LowFC, 'ShortestPath': ShortestPath}
        # Save to Matlab File
        M.saveMat(Subject, CarrierFreq, MatDictionary, Type='low')


for Subject in M.FCList:
    CarrierFrequencies = list(np.arange(2, 48, 2)) + list(np.arange(64, 92, 2))
    for CarrierFreq in CarrierFrequencies:
        FC = M.loadFC(Subject, CarrierFreq, Type='')
        Network = net.network(FC)
        # Calculate Shortest Path
        ShortestPath = np.asarray(Network.shortestpath())
        MatDictionary = {'FC': FC, 'ShortestPath': ShortestPath}
        # Save to Matlab File
        M.saveMat(Subject, CarrierFreq, MatDictionary, Type='')

