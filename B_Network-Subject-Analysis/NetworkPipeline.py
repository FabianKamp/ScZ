from DataManagerNet import FileManager
import numpy as np
import NetworkFunctions.network as net

def NetAnalysis(Subject, CarrierFreq, Type):
    """
    Function to load FC matrices and calculate network measures for each carrier frequency
    :param Subject: Subject Number
    :param CarrierFreq: carrier frequency
    :param Type: 'low' or ''
    :return: Dictionary with results
    """
    # Load FC
    FC = M.loadFC(Subject, CarrierFreq, Type=Type)
    # Set diagonal to zero
    np.fill_diagonal(FC, 0)
    # Take absolut value
    aFC = np.abs(FC)
    # Plug into network
    Network = net.network(aFC)
    # Get degrees
    Degree = Network.degree()
    # Get shortest path lengths
    ShortestPath = Network.shortestpath()
    # Get Characteristic path lengths
    CharPath = Network.char_path(node_by_node=True)
    # Get Avg Char path length
    AvgCharPath = Network.char_path(node_by_node=False)
    # Get Avg neighbour degree
    AvgNeighDegree = Network.avg_neigh_degree()
    # Get Assortativity
    Assortativity = Network.assortativity()
    # Register in Dictionary
    Results = {'Degree': Degree, 'ShortestPath': ShortestPath, 'CharPath': CharPath, 'AvgCharPath': AvgCharPath,
               'AvgNeighDegree': AvgNeighDegree, 'Assortativity': Assortativity}

    return Results

# Load List of Data with FileManager
M = FileManager()

# Iterate over subjects in each Subject List
for Subject in M.LowFCList:
    # Check if Result File already exists
    if M.Resultexists(Subject, Type='low'):
        print(f'Skipped Subject {Subject}.')
        continue
    # Initialize Dictionaries to save results
    LowResultDict = {}
    # Create array of carrier Frequencies
    CarrierFrequencies = list(np.arange(2, 48, 2)) + list(np.arange(64, 92, 2))
    for CarrierFreq in CarrierFrequencies:
        if not M.exists(Subject, CarrierFreq, Type='low'):
            print(f'Skipped Subject {Subject}. Carrier Freq {CarrierFreq} missing')
            continue
        # Save to Result Dictionary
        LowResultDict[CarrierFreq] = NetAnalysis(Subject, CarrierFreq, Type='low')
    # Save Results for each participant to File
    M.save(Subject, LowResultDict, Type='low')

# Iterate over subjects in each Subject List
for Subject in M.FCList:
    # Check if Result File already exists
    if M.Resultexists(Subject, Type=''):
        print(f'Skipped Subject {Subject}.')
        continue
    # Initialize Dictionaries to save results
    ResultDict = {}
    # Create array of carrier Frequencies
    CarrierFrequencies = list(np.arange(2, 48, 2)) + list(np.arange(64, 92, 2))
    for CarrierFreq in CarrierFrequencies:
        if not M.exists(Subject, CarrierFreq, Type=''):
            print(f'Skipped Subject {Subject}. Carrier Freq {CarrierFreq} missing')
            continue
        # Save to Result Dictionary
        ResultDict[CarrierFreq] = NetAnalysis(Subject, CarrierFreq, Type='')
    # Save Results for each participant to File
    M.save(Subject, ResultDict, Type='')







