from utils.FileManager import MEGManager
import utils.NetworkFunctions.network as net
import Z_config as config
import numpy as np

def getGraphMeasure(FC, Measure):
    """
    Function to load FC matrices and calculate network measures for each carrier frequency
    :param Subject: Subject Number
    :param CarrierFreq: carrier frequency
    :param Type: 'low' or ''
    :return: Dictionary with results
    """
    # Set diagonal to zero
    np.fill_diagonal(FC, 0)
    # Take absolut value
    aFC = np.abs(FC)
    # Plug into network
    Network = net.network(aFC)

    if 'AvgDegree' == Measure:
        # Get degrees
        Degrees = Network.degree()
        Result = np.mean(Degrees)
    elif 'AvgCharPath' == Measure:
        # Get Avg Char path length
        Result = Network.char_path(node_by_node=False)
    elif 'AvgNeighDegree' == Measure:
        # Get Avg neighbour degree
        Result = Network.avg_neigh_degree()
    elif 'Assortativity' == Measure:
        # Get Assortativity
        Result = Network.assortativity()
    elif 'Transitivity' == Measure:
        # Get Transivity
        Result = Network.transitivity()
    elif 'AvgBetwCentrality' == Measure:
        # Get Avg Betweenness Centrality
        Result = Network.betweenness_centrality(avg=True)
    elif 'ClustCoeff' == Measure:
        # Get Cluster Coefficient
        Result = Network.clust_coeff(node_by_node=False)
    elif 'CloseCentrality' == Measure:
        Result = Network.closeness_centrality()
    elif 'AvgCloseCentrality' == Measure:
        Result = Network.closeness_centrality(avg=True)
    elif 'GlobEfficiency' == Measure:
        Result = Network.glob_efficiency()

    return Result

def computeNetMeasure(Measures):
    """
    :param Measures: Graph Measure as defined in configuration file
    :param mode: 'low-FC' or 'FC'
    """
    # Load List of Data with FileManager
    M = MEGManager()
    # Convert Measures into list
    Measures = list(Measures)
    # Choose correct Subjectlist
    SubjectList = M.getFCList()

    for Measure in Measures:
        ResultDict = {}
        for Subject in SubjectList:
            # Initiate subject specific subdirectory
            ResultDict[Subject] = {}

            for FreqBand, Limits in config.FrequencyBands.items():
                print(f'{FreqBand} Frequency Band, Subject {Subject}')
                if not M.exists(suffix=config.mode, SubjectNum=Subject, CarrierFreq=FreqBand):
                    print(f'Skipped Subject {Subject}. Frequency Band {FreqBand} missing')
                    continue
                # Load FC
                FC = M.loadFC(suffix=config.mode, SubjectNum=Subject, CarrierFreq=FreqBand)
                # Save to Result Dictionary
                ResultDict[Subject][FreqBand] = getGraphMeasure(FC, Measure)
        # Save Results to DataFrame
        M.saveDataFrame(ResultDict, suffix=Measure+'_'+config.mode)

# Execute Script
computeNetMeasure(config.GraphMeasures)









