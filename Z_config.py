# File to configure the pipeline settings

# Specify Parent Directory and Data Directory
ParentDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/D_Analysis'
DataDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/A_rsMEG-Data'
DTIDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/B_DTI-Data'

InfoDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG/Z_Info'
InfoFileName = 'Info.xlsx'
InfoFile = InfoDir + r'/' + InfoFileName

# TODO: Specify if standard frequency bands are used and if lowpass filter is applied to signal envelope

Standard = True
mode = 'lowpass-FC' #'FC' or 'lowpass-FC'

if Standard:
    # Specify dictionary of the standard frequency bands
    FrequencyBands = {'Delta': [1,3], 'Theta': [4,7], 'Alpha':[8,12],
                          'Beta': [18, 22], 'Gamma': [38, 42]}
else:
    ## Specify Dictionary of Carrierfrequencies from 0 - 90 in steps of 2 Hz
    FrequencyList = list(range(2, 48, 2)) + list(range(64,92,2))
    FrequencyBands = {Freq: ([0.1, 4] if Freq <=2 else [Freq-2, Freq+2]) for Freq in FrequencyList}

# Specify Low-Pass filter Frequency for filtering the envelope
LowPassFreq = 0.2

# Specify downsampling frequency
DownFreq = 200

# Specifies the output Graph Measures
GraphMeasures = ['AvgDegree', 'AvgCharPath', 'AvgNeighDegree', 'Assortativity', 'Transitivity',
                 'AvgBetwCentrality', 'ClustCoeff', 'AvgCloseCentrality', 'GlobEfficiency']