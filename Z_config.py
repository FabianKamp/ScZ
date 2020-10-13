import os
# File to configure the pipeline settings
rsMEGDir = r'/mnt/raid/data/SFB1315/SCZ/rsMEG'

# Configure path to repos and scripts
repoDir = os.path.join(rsMEGDir, 'code')
NetDir = os.path.join(repoDir, 'Network')


# adds NetDir to sys.path, network module can be imported easily
import sys
sys.path.append(NetDir)

# Specify Parent Directory and Data Directory
ParentDir = os.path.join(rsMEGDir, 'Analysis')
DataDir = os.path.join(rsMEGDir, 'rsMEG-Data')
DTIDir = os.path.join(rsMEGDir, 'DTI-Data')

InfoDir = os.path.join(rsMEGDir, 'Info')
InfoFileName = 'Info.xlsx'
InfoFile = os.path.join(InfoDir, InfoFileName)

SubjectList = ''

# TODO: Specify if standard frequency bands are used and if lowpass filter is applied to signal envelope

Standard = True
mode = 'lowpass' #'no-lowpass'

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

# Specify thresholds for binarization
binthresholds = [0.1,0.15,0.20,0.25,0.30]

# Specifies the output Graph Measures
GraphMeasures = ['AvgDegree', 'AvgCharPath', 'AvgNeighDegree', 'Assortativity', 'Transitivity',
                 'AvgBetwCentrality', 'ClustCoeff', 'AvgCloseCentrality', 'GlobEfficiency']