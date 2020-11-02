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

#Infor File
InfoDir = os.path.join(rsMEGDir, 'Info')
InfoFileName = 'Info.xlsx'
InfoFile = os.path.join(InfoDir, InfoFileName)

#AAL names file 
AAL2CoordsFileName = 'aal2_center_coords.txt'
AAL2CoordsFile = os.path.join(InfoDir, AAL2CoordsFileName)
AAL2NamesFileName = 'aal2.nii.txt'
AAL2NamesFile = os.path.join(InfoDir, AAL2NamesFileName)

# List of subjects that you want to process, if empty all subjects in info list are processeed.
SubjectList = [] # must be list

# TODO: Specify if standard, high/low gamma or all frequency bands are used 
Frequencies = 'Wide-Low-Gamma' # 'Low-Gamma', 'Standard', 'Complete'
# TODO: Specify connectivity mode ~ Measure for Edges (orth-lowpass correlation, orth correlation etc.)
conn_mode = 'orth-corr' # 'orth-lowpass-corr', 'orth-corr'
# TODO: Specify which Network version to use ~ MST, Binarized etc.
net_version = 'FC' # 'FC', 'MST'

# Specify dictionary of the standard frequency bands
if Frequencies=='Standard':
    FrequencyBands = {'Delta': [1,3], 'Theta': [4,7], 'Alpha':[8,12], 'Beta': [18, 22]}
if Frequencies=='Narrow-Low-Gamma': 
    FrequencyBands = {'Gamma-1':[30,34], 'Gamma-2':[34,38], 'Gamma-3':[38,42], 'Gamma-4':[42,46]}
    
## Specify Dictionary of Carrierfrequencies from 0 - 90 in steps of 2 Hz
elif Frequencies=='Wide-Low-Gamma':
    FrequencyBands = {'Low-Gamma-1':[30,38], 'Low-Gamma-2':[38,46]}
elif Frequencies=='Wide-High-Gamma':
    FrequencyBands = {'High-Gamma-1':[64, 72], 'High-Gamma-2': [72,80], 'High-Gamma-3':[80,88], 'High-Gamma-4':[88,96]}
elif Frequencies == 'Complete':
    FrequencyList = list(range(2, 48, 2)) + list(range(64,92,2))
    FrequencyBands = {Freq: ([0.1, 4] if Freq <=2 else [Freq-2, Freq+2]) for Freq in FrequencyList}

# Specify Low-Pass filter Frequency for filtering the envelope
LowPassFreq = 0.2

# Specify downsampling frequency
DownFreq = 200

# Specify thresholds for binarization
binthresholds = [0.1,0.15,0.20,0.25,0.30]

# Specifies the Name of each Graph Measures in the network module
GraphMeasures = {'Avg. Degree': 'degrees', 'Char. Pathlengths': 'char_path', 'Global Efficiency':'glob_efficiency',
                'Cluster Coefficient':'clust_coeff', 'Transitivity':'transitivity', 'Close. Centrality':'closeness_centrality',
                'Betw. Centrality':'betweenness_centrality', 'Avg. Neighbour Degree':'avg_neigh_degree', 
                'Assortativity':'manual_assortativity'}