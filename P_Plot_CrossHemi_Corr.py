import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import Z_config as config
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from utils.FileManager import *
from time import time
import seaborn as sns
import itertools
import network as net

def plot_cross_hemi_corr():
    P = PlotManager()
    # Creating Pd DataFrame
    DataDict = {'Group':[], 'Freq':[], 'Cross-Hemi. Correlation':[], 'Region':[], 'Code':[]}
    for Group, FreqBand in itertools.product(P.GroupIDs.keys(), config.FrequencyBands.keys()):
        FC = np.load(P.find(suffix='Mean-FC', filetype='.npy', Group=Group, Freq=FreqBand))
        NodeCodes = P.RegionCodes
        np.fill_diagonal(FC,0)
        network = net.network(FC, NodeCodes)
        Decode = {'Temporal':8, 'Occipital':5, 'Parietal':6}
        CodeList = [8111, 8101, 8201,  5101, 5201, 5301, 2001, 6001] # Temporal, Occipital, Central Regions defined in the AAL2
        for Code in CodeList:
            DataDict['Group'].append(Group); DataDict['Freq'].append(FreqBand)
            Left = Code
            Right = Code + 1
            DataDict['Cross-Hemi. Correlation'].append(network[Left, Right])
            DataDict['Code'].append(Code)
            # Append Lobe Name
            if np.floor(Code/100) in [60, 20]:
                DataDict['Region'].append('Central')
            else: 
                for key, value in Decode.items():
                    if np.floor(Code/1000) == value:           
                        DataDict['Region'].append(key)
    
    df = pd.DataFrame(DataDict)

    FileName = P.createFileName(suffix='Cross-Hemi',filetype='.pdf')
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        # Plot swarmplot
        g = sns.catplot(x="Freq", y="Cross-Hemi. Correlation", hue="Region", col="Group",
                        data=df, kind="swarm", height=4, aspect=1, palette='Set2')
        g.set_axis_labels("", "Cross-Hemi. Correlation")
        pdf.savefig()

if __name__ == "__main__":
    start = time()
    plot_cross_hemi_corr()
    end = time()
    print('Time: ', end-start)