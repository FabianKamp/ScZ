import numpy as np 
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
import ptitprince as pt
import pandas as pd
import Z_config as config
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

# Load File Manager, handle file dependencies
M = MEGManager()
Groups = ['Control', 'FEP']

def create_mean-edge_df():
    DataDict = {'Group':[]}
    DataDict.update({key:[] for key in config.FrequencyBands.keys()})
    # iterate over all freqs and groups
    for FreqBand, Group in itertools.product(Groups, config.FrequencyBands.keys())
        print(f'Processing Freq {FreqBand}, Group {Group}.')
        Data = np.load(M.find(suffix='Mean-Edge.npy', Group=Group, Freq=FreqBand))
        DataDict[FreqBand].extend(Data)
        DataDict[Group].extend([Group]*len(Data))
    DataFrame = pd.DataFrame(DataDict)
    FileName = M.createFileName(suffix='Mean-Edge.npy', Freq=FreqBand)
    FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', FileName)
    DataFrame.to_pickle(FilePath)
    return DataFrame

# Create Instance of Plot-Manager
P = PlotManager()

def plot_mean_edge(df): 
    P.createFileName(suffix='Mean-Edge.pdf')
    P.createFilePath(P.PlotEdge, FileName)
    with PdfPages(FilePath) as pdf:
        for FreqBand in config.FrequencyBands.keys():
        dx="Group"; dy=FreqBand; 
        # Settings
        ort="v"; pal = "Set2"; sigma = .2
        f, ax = plt.subplots(figsize=(7, 5))

        ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                        width_viol = .5, ax = ax, orient = ort)
        
        pdf.savefig()

if __name__ == "__main__":
    start  = time()
    DataFrame = create_mean-edge_df()
    plot_mean_edge(DataFrame)
    end = time()
    print('Time: ', end-start)