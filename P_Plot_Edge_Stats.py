import numpy as np 
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
import ptitprince as pt
import pandas as pd
import Z_config as config
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from utils.FileManager import *
from time import time
import seaborn as sns

# set if DataFrame has to be created 
CreateDF = False
# Load File Manager, handle file dependencies
M = MEGManager()
Groups = {'Control':M.ControlIDs, 'FEP':M.FEPIDs}

def create_mean_edge_df():
    DataDict = {'Subject':[], 'Group':[]}
    DataDict.update({key:[] for key in config.FrequencyBands.keys()})
    # iterate over all freqs and groups
    for Group, Subjects in Groups.items():
        DataDict['Group'].extend([Group]*len(Subjects))
        DataDict['Subject'].extend(Subjects)
        for FreqBand in config.FrequencyBands.keys():
            print(f'Processing Freq {FreqBand}, Group {Group}.')
            Data = np.load(M.find(suffix='Mean-Edge.npy', Group=Group, Freq=FreqBand))
            DataDict[FreqBand].extend(Data.tolist())
    
    DataFrame = pd.DataFrame(DataDict)
    FileName = M.createFileName(suffix='Subject-Mean_Edge-Weights.pkl')
    FilePath = M.createFilePath(M.GroupStatsFC, 'Mean', FileName)
    DataFrame.to_pickle(FilePath)

# Create Instance of Plot-Manager
P = PlotManager()

def plot_edge_dist():
    """
    Plots the Edge Distribution of all edges 
    """
    FileName = P.createFileName(suffix='Total_Edge-Weights.pdf')
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        for FreqBand in config.FrequencyBands.keys():
            Data = np.load(P.find(suffix='stacked-FCs.npy', Freq=FreqBand))
            max_edge = np.round(np.max(Data),2)
            min_edge = np.round(np.min(Data),2)
            # Flatten np array, take only upper triangle of mat
            fData=[edge for Sub in Data for n,row in enumerate(Sub[:-1]) for edge in row[n+1:]]

            # Plot histogramm
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f'Edge Weights - {FreqBand} Band', fontsize=15)
            sns.histplot(fData, bins=50, ax=ax)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
            txtstr = '\n'.join([f'Max = {max_edge}', f'Min = {min_edge}'])
            ax.text(0.85,0.85, txtstr, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
                bbox = bbox_props, size='x-large')
            ax.set_xlim(-0.05,0.6)
            pdf.savefig() 

def plot_group_mean_edge(): 
    """
    Plots the Distribution of group mean edges
    """
    FileName = P.createFileName(suffix='Group-Mean_Edge-Weights.pdf')
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        for FreqBand in config.FrequencyBands.keys():
            # Set up figures
            sns.set_style("whitegrid")
            colors = ['tab:blue', 'tab:orange']
            f1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
            f1.suptitle(f'Group Mean Functional Connectivity - {FreqBand} Band', fontsize=15)
            f2, ax2 = plt.subplots(figsize=(7,5))
            ax2.set_title(f'Group Mean Edge Weights - {FreqBand} Band')
            cbar_ax = f1.add_axes([.92, .2, .03, .6])

            for idx, Group in enumerate(Groups.keys()):
                Data = np.load(P.find(suffix='Mean-FC.npy', Group=Group, Freq=FreqBand))
                im = sns.heatmap(Data, ax=ax1[idx], cbar=idx == 0, linewidth=.05, vmin=0, vmax=0.5, cmap="YlGnBu", square=True, cbar_ax=None if idx else cbar_ax)
                ticks = range(0,94,10)
                ax1[idx].set_xticks(ticks); ax1[idx].set_xticklabels(ticks)
                ax1[idx].set_yticks(ticks); ax1[idx].set_yticklabels(ticks)
                ax1[idx].set_title(Group, fontsize=15)
                cbar_ax.tick_params(labelsize=10)
                # Flatten np array, take only upper triangle of mat
                fData=[edge for n,i in enumerate(Data[:-1]) for edge in i[n+1:]]
                sns.histplot(fData, bins=50, ax=ax2, color=colors[idx], label=Group)
            
            ax2.legend()     
            pdf.savefig(f1)
            pdf.savefig(f2)

def plot_mean_edge():
    df = pd.read_pickle(P.find(suffix='Subject-Mean_Edge-Weighhts.pkl')) 
    FileName = P.createFileName(suffix='Subject-Mean_Edge-Weights.pdf')
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        for FreqBand in config.FrequencyBands.keys():
            sns.set_style("whitegrid")
            dx="Group"; dy=FreqBand; 
            # Settings
            ort="h"; pal = "Set2"; sigma = .2
            f, ax = plt.subplots(figsize=(7, 5))

            ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                            width_viol = .5, ax = ax, orient = ort)
            
            ax.set_title(f'Mean Edge Values in {FreqBand} Band')
            ax.set_xlabel('Mean Edge Value') 
            ax.set_xlim(0.08,0.38)           
            pdf.savefig()

if __name__ == "__main__":
    start  = time()
    if CreateDF:
        create_mean_edge_df()
    #plot_mean_edge()
    plot_group_mean_edge()
    plot_edge_dist()
    end = time()
    print('Time: ', end-start)