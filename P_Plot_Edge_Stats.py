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
import itertools

# Create Instance of Plot-Manager
P = PlotManager()

def plot_edge_dist():
    """
    Plots the Edge Distribution of all edges 
    """
    FileName = P.createFileName(suffix='Group_Edge-Weights',filetype='.pdf', Freq=config.Frequencies)
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)

    DataDict = {FreqBand:[] for FreqBand in config.FrequencyBands.keys()}
    DataDict.update({'Group':[]})
    min_val = 1
    max_val = 0
    
    # Load Data to DataFrame
    for Group in P.GroupIDs.keys():
        for FreqBand in config.FrequencyBands.keys():
            Data = np.load(P.find(suffix='stacked-FCs', filetype='.npy', Group=Group, Freq=FreqBand))
            mask = np.triu_indices(Data.shape[-1], k=1)
            Data = np.stack([SubData[mask] for SubData in Data])
            fData = np.ravel(Data).tolist()
            DataDict[FreqBand].extend(fData)
            min_val = min(min_val, np.min(fData))
            max_val = max(max_val, np.max(fData))
            n = len(fData)
        DataDict['Group'].extend([Group]*n)
    df = pd.DataFrame(DataDict)
    df = pd.melt(df, id_vars=['Group'], value_vars=list(config.FrequencyBands.keys()), var_name='Frequency', value_name='Edge Weights')
    
    
    with PdfPages(FilePath) as pdf:
            # Plot histogramm
            sns.set_style("whitegrid")
            # Plot the orbital period with horizontal boxes
            g = sns.displot(data=df, x="Edge Weights", hue="Group", col="Frequency", palette='Set2',
                kind='kde', height=3, aspect=.8)
            g.set_axis_labels("Edge Weight", "Count")
            g.set_titles("{col_name} Band")
            # Format ticks
            min_val = np.min(df['Edge Weights'])
            max_val = np.max(df['Edge Weights'])
            ticks = np.round(np.arange(min_val,max_val,0.2),1).tolist()
            g.set(yticks=[0, 0.5, 1], xticks=ticks)  
            # Save to pdf
            g.tight_layout()        
            pdf.savefig() 

def plot_group_mean_edge(): 
    """
    Plots the Distribution of group mean edges
    """
    FileName = P.createFileName(suffix='Group-Mean_Edge-Weights', filetype='.pdf', Freq=config.Frequencies)
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    DataDict = {FreqBand:{} for FreqBand in config.FrequencyBands.keys()}
    min_val = 1
    max_val = 0
    
    # Load Data to Data Dict
    for FreqBand, Group in itertools.product(config.FrequencyBands.keys(),P.GroupIDs.keys()):
        Data = np.load(P.find(suffix='Mean-FC', filetype='.npy', Group=Group, Freq=FreqBand))
        DataDict[FreqBand].update({Group:Data})
        min_val = min(min_val, np.min(Data))
        max_val = max(max_val, np.max(Data))

    with PdfPages(FilePath) as pdf:
        for FreqBand in config.FrequencyBands.keys():
            # Set up figures
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Group-Mean Functional Connectivity - {FreqBand} Band', fontsize=15)
            cbar_ax = fig.add_axes([.06, .2, .02, .6])
            fDataDict = {'Group':[], 'Edge Weight':[]}
            for idx, Group in enumerate(P.GroupIDs.keys()):
                Data = DataDict[FreqBand][Group]
                im = sns.heatmap(Data, ax=axes[idx], cbar=idx == 0, linewidth=.05, cmap="YlGnBu", vmin=min_val, vmax=max_val, cbar_ax=None if idx else cbar_ax)
                ticks = range(0,Data.shape[0],10)
                axes[idx].set_xticks(ticks); axes[idx].set_xticklabels(ticks)
                axes[idx].set_yticks(ticks); axes[idx].set_yticklabels(ticks)
                axes[idx].set_title(Group, fontsize=15)
                
                # Flatten np array, take only upper triangle of mat, save to dict which will be transformed to pd.DataFrame
                mask = np.triu_indices(Data.shape[-1], k=1)
                fData= Data[mask].tolist()
                fDataDict['Group'].extend([Group]*len(fData))
                fDataDict['Edge Weight'].extend(fData)

            DataFrame = pd.DataFrame(fDataDict)
            sns.histplot(DataFrame, x='Edge Weight', bins=50, ax=axes[2], hue="Group", palette='Set2')            
            # configure histplot
            axes[2].set_xlim(min_val,max_val)
            axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda value,nr: np.round(value,2)))
            axes[2].xaxis.set_major_locator(plt.MaxNLocator(5))
            # configure colorbar
            cbar_ax.tick_params(labelsize=10, left=True, labelleft=True, right=False, labelright=False)
            cbar_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value,nr: np.round(value,2)))
            cbar_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            pdf.savefig(fig)

def plot_subject_mean_edge():
    """
    Plot distribution of subject-mean edge values
    """
    # Load Data File
    df = pd.read_pickle(P.find(suffix='Subject-Mean_Edge-Weights', filetype='.pkl', Freq=config.Frequencies)) 
    FileName = P.createFileName(suffix='Subject-Mean_Edge-Weights', filetype='.pdf', Freq=config.Frequencies)
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        df = pd.melt(df, id_vars=['Subject','Group'], value_vars=list(config.FrequencyBands.keys()),
        var_name='Frequency', value_name='Mean Edge Weight') 
        sns.set_style("whitegrid")

        f, ax = plt.subplots(figsize=(12, 12))
        ax=pt.RainCloud(x = 'Frequency', y = 'Mean Edge Weight', hue = 'Group', data = df, palette = 'Paired', bw = .2,
                 width_viol = .7, ax = ax, orient = 'h' , alpha = .65, dodge = True, move=.2)
        
        ax.set_title(f'Subject-Mean Edge Weight')
        ax.set_xlim(np.min(df['Mean Edge Weight'])-0.005, np.max(df['Mean Edge Weight'])+0.005)           
        pdf.savefig(bbox_inches='tight')

def plot_GBC():
    df = pd.read_pickle(P.find(suffix='GBC',filetype='.pkl', Freq=config.Frequencies))
    FileName = P.createFileName(suffix='Mean-GBC', filetype='.pdf', Freq=config.Frequencies)
    FilePath = P.createFilePath(P.PlotDir, 'EdgeStats', FileName)
    with PdfPages(FilePath) as pdf:
        sns.set_style("whitegrid")
        fsize = len(list(config.FrequencyBands.keys()))
        f, ax = plt.subplots(figsize=(12, fsize*2))
        ax=pt.RainCloud(x = 'Frequency', y = 'Avg. GBC', hue = 'Group', data = df, palette = 'Paired', bw = .2,
                 width_viol = .7, ax = ax, orient = 'h' , alpha = .65, dodge = True, move=.2)
        
        ax.set_title(f'Average Global Brain Connectivity')
        #ax.set_xlim(np.min(df['Mean Edge Weight'])-0.005, np.max(df['Mean Edge Weight'])+0.005)           
        pdf.savefig(bbox_inches='tight')

def plot_NBS_comp(file):
    # Then get labels of the aal2 
    from mne.viz import circular_layout, plot_connectivity_circle
    labels = P.RegionNames
    labels = [label.replace('_', ' ') for label in labels]
    lh = [(zpos[idx], name) for idx, name in enumerate(labels) if name.endswith('L')]
    lh = sorted(lh)[::-1]
    rh  = [(zpos[idx], name) for idx, name in enumerate(labels) if name.endswith('R')]
    rh = sorted(rh)

    ordered = [item[1] for item in lh]
    ordered.extend([item[1] for item in rh])
    node_angles = circular_layout(labels, ordered, start_pos=90, group_boundaries=[0, len(labels)/2], group_sep=20)
    
    # Load Component
    adj = np.load(file)
    fig = plt.figure(figsize=(18,18))
    plot_connectivity_circle(adj, labels, node_angles=node_angles, n_lines=10, colormap='PuBu', colorbar_size = .4, 
                            colorbar_pos=(-0.5,.5), textcolor='k', facecolor='white', fig=fig)

if __name__ == "__main__":
    start  = time()
    #plot_subject_mean_edge()
    plot_GBC()
    #plot_group_mean_edge()
    #plot_edge_dist()
    end = time()
    print('Time: ', end-start)