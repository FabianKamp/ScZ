import numpy as np 
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
import ptitprince as pt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from utils.FileManager import MEGManager
from time import time
import seaborn as sns
import itertools

class visualization(MEGManager):
    """
    Visualize result from FC analysis
    """
    def plot_edge_dist(self):
        """
        Plots the Edge Distribution of all edges 
        """
        FileName = self.createFileName(suffix='Group_Edge-Weights',filetype='.pdf', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.PlotDir, 'EdgeStats', FileName)

        DataDict = {FreqBand:[] for FreqBand in self.FrequencyBands.keys()}
        DataDict.update({'Group':[]})
        min_val = 1
        max_val = 0
        
        # Load Data to DataFrame
        for Group in self.GroupIDs.keys():
            for FreqBand in self.FrequencyBands.keys():
                Data = np.load(self.find(suffix='stacked-FCs', filetype='.npy', Group=Group, Freq=FreqBand))
                mask = np.triu_indices(Data.shape[-1], k=1)
                Data = np.stack([SubData[mask] for SubData in Data])
                fData = np.ravel(Data).tolist()
                DataDict[FreqBand].extend(fData)
                min_val = min(min_val, np.min(fData))
                max_val = max(max_val, np.max(fData))
                n = len(fData)
            DataDict['Group'].extend([Group]*n)
        df = pd.DataFrame(DataDict)
        df = pd.melt(df, id_vars=['Group'], value_vars=list(self.FrequencyBands.keys()), var_name='Frequency', value_name='Edge Weights')
        
        
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

    def plot_group_mean_edge(self): 
        """
        Plots the Distribution of group mean edges
        """
        FileName = self.createFileName(suffix='Group-Mean_Edge-Weights', filetype='.pdf', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.PlotDir, 'EdgeStats', FileName)
        DataDict = {FreqBand:{} for FreqBand in self.FrequencyBands.keys()}
        min_val = 1
        max_val = 0
        
        # Load Data to Data Dict
        for FreqBand, Group in itertools.product(self.FrequencyBands.keys(),P.GroupIDs.keys()):
            Data = np.load(self.find(suffix='Mean-FC', filetype='.npy', Group=Group, Freq=FreqBand))
            DataDict[FreqBand].update({Group:Data})
            min_val = min(min_val, np.min(Data))
            max_val = max(max_val, np.max(Data))

        with PdfPages(FilePath) as pdf:
            for FreqBand in self.FrequencyBands.keys():
                # Set up figures
                sns.set_style("whitegrid")
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'Group-Mean Functional Connectivity - {FreqBand} Band', fontsize=15)
                cbar_ax = fig.add_axes([.06, .2, .02, .6])
                fDataDict = {'Group':[], 'Edge Weight':[]}
                for idx, Group in enumerate(self.GroupIDs.keys()):
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

    def plot_subject_mean_edge(self):
        """
        Plot distribution of subject-mean edge values
        """
        # Load Data File
        df = pd.read_pickle(self.find(suffix='Subject-Mean_Edge-Weights', filetype='.pkl', Freq=self.Frequencies)) 
        FileName = self.createFileName(suffix='Subject-Mean_Edge-Weights', filetype='.pdf', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.PlotDir, 'EdgeStats', FileName)
        with PdfPages(FilePath) as pdf:
            df = pd.melt(df, id_vars=['Subject','Group'], value_vars=list(self.FrequencyBands.keys()),
            var_name='Frequency', value_name='Mean Edge Weight') 
            sns.set_style("whitegrid")

            f, ax = plt.subplots(figsize=(12, 12))
            ax=pt.RainCloud(x = 'Frequency', y = 'Mean Edge Weight', hue = 'Group', data = df, palette = 'Paired', bw = .2,
                    width_viol = .7, ax = ax, orient = 'h' , alpha = .65, dodge = True, move=.2)
            
            ax.set_title(f'Subject-Mean Edge Weight')
            ax.set_xlim(np.min(df['Mean Edge Weight'])-0.005, np.max(df['Mean Edge Weight'])+0.005)           
            pdf.savefig(bbox_inches='tight')

    def plot_avg_GBC(self):
        """
        Plot Average Global Connectivity
        """
        df = pd.read_pickle(self.find(suffix='GBC',filetype='.pkl', Freq=self.Frequencies))
        FileName = self.createFileName(suffix='Mean-GBC', filetype='.pdf', Freq=self.Frequencies)
        FilePath = self.createFilePath(self.PlotDir, 'EdgeStats', FileName)
        with PdfPages(FilePath) as pdf:
            sns.set_style("whitegrid")
            fsize = len(list(self.FrequencyBands.keys()))
            f, ax = plt.subplots(figsize=(12, fsize*2))
            ax=pt.RainCloud(x = 'Frequency', y = 'Avg. GBC', hue = 'Group', data = df, palette = 'Paired', bw = .2,
                    width_viol = .7, ax = ax, orient = 'h' , alpha = .65, dodge = True, move=.2)
            
            ax.set_title(f'Average Global Brain Connectivity')
            #ax.set_xlim(np.min(df['Mean Edge Weight'])-0.005, np.max(df['Mean Edge Weight'])+0.005)           
            pdf.savefig(bbox_inches='tight')

    def plot_NBS_comp(self, file):
        """
        Plot component of NBS analysis in circle
        """
        # Then get labels of the aal2 
        from mne.viz import circular_layout, plot_connectivity_circle
        labels = self.RegionNames
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
        
    def plot_cross_hemi_corr(self):
        # Creating Pd DataFrame
        DataDict = {'Group':[], 'Freq':[], 'Cross-Hemi. Correlation':[], 'Region':[], 'Code':[]}
        for Group, FreqBand in itertools.product(self.GroupIDs.keys(), self.FrequencyBands.keys()):
            FC = np.load(self.find(suffix='Mean-FC', filetype='.npy', Group=Group, Freq=FreqBand))
            NodeCodes = self.RegionCodes
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

        FileName = self.createFileName(suffix='Cross-Hemi',filetype='.pdf')
        FilePath = self.createFilePath(self.PlotDir, 'EdgeStats', FileName)
        with PdfPages(FilePath) as pdf:
            # Plot swarmplot
            g = sns.catplot(x="Freq", y="Cross-Hemi. Correlation", hue="Region", col="Group",
                            data=df, kind="swarm", height=4, aspect=1, palette='Set2')
            g.set_axis_labels("", "Cross-Hemi. Correlation")
            pdf.savefig()
        
    def plot_net_measures(self): 
        """
        Function to plot net measures over different Frequency Values
        """
            # Load Data 
            suffix = 'Graph-Measures-'+self.net_version
            df = pd.read_pickle(self.find(suffix=suffix, filetype='.pkl'))
            # Set Pdf-FilePath
            FileName = self.createFileName(suffix=suffix, filetype='.pdf')
            FilePath = self.createFilePath(self.PlotDir, 'Graph-Measures', FileName)

            # Plot net measures
            with PdfPages(FilePath) as pdf:
                for Measure in self.GraphMeasures.keys():
                    sns.set_theme(style="whitegrid")
                    fig, ax = plt.subplots(figsize=(12,8))
                    ax = sns.violinplot(x="Frequency", y=Measure, hue="Group", data=df, palette="Set2", split=True,
                                        scale="count", inner="stick")
                    ax.set_title(Measure)
                    pdf.savefig()

if __name__ == "__main__":
    start  = time()
    viz=visualization()
    viz.plot_cross_hemi_corr()
    viz.plot_edge_dist()
    viz.plot_net_measures()
    viz.plot_subject_mean_edge()
    viz.plot_avg_GBC()
    end = time()
    print('Time: ', end-start)