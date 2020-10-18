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

def plot_net_measures(version=''): 
    """
    Function to plot net measures over different Frequency Values
    """
    P = PlotManager()   
    # Load Data 
    df = pd.read_pickle(P.find(suffix='Graph-Measures', filetype='.pkl', net_version=True))
    # Set Pdf-FilePath
    FileName = P.createFileName(suffix='Graph-Measures', filetype='.pdf')
    FilePath = P.createFilePath(P.PlotDir, 'Graph-Measures', FileName)

    # Plot net measures
    with PdfPages(FilePath) as pdf:
        for Measure in config.GraphMeasures.keys():
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(12,8))
            ax = sns.violinplot(x="Frequency", y=Measure, hue="Group", data=df, palette="Set2", split=True,
                                scale="count", inner="stick")
            ax.set_title(Measure)
            pdf.savefig()

if __name__ == "__main__":
    start = time()
    plot_net_measures()
    end = time()
    print('Time: ', end-start)
    