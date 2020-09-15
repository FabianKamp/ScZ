from utils.FileManager import PlotManager
import Z_config as config
import dabest
import matplotlib.pyplot as plt

def mean_diff(Measures, CarrierFrequency, ci = 95, plot=True):
    """
    This function calculates the mean difference of the graphmeasure between Con and FEP group.
    Uses the DABEST library.
    Loads the Graph measures of all subjects into a pd.dataframe. Saves the Dataframe if not existing.

    :param GraphMeasure: str, Graphmeasure that should be analysed.
    :param plot: boolean, specifies if plot should be generated
    :return: mean difference between groups
    """
    M = PlotManager()
    for Measure in Measures:
        df = M.loadGraphMeasures(suffix=Measure)

        FreqKey = str(CarrierFrequency)
        analysis = dabest.load(df, idx=("Control", "FEP"), x='Group', y=FreqKey, ci=ci)
        #Results = analysis.mean_diff.results
        if plot:
            Plot = analysis.mean_diff.plot()
            # Save Plot in Plot Directory
            M.saveMeanDiffPlot(Plot, suffix=Measure+'_Mean-Diff-Plot', CarrierFreq=CarrierFrequency)
            plt.close('all')
    #return Results

for FreqBand, Limits in config.FrequencyBands.items():
    mean_diff(['GlobEfficiency'], CarrierFrequency=FreqBand)
#'ClustCoeff', 'AvgCloseCentrality', 'GlobEfficiency'