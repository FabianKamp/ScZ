from utils.FileManager import PlotManager
import Z_config as config
import dabest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mean_diff(Measures, CarrierFrequency, ci = 95, plot=True, MST=False):
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
        if MST:
            suffix = Measure+'_MST'
        else:
            suffix = Measure
        df = M.loadGraphMeasures(suffix=suffix)

        FreqKey = str(CarrierFrequency)
        analysis = dabest.load(df, idx=("Control", "FEP"), x='Group', y=FreqKey, ci=ci)
        #Results = analysis.mean_diff.results
        if plot:
            Plot = analysis.mean_diff.plot()
            # Save Plot in Plot Directory
            suffix = Measure + '_Mean-Diff-Plot'
            if MST:
                suffix = suffix + '_MST'
            M.saveMeanDiffPlot(Plot, suffix=suffix, CarrierFreq=CarrierFrequency)
            plt.close('all')
    #return Results

for FreqBand, Limits in config.FrequencyBands.items():
    mean_diff(['AvgDegree', 'AvgCharPath', 'AvgNeighDegree', 'Assortativity', 'Transitivity', 'ClustCoeff', 'AvgCloseCentrality', 'GlobEfficiency'], CarrierFrequency=FreqBand, MST=True)
#'ClustCoeff', 'AvgCloseCentrality', 'GlobEfficiency'