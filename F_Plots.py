from utils.FileManager import PlotManager
from utils.Plotter import Plotter
import Z_config as config
from utils.SignalAnalysis import Signal, Envelope

def createEnvPlot(SubjectNum = None, FreqBand = None):
    M = PlotManager()
    Signal, fsample = M.loadSignal(Subject=SubjectNum)
    Signal = Signal(Signal, fsample)
    Signal.downsampleSignal(TargetFreq=config.DownFreq)
    Index, ReferenceIndex = 1, 2
    OrthSignal, ReferenceSignal = Signal.getOrthEnvelope(Index, ReferenceIndex, FreqBand=FreqBand)
    Envelope = Signal.getEnvelope(Limits=config.FrequencyBands[FreqBand])
    P = Plotter(OrthSignal, ReferenceSignal, Signal[Index], Signal[ReferenceIndex], Envelope[Index])
    fig = P.plotEnvelope(Regions=[Index,ReferenceIndex])
    M.saveEnvelopePlot(fig, SubjectNum, FreqBand, suffix='Envelope-Plot')

def createFCPlot(SubjectNum=None, FreqBands=None):
    M = PlotManager()
    FCs = []
    for FreqBand in FreqBands:
        FCs.append(M.loadFC(SubjectNum,FreqBand))
    fig = Plotter(FCs).plotFC()
    M.saveFCPlot(fig, SubjectNum, FreqBands, suffix='Imshow-Plot')

def createMetaPlot():
        MetaDF = M.loadMetastability()
        fig = Plotter(MetaDF).plotMetastability()
        M.saveMetaPlot(fig, suffix='Metastability-Plot')
    elif mode in config.GraphMeasures:
        GraphMeasures = M.loadGraphMeasures()
        fig = Plotter(GraphMeasures).plotGraphMeasures()
        M.saveGraphMeasures(fig, suffix=mode)
    else:
        raise Exception('Mode not found.')

createPlot(['Envelope'])