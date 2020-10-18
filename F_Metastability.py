import Z_config as config
from utils.FileManager import MEGManager
from utils.SignalAnalysis import Signal, Envelope

# Load File Manager that handles file dependencies.
M = MEGManager()

if config.SubjectList:
    SubjectList = config.SubjectList
else:
    SubjectList = M.getSubjectList()
MetaDict = {}

for Subject in SubjectList:
    print(f'Processing Subject: {Subject}')
    # Loads Data of one Subject at a time
    SubjectData = M.loadSignal(Subject)
    # Convert to Signal type
    SubjectSignal = Signal(SubjectData['Signal'], fsample=SubjectData['SampleFreq'])
    # Get Resampling Size and Downsample Signal
    ResampleNum = SubjectSignal.getResampleNum(TargetFreq=config.DownFreq) #Calculate number of resampled datapoints
    SubjectSignal.downsampleSignal(ResampleNum)

    MetaDict[Subject] = {}

    # Filter data
    for FreqBand, Limits in config.FrequencyBands.items():
        print('Processing: ', FreqBand)

        # Compute Metastability and register into Dictionary
        SubjectEnvelope = SubjectSignal.getLowPassEnvelope(Limits)
        Metastability = Envelope(SubjectEnvelope).getMetastability()
        MetaDict[Subject][FreqBand] = Metastability

# Save Subject specific Metastability Dictionary
M.saveMetastability(MetaDict)

print('Preprocessing done.')