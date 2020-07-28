%  This script runs the small-world propensity script by Bassett et. al

ParentDir = '/mnt/raid/data/SFB1315/SCZ/rsMEG/B_Analysis';
DataDir = fullfile(ParentDir, 'D_FunctCon');
temp = dir(DataDir);
SubjectList = {temp(3:end).name};

% Loop over FC Files
for Subject = SubjectList
    SubjectNum = Subject{1}
    Freq1 = [2:2:46];
    Freq2 = [64:2:90];
    CarrierFreqs = [Freq1, Freq2]
    SPList = [];
    for CarrierFreq = CarrierFreqs
        FileName = [SubjectNum, '_Carrier-Freq-', num2str(CarrierFreq), '_-FC-ShortPath.mat'];
        FilePath = fullfile(DataDir, SubjectNum, FileName);
        % load functional connectivity matrix
        FC = load(FilePath, 'FC');
        aFC = abs(FC.FC);
        % calculate small world propensity 
        SP = small_world_propensity(aFC);
        % append to list
        SPList = [SPList, SP];
    end
    
    ResultDir = fullfile(ParentDir, 'G_GraphMeasures', 'A_SubjectAnalysis', SubjectNum);
    if ~isfolder(ResultDir)
        mkdir(ResultDir)
    end
    ResultFile = [SubjectNum, '_smallworld.mat'];
    ResultPath = fullfile(ResultDir, ResultFile);
    save(ResultPath, 'SPList');    
end

% Loop over Low FC Files
for Subject = SubjectList
    SubjectNum = Subject{1}
    Freq1 = [2:2:46];
    Freq2 = [64:2:90];
    CarrierFreqs = [Freq1, Freq2]
    SPList = [];
    for CarrierFreq = CarrierFreqs
        FileName = [SubjectNum, '_Carrier-Freq-', num2str(CarrierFreq), '_low-FC-ShortPath.mat'];
        FilePath = fullfile(DataDir, SubjectNum, FileName);
        % load functional connectivity matrix
        LowFC = load(FilePath, 'LowFC');
        aFC = abs(LowFC.LowFC);
        % calculate small world propensity 
        SP = small_world_propensity(aFC);
        % append to list
        SPList = [SPList, SP];
    end
    
    ResultDir = fullfile(ParentDir, 'G_GraphMeasures', 'A_SubjectAnalysis', SubjectNum);
    if ~isfolder(ResultDir)
        mkdir(ResultDir)
    end
    ResultFile = [SubjectNum, '_low-smallworld.mat'];
    ResultPath = fullfile(ResultDir, ResultFile);
    save(ResultPath, 'SPList');  
end