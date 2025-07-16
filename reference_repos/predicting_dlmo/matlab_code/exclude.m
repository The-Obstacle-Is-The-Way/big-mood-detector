function data = exclude(dlmos,subject,dlmoTypeToUse,inputType)

wantExclude48Hours = 0;
wantEverybodyExceptOSAAndSubject21 = 1;

excludedIndices = [];
dt = (subject(1).epochtime(2) - subject(1).epochtime(1));

fractionalDLMO_Min = [];
fractionalDLMO_Max = [];
fractionalDLMO_CurveFit = [];
dlmoAsDatenum = [];

for i = 1:length(subject)
    if wantExclude48Hours && (subject(i).epochtime(end) < dlmos(i).maxDLMO_datenum - dt*60*48)
        % Excludes subjects with data ending more than 48 hours before DLMO collection time
        excludedIndices = [excludedIndices; i];
    end
    
    if wantEverybodyExceptOSAAndSubject21
       excludedIndices = [12 43];
    end
    
    fractionalDLMO_Max(i) = dlmos(i).maxDLMO_fraction;
    fractionalDLMO_Min(i) = dlmos(i).minDLMO_fraction;
    
    correctionTerm = 0.15;
    fractionalDLMO_CurveFit(i) = dlmos(i).curveFitDLMO_fraction + correctionTerm;
    
end

nonexcludedIndices = setdiff([1:length(subject)]', excludedIndices);
cleanedSubjectPool = subject(nonexcludedIndices);

if strcmp(dlmoTypeToUse,'max')
    fractionalDLMOToUse = fractionalDLMO_Max;
end

if strcmp(dlmoTypeToUse,'min')
    fractionalDLMOToUse = fractionalDLMO_Min;
end

if strcmp(dlmoTypeToUse,'curveFit')
    fractionalDLMOToUse = fractionalDLMO_CurveFit;
end

cleanedFractionalDLMOToUse = fractionalDLMOToUse(nonexcludedIndices);
startPointOffsetInHours = [];

for i = 1:length(subject)
    
    [startIndex,~,~] = getWindow(subject(i),inputType, dlmos(i).minDLMO_datenum);
    dlmoToUseAsDatenum = dlmos(i).maxDLMO_datenum - dlmos(i).maxDLMO_fraction + fractionalDLMOToUse(i);
    dlmoShiftedByStartTime(i) = (dlmoToUseAsDatenum - subject(i).epochtime(startIndex))*24;
    startPointOffsetInHours(i) = mod(24*(dlmos(i).maxDLMO_datenum - dlmos(i).maxDLMO_fraction - subject(i).epochtime(startIndex)),24);
    dlmoAsDatenum(i) = dlmoToUseAsDatenum;

end

cleanedOffsets = startPointOffsetInHours(nonexcludedIndices);
cleanedDLMOShiftedByStartTimeInHours = dlmoShiftedByStartTime(nonexcludedIndices);

dlmoIdentifiers = {dlmos.ID};
cleanedDLMOIdentifiers = dlmoIdentifiers(nonexcludedIndices);

sleepMidpointDLMO = csvread('data/SleepVariablesActigraphy.csv');
cleanedSleepMidpointDLMO = sleepMidpointDLMO(nonexcludedIndices,4); % Average sleep midpoint

cleanedDLMOAsDatenum = dlmoAsDatenum(nonexcludedIndices);

data = struct();
data.dlmoRelativeToStart = cleanedDLMOShiftedByStartTimeInHours;
data.identifiers = cleanedDLMOIdentifiers;
data.offsets = cleanedOffsets;
data.sleepMidpointDLMO = cleanedSleepMidpointDLMO;
data.subjects = cleanedSubjectPool;
data.fractionalDLMOToUse = cleanedFractionalDLMOToUse;
data.dlmoAsDatenum = cleanedDLMOAsDatenum;

end