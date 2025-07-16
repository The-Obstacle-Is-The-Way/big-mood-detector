function initialConditions = getICsFromLimitCycle(model, data)

wake = 7.5;
bed = 23.5;
max_light = 800;

subjects = data.subjects;
startPointOffsets = data.offsets;
sleepMidpointDLMOs = data.sleepMidpointDLMO;
observedDLMO = data.dlmoAsDatenum;

for k = 1:length(subjects)
    params = dateNumToParams(observedDLMO(k));
    limitCycle = getLimitCycle(wake, bed, max_light, model,params);
    
    sleepMidpointDLMO = sleepMidpointDLMOs(k);
    predictedSleepOnsetFromSleepMidpoint = mod(sleepMidpointDLMO + 2,24);
    
    % Optional: Try shifting ICs by sleep onset.
    % bedtimeDifference = 24 - predictedSleepOnsetFromSleepMidpoint;
    % index = floor(mod(floor(startTime(k).time) - 1 - bedtimeDifference,24)) + 1; % - 1;
       
    initialConditions(k,:) = limitCycle(24 - startPointOffsets(k));

end
end