function [startIndex, endIndex, data] = getWindow(subject,inputType,dlmoDatenum)
% GETWINDOW starts at the first valid data point for the given subject,
% ends on the day of DLMO collection (or closest to it), and pads with
% darkness. The function takes subject data and information about the
% month, day, and time of DLMO collection as input
% str takes the input of data, 'activity','light' or 'convertActivity'

% Find the index of the first valid (in terms of light and activity) data point
USAGE_THRESH = 5000;


if strcmp(inputType,'light')
    input = subject.light;
else
    activity = subject.activity;
    input = activity;
    
end

tempActivity = subject.activity;
tempActivity(isnan(tempActivity)) = 0;

firstBrightLightIndex = min(find(cumsum(tempActivity) > USAGE_THRESH));
validIndices = find(~isnan(subject.light) & ~isnan(subject.activity));
validIndices(validIndices < firstBrightLightIndex) = [];
startIndex = validIndices(1);

timeBetweenTimestampAndDLMO = subject.epochtime - dlmoDatenum;
[~,endIndex] = min(abs(timeBetweenTimestampAndDLMO)); % Get index of recording that's closest to DLMO 
elapsedTimeBetweenSubjectDataEndAndDLMO = subject.epochtime(end) - dlmoDatenum;

data = double(input(startIndex:endIndex));

if elapsedTimeBetweenSubjectDataEndAndDLMO < 0
    
    % Use this value to fill ALL the time between end of collection and DLMO with darkness -- only sensible if the collection is close to the end date of DLMO
    % patch_num = fix(-elapsedTimeBetweenSubjectDataEndAndDLMO*24*60 + 24*60); % Number of minutes between DLMO and end date of data.
    
    patch_num = 0.5*24*60; 
    
    timeDelta = subject.epochtime(2) - subject.epochtime(1);
    if round(timeDelta*24*60*2) == 1
        patch_num = 2*patch_num;
    end
    
    data = [double(input(startIndex:endIndex)); zeros(patch_num,1)];
    
end

end