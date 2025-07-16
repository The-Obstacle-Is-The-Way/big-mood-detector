function predictionSummary = extractX_min(cleanedData, ics, str, model)

dlmoRelativeToStartInHours = cleanedData.dlmoRelativeToStart;
subjects = cleanedData.subjects;
dlmoAsDatenum = cleanedData.dlmoAsDatenum;

predictionError = zeros(length(subjects),1); % Records the error between predicted dlmo and actual dlmo
predictedPhase = zeros(length(subjects),1); % Records predicted dlmo

circadianOutputs = {};

for index = 1:length(subjects) 
   
    figure(index + 22);
    params = dateNumToParams(dlmoAsDatenum(index));
    [t, y, xmin] = getX_min(subjects(index), dlmoAsDatenum(index), ics(index,:), params, str, model);
    dlmoOffset = 7.1;

    plot(t,5*y(:,1),'Color',[0.2, 0.2, 0.9]);
    hold on;
    plot(t,y(:,3),'Color',[0.8, 0.8, 0.2]);
    plot([dlmoRelativeToStartInHours(index) dlmoRelativeToStartInHours(index)],[-10 10],'Color',[0.2, 0.8, 0.2]);
    plot([dlmoRelativeToStartInHours(index) + dlmoOffset dlmoRelativeToStartInHours(index) + dlmoOffset],[-10 10],'Color',[0.8, 0.2, 0.2]);
    subject = subjects(index);
    title(subject.ID);
    drawnow; 
    saveas(gcf,['figures/', subject.ID ,'.tiff']);

    % Get the xmin on the day that was closest to the day of DLMO collection
    differenceBetweenActualCBTMinAndPrediction = xmin - (dlmoRelativeToStartInHours(index) + dlmoOffset); 
    indexOfMinimumClosestToCollection = find(abs(differenceBetweenActualCBTMinAndPrediction) == min(abs(differenceBetweenActualCBTMinAndPrediction)),1); 
    predictionError(index) = differenceBetweenActualCBTMinAndPrediction(indexOfMinimumClosestToCollection)  % error in hours
    predictedPhase(index) = xmin(indexOfMinimumClosestToCollection) - dlmoOffset;
    
    circadianOutputs{index} = struct('time',t,'circadianOutput',y);
end

predictionError = mod(predictionError,24);

for index = 1:length(predictionError)
    if predictionError(index) < -12
        predictionError(index) = predictionError(index) + 24;
    end
    if predictionError(index) > 12
        predictionError(index) = predictionError(index) - 24;
    end
end

predictionSummary = struct();
predictionSummary.predictedPhase = predictedPhase;
predictionSummary.dlmoRelativeToStartInHours = dlmoRelativeToStartInHours;
predictionSummary.error = predictionError;
predictionSummary.circadianOutputs = circadianOutputs;

fprintf('Mean absolute error: %f\n\n',mean(abs(predictionError)))


end

