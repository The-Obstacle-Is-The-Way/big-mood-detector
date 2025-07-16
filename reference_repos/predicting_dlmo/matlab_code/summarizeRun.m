%% Extract values 
sleepMidpointDLMO = data.sleepMidpointDLMO;
offsetsFromDatenumActual = data.offsets;
predictedPhaseShiftedByStartOffset = predictedPhase - offsetsFromDatenumActual';

dlmoShiftedByDatenum = dlmoRelativeToStartInHours - offsetsFromDatenumActual;

ActualDLMOToWriteToFile = mod(dlmoShiftedByDatenum,24)';
PredictedPhaseToWriteToFile = mod(predictedPhaseShiftedByStartOffset,24);

%% Shift and plot lines 
predictionToPlot = applyBounds(PredictedPhaseToWriteToFile);
actualDLMOToPlot = applyBounds(ActualDLMOToWriteToFile);
sleepMidpointToPlot = applyBounds(sleepMidpointDLMO);

plotSlope1Fit(predictionToPlot,actualDLMOToPlot, 'Predicted DLMO');
print(gcf,[pwd '/figures/figure_limit_cycle_model'],'-dpng','-r600');  

plotSlope1Fit(sleepMidpointToPlot,actualDLMOToPlot,'DLMO Proxy');
print(gcf,[pwd '/figures/figure_sleep_onset'],'-dpng','-r600');  

%% Write to table
filename = [pwd '/figures/',model,'_',inputType, '_',dlmoTypeToUse, '.csv'];
T = table(ActualDLMOToWriteToFile,sleepMidpointDLMO,PredictedPhaseToWriteToFile,'RowNames',data.identifiers);
writetable(T,filename,'WriteRowNames',true);

%% Day length figure 
dlmoDates = [data.dlmoAsDatenum];
dayLengths = dateNumToDayLength(dlmoDates);
dayLengthFileName = [pwd '/figures/dayLengths',model,'_',inputType, '_',dlmoTypeToUse, '.csv'];

T = table(dayLengths','RowNames',data.identifiers);
writetable(T,dayLengthFileName,'WriteRowNames',true);

figure; plot(dayLengths,abs(predictionError),'ko');
set(gcf,'Color','w')
set(gca,'FontSize',18);
xlabel('Day length (hours)')
ylabel('Absolute prediction error (hours)')
box off

p = polyfit(dayLengths',abs(predictionError),1);
dayLengthX = min(dayLengths)-2:.1:max(dayLengths)+2;
hold on;
plot(dayLengthX,dayLengthX*p(1) + p(2),'k--')
print(gcf,[pwd '/figures/figure_day_length'],'-dpng','-r600');  

durations = 8.5:2:15.5;
avgErrorByDuration = [];
stdByDuration = [];
semByDuration = [];

for dayLengthDuration = durations
    inds = intersect(find(dayLengths >= dayLengthDuration - 1),find(dayLengths < dayLengthDuration+1));
    avgErrorByDuration = [avgErrorByDuration mean(abs(predictionError(inds)))];
    stdByDuration = [stdByDuration std(abs(predictionError(inds)))];
    semByDuration = [semByDuration std(abs(predictionError(inds)))/sqrt(length(inds))];
end

figure; 
errorbar(durations,avgErrorByDuration,stdByDuration,'ko--','MarkerSize',15,'LineWidth',1.5)
xlim([8 15.5])
box off; set(gca,'Color','w'); set(gca,'FontSize',16); set(gcf,'Color','w');
xlabel('Day length (hours)')
ylabel('Average absolute error (hours)')

print(gcf,[pwd '/figures/figure_day_length_binned'],'-dpng','-r600');  

%% Look at outliers 
cleanedSubjects = data.subjects;
ids = {cleanedSubjects.ID};
highErrorSubjects = find(abs(predictionError) > 7)
ids(highErrorSubjects)

for highError = highErrorSubjects'
   figure(22+highError) 
end

outliersRemovedPrediction = predictionToPlot;
outliersRemovedActual = actualDLMOToPlot;
outliersRemovedPrediction(highErrorSubjects') = [];
outliersRemovedActual(highErrorSubjects') = [];

predictionErrorFromSleepMidpoint = mod(ActualDLMOToWriteToFile - sleepMidpointDLMO + 12, 24) - 12;
predictionErrorOutliersRemoved = mod(outliersRemovedActual - outliersRemovedPrediction + 12, 24) - 12;

fprintf('Mean absolute prediction error from model: %f\n', mean(abs(predictionError)))
fprintf('Mean absolute prediction error from sleep midpoint: %f\n', mean(abs(predictionErrorFromSleepMidpoint)))
fprintf('Mean absolute prediction error from model (outliers removed): %f\n', mean(abs(predictionErrorOutliersRemoved)))


%% MISC

% testLightTherapy(PredictedPhaseToWriteToFile,ActualDLMOToWriteToFile,model,['model'  dlmoTypeToUse])
% print(gcf,[pwd '/figures/figure_model_histogram'],'-dpng','-r600');  
% 
% testLightTherapy(SleepMidpointDLMOEstimateToWriteToFile,ActualDLMOToWriteToFile,model,['sleep' dlmoTypeToUse])
% print(gcf,[pwd '/figures/figure_sleep_onset_histogram'],'-dpng','-r600');  

% runAmplitudeAnalysis(predictionSummary)
