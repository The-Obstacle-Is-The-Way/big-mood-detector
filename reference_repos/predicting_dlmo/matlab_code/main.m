% readData reads in the data, readDLMO reads in DLMO, exclude excludes the
% subjects, ics_limcycle generates the initial conditions, and extractxmin
% generates the error between DLMOactual and DLMOpred

% To change the model, change "model".
% To change the input (light/activity), change "str".

clear; clc; close all;
tic;
 
%% Specify the model and data input we want to use
%Four options: 'simple','nonphotic','hannay','kronauerJewett','cosine','hannay_twopop'
model = 'kronauerJewett';

%Four options: 'activity','light','convertActivity','machineLearning'
inputType = 'activity';

%Three options: max, min, curveFit
dlmoTypeToUse = 'min';

subject = readData;
dlmos = readDLMO;

data = exclude(dlmos,subject,dlmoTypeToUse,inputType);

if strcmp(model,'cosine')
    cosinor_analysis(data);
else
    ics = getICsFromLimitCycle(model, data);
    
    if strcmp(inputType,'machineLearning')
        ml_prediction;
        plotSlope1Fit(predictedPhase,min_dlmo);
    elseif strcmp(model,'hannay') || strcmp(model,'hannay_twopop')
        
        if strcmp(model,'hannay')
            predictionSummary = kevin_dlmo(data, ics, inputType);
        else
            predictionSummary = kevin_dlmo_twopop(data, ics, inputType);
        end
        
        predictedPhase = predictionSummary.predictedPhase;
        predictedPhase = predictedPhase';
        dlmoRelativeToStartInHours = predictionSummary.dlmoRelativeToStartInHours;
        plotSlope1Fit(mod(predictedPhase,24),mod(dlmoRelativeToStartInHours,24), 'Model');
        predictionError = predictionSummary.error;
    else
        predictionSummary = extractX_min(data, ics, inputType, model);
        predictedPhase = predictionSummary.predictedPhase;
        dlmoRelativeToStartInHours = predictionSummary.dlmoRelativeToStartInHours;
        predictionError = predictionSummary.error;
    end
    savefig(['figures/',model,'_',inputType,'_',dlmoTypeToUse,'.fig']);
    saveas(gcf,['figures/',model,'_',inputType,'_',dlmoTypeToUse,'.tiff']);
    save(['figures/',model,'_',inputType,'_',dlmoTypeToUse,'.mat'],'predictionError');
end


summarizeRun;

toc
