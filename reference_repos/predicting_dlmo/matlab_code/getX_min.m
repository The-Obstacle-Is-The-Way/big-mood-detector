function [t,y,xmin]  = getX_min(subject,dlmoAsDatenum, ic, params, str, model)
% DLMO takes data vector y (with three columns), time vector t, and reference angle
% theta_ref as input and computes the DLMO times on the last time.
% ic is the initial condition to simulate the model.
% tau is the circadian period, which is 24.2.
% str takes the input of data, 'activity','light' or 'convertActivity'
% model takes the model we want to simulate,
% 'kronauerJewett','simple','nonphotic',or'hannay'

[startIndex,endIndex,modelInput] = getWindow(subject,str,dlmoAsDatenum);

if strcmp(str,'convertActivity')
    modelInput = basicsteps(0,0.1*max(subject.activity)/2,...
        0.25*max(subject.activity)/2, 0.4*max(subject.activity)/2,...
        0,100,200,500,2000,modelInput)';
end

modelInput(isnan(modelInput)) = 0;
timeDelta = subject.epochtime(2) - subject.epochtime(1);

if round(timeDelta*24*60*2) == 1
    dt = 0.5/60;
else
    dt = 1/60;
end

subjectDuration = (length(modelInput) - 1)*dt;
time = (0:dt:subjectDuration)';
sleepWake = double(subject.sleepwake(startIndex:min(endIndex,length(subject.sleepwake))));

if length(sleepWake) < length(modelInput)
    l = length(modelInput) - length(sleepWake);
    sleepWake = [sleepWake; zeros(l,1)];
end

timeAsDatenums = subject.epochtime;
datetimeObject = datetime(timeAsDatenums(startIndex),'ConvertFrom','datenum','TimeZone','America/New_York');
localHourAtStart = hour(datetimeObject) + minute(datetimeObject)/60.0; % + hours(tzoffset(datetimeObject)); % Time Zone Offset is unnecessary here.

hold on;
drawDays(time,localHourAtStart);
plot(time,modelInput/200,'Color',[0.7 0.9 1.0]);

lightStruct = struct('dur',subjectDuration,'time',time,'light',modelInput,...
    'sw',sleepWake);
[t,y] = circadianModel(lightStruct,params,ic,model); 

if strcmp(model,'hannay')
    xmin = [];
else
    useMinXAsCBTMin = 1;
    
    if useMinXAsCBTMin
        xMinima = islocalmin(y(:,1),'MinProminence',0.05);
        xmin = t(xMinima);
    else 
        phasePoints = islocalmin(abs(atan2(y(:,2),y(:,1)) - -2.98), 'MinProminence',0.05);
        xmin = t(phasePoints) + 0.97;
    end
end
