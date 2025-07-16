function subjects = readData
% Specify the folder where the files live.
dataFolder = 'data';

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(dataFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s', dataFolder);
    uiwait(warndlg(errorMessage));
    return;
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(dataFolder, '*Table 1.csv'); % Change to whatever pattern you need.
filesMatchingPattern = dir(filePattern);
subjects = struct();

for k = 1:length(filesMatchingPattern)    
    baseFileName = filesMatchingPattern(k).name;
    fullFileName = fullfile(dataFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    % First we read in the file whose name does not contain PH
    % Subject 13, 19, 22, 31, 34, 38, 39 have different format
    
    if ~contains(baseFileName,'PH')
        if contains(baseFileName,'13') || contains(baseFileName,'19') || contains(baseFileName,'22') || contains(baseFileName,'31') || contains(baseFileName,'34') || contains(baseFileName,'38') || contains(baseFileName,'39')
            fileID = fopen(fullFileName);
            csvFile = textscan(fileID,'%s %d %d/%d/%d %d:%d:%d%s %d %f %d %f %f %s','Delimiter',',','HeaderLines',1);
            fclose(fileID);
            ampm = csvFile{9};
            hour = double(csvFile{6});
            minute = double(csvFile{7});
            second = double(csvFile{8});
            month = double(csvFile{3});
            date = double(csvFile{4});
            year = double(csvFile{5});
            for i = 1:length(ampm)
                if strcmp(ampm(i),'AM') && hour(i) == 12
                    hour(i)=0;
                elseif strcmp(ampm(i),'PM') && hour(i) < 12
                    hour(i) = hour(i) + 12;
                end
            end
            
            format long
            epochtime = datenum(year,month,date,hour,minute,second);
            
            subjects(k).ID = baseFileName;
            subjects(k).activity = csvFile{11};
            subjects(k).light = csvFile{13};
            subjects(k).epochtime = epochtime;
            subjects(k).sleepwake = csvFile{14};
            
            interval = csvFile{15};
            status = zeros(length(interval),1);
            
            for i=1:length(interval)
                if strcmp(interval(i),'ACTIVE')
                    status(i) = 0;
                elseif strcmp(interval(i),'EXCLUDED')
                    status(i) = 0;
                else
                    status(i) = 1;
                end
            end
            subjects(k).status = status;
            
        else
            fileID = fopen(fullFileName);
            csvFile = textscan(fileID,'%s %d %d/%d/%d %d:%d:%d%s %f %d %f %f %s','Delimiter',',','HeaderLines',1);
            fclose(fileID);
            ampm = csvFile{9};
            hour = double(csvFile{6});
            minute = double(csvFile{7});
            second = double(csvFile{8});
            month = double(csvFile{3});
            date = double(csvFile{4});
            year = double(csvFile{5});
            for i = 1:length(ampm)
                if strcmp(ampm(i),'AM') && hour(i) == 12
                    hour(i) = 0;
                elseif strcmp(ampm(i),'PM') && hour(i) < 12
                    hour(i) = hour(i) + 12;
                end
            end
            format long
            epochtime = datenum(year,month,date,hour,minute,second);
            subjects(k).ID = baseFileName;
            
            subjects(k).activity = csvFile{10};
            subjects(k).light = csvFile{12};
            subjects(k).epochtime = epochtime;
            subjects(k).sleepwake = csvFile{13};
            interval = csvFile{14};
            status = zeros(length(interval),1);
            
            for i=1:length(interval)
                if strcmp(interval(i),'ACTIVE')
                    status(i) = 0;
                elseif strcmp(interval(i),'EXCLUDED')
                    status(i) = 0;
                else
                    status(i) = 1;
                end
            end
            subjects(k).status = status;
        end
    else
        % Now read in the file with name 'PH+subject'
        
        fileID = fopen(fullFileName);
        csvFile = textscan(fileID,'%s %d %d/%d/%d %d:%d:%d%s %d %f %d %f %f %f %f %f %s','Delimiter',',','HeaderLines',1);
        fclose(fileID);
        ampm = csvFile{9};
        hour = double(csvFile{6});
        minute = double(csvFile{7});
        second = double(csvFile{8});
        month = double(csvFile{3});
        date = double(csvFile{4});
        year = double(csvFile{5});
        for i = 1:length(ampm)
            if strcmp(ampm(i),'AM') && hour(i) == 12
                hour(i) = 0;
            elseif strcmp(ampm(i),'PM') && hour(i) < 12
                hour(i) = hour(i) + 12;
            end
        end
        format long
        epochtime = datenum(year,month,date,hour,minute,second);
        subjects(k).ID = baseFileName;
        
        subjects(k).activity = csvFile{11};
        subjects(k).light = csvFile{13};
        subjects(k).epochtime = epochtime;
        subjects(k).sleepwake = csvFile{17};
        
        interval = csvFile{18};
        status = zeros(length(interval),1);
        
        for i=1:length(interval)
            if strcmp(interval(i),'ACTIVE')
                status(i) = 0;
            elseif strcmp(interval(i),'EXCLUDED')
                status(i) = 0;
            else
                status(i) = 1;
            end
        end
        subjects(k).status = status;
    end
    
end



end