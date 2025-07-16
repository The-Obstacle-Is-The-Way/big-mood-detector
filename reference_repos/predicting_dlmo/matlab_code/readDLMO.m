function dlmos = readDLMO
dlmoFileName = 'data/DLMOCompiled.csv';

fileID = fopen(dlmoFileName);
csvFile = textscan(fileID,'%s %f %f %f %f %f %f %f %f %s %s %s','Delimiter',',','HeaderLines',1);
fclose(fileID);

dlmos = struct();

for i = 1:length(csvFile{1})
    
    dlmos(i).ID = csvFile{1}{i};
    dlmos(i).minDLMO_datenum = datenum(csvFile{12}{i});
    dlmos(i).maxDLMO_datenum = datenum(csvFile{11}{i});
    dlmos(i).maxDLMO_fraction = csvFile{2}(i);
    dlmos(i).minDLMO_fraction = csvFile{6}(i);
    dlmos(i).curveFitDLMO_fraction = csvFile{3}(i);
        
end

end