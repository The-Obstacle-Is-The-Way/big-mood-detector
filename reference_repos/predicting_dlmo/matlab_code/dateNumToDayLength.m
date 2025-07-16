function dayLengths = dateNumToDayLength(dlmoDates)

daysInYear = day(datetime(dlmoDates,'convertFrom','datenum'),'dayofyear');
latitude = 42.3314;
dayLengths = day_length(daysInYear,latitude);

end