function params = dateNumToParams(datenumValue)
dayLength = dateNumToDayLength(datenumValue);
params = containers.Map('alphaScalar',1.0);
if dayLength < 10
    params = containers.Map('alphaScalar',5.0);
end
end