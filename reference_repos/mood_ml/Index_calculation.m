
raw = readtable('example.csv');
newdata = mnsd(raw);
writetable(newdata, 'test.csv');
