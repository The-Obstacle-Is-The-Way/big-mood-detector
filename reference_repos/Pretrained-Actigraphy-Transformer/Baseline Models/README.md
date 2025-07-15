# Baseline Models


Note that the data preparation section shows you the setting for the task that's being performed - that is, predicting Benzodiazepine usage, SSRI usage, Depression, Sleep Disorder, or Sleep Abnormalities. These can be changed. The notebooks above are showing results for:

1. 1D_CNN: 
```python
Smoothing = False # False for Raw data, True for Smooth data
Task = "SSRI"
# e.g. Tasks = ["SSRI", "Benzodiazepine", "Sleep Strict", "Sleep Liberal", "Depression"] # pick a task from Tasks and set the "Task" variable in the above line
```

2. 3D_CNN: 
```python
Smoothing = False # False for Raw data, True for Smooth data
Task = "SSRI"
# e.g. Tasks = ["SSRI", "Benzodiazepine", "Sleep Strict", "Sleep Liberal", "Depression"] # pick a task from Tasks and set the "Task" variable in the above line
```

3. ConvLSTM: 
```python
Smoothing = False # False for Raw data, True for Smooth data
Task = "SSRI"
# e.g. Tasks = ["SSRI", "Benzodiazepine", "Sleep Strict", "Sleep Liberal", "Depression"] # pick a task from Tasks and set the "Task" variable in the above line
```

4. LSTM: 
```python
Smoothing = False # False for Raw data, True for Smooth data
Task = "SSRI"
# e.g. Tasks = ["SSRI", "Benzodiazepine", "Sleep Strict", "Sleep Liberal", "Depression"] # pick a task from Tasks and set the "Task" variable in the above line
```
