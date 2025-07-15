# mood_ml

This repository provides the code for implementing the trained XGBoost mood prediction models for the paper, "Accurately predicting mood episodes in mood disorder patients using wearable sleep and circadian rhythm features".

### User guide
**Calculating Sleep and Circadian Indexes**

1. **Upload the Sleep Record**
   - Ensure your sleep record is formatted to match the structure of `example.csv`.

2. **Download Required Files**
   - Obtain the following files:
     - `example.csv`
     - `Index_calculation.m`
     - `mnsd.p`
   - Place these three files in the same directory.

3. **Run MATLAB Script**
   - Open MATLAB and run the script `Index_calculation.m`.
   - The script will automatically generate a file named `test.csv`, containing 36 sleep and circadian indexes for each day.

**Estimating Mood Episodes with Pre-Trained Model**

1. **Download Required Files**
   - Obtain the following files:
     - `XGBoost_DE.pkl`
     - `XGBoost_ME.pkl`
     - `XGBoost_HME.pkl`
     - `mood_ml.ipynb`
   - Place these files in the same directory as the previously generated `test.csv` file.

2. **Run Jupyter Notebook**
   - Open the `mood_ml.ipynb` file in Jupyter Notebook.
   - Execute each cell in sequence.
   - This process will generate three CSV files:
     - `expected_outcome_de.csv`
     - `expected_outcome_me.csv`
     - `expected_outcome_hme.csv`


### File description

`example.csv`: example of sleep data before calculating the sleep and circadian indexes. It consists of `sleep_start`, `sleep_end`, `time_in_bed`, `minutes_sleep`, and `minutes_awake`   
&emsp; `sleep_start`: the starting time of sleep window (ex) '2024-01-01 22:00'  
&emsp; `sleep_end`: the end time of sleep window (ex) '2024-01-02 06:00'  
&emsp; `time_in_bed`: the duration (in minutes) of sleep window (ex) 480  
&emsp; `minutes_sleep`: the total sleep time (in minutes) during the sleep window (ex) 475  
&emsp; `minutes_awake`: the wake time (in minutes) during the sleep window (ex) 5  
`mnsd.p`: functions for calculating 36 sleep and circadian indexes  
`Index_calculation.m`: Matlab code for calculating sleep and circadian indexes and save it as an csv file  
`test.csv`: sample data set including 36 sleep and circadian indexes (output of the `Index_calculation.m` file)  
`expected outcome (de).csv`: date (1st column), probability of no depressive episode (2nd column), probability of depressive episode (3rd column)  
`expected outcome (me).csv`: date (1st column), probability of no manic episode (2nd column), probability of manic episode (3rd column)  
`expected outcome (hme).csv`: date (1st column), probability of no hypomanic episode (2nd column), probability of hypomanic episode (3rd column)  
`XGBoost_DE.pkl`: the trained XGBoost model predicting depressive episodes  
`XGBoost_ME.pkl`: the trained XGBoost model predicting manic episodes  
`XGBoost_HME.pkl`: the trained XGBoost model predicting hypomanic episodes  
`mood_ml.ipynb`: the Jupyter notebook code for predicting the mood episodes using the trained models.  

For the calculation of sleep indexes, we referred to the following paper:  
Katori et al., The 103,200-arm acceleration dataset in the UK Biobank revealed a landscape of human sleep phenotypes. PNAS (2022)

For the simulation of human circadian pacemaker, we referred to the codes in https://github.com/ojwalch/predicting_dlmo

### Required packages

`mnsd.p` and `Index_calculation.m` is witten in MATLAB R2022b.
`mood_ml.ipynb` is written in Python 3.  
All codes can be implemented unless the users installed Python 3, Jupyter notebook, and Matlab.  
  
Required Packages in Matlab:  

  findpeaks (in 'signal processing toolbox')
  
Required Packages in Python:  

  pandas  
  pickle

Optionally, 'time' package is needed to calculate the run time. However, it is not crucial to get outcome of the `mood_ml.ipynb` code.  
Installation of these packages typically takes 5~8 sec. 

### Expected outcome and run time

Run time for calculating sleep and circadian indexes depends on the size of data.  
For `example.csv`, it takes 7.125 seconds.

`mood_ml.ipynb` codes includes four cells. Among them, the fourth cell is the most important cell which gives the probability  
Expected run time for the second cell, which imports the `test.csv` and trained models, is 2.089 second.  
Expected run time for the fourth cell, which calculates the probability of mood episodes is 0.029 second.  
