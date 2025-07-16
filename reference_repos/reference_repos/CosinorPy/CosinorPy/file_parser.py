import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CosinorPy import cosinor
import re

from CosinorPy.helpers import df_add_row

"""
    file_name
    trim: cut the data to keep only timepoints between t_start and t_end. t_start and t_end are read from the first line of the sheet. If their values are -1, -1, no data is removed
    diff: use the differences between two subsequent data as a measurement value (for lumicycle)
    rescale_x: when multiple measurement are available per timepoint distribute the data uniformly within 1 hour (for lumicycle)
    individual: separate the multiple iterations of the same measurement (for lumicycle)
"""
    
def read_excel(file_name, trim=False, diff=False, rescale_x=False, independent=True, remove_outliers=False, skip_tabs=False, min_measurements = 0):
    names = []
     
        
    df = pd.DataFrame(columns=['x', 'y', 'test'], dtype=float)
        
        
    xls_file = pd.ExcelFile(file_name)
    for sheet_idx, sheet_name in enumerate(xls_file.sheet_names):
            
        if skip_tabs and sheet_idx < skip_tabs:    
            continue


        sheet = xls_file.parse(sheet_name, header=None)
        print(f"Reading {sheet_name}...")
            
        M = np.array(sheet)
            
        # get x and y
        x = M[:,0].astype(float)
        y = M[:,1].astype(float)
            
                       
        """
            nans, nans, nas
        """          
        # remove nans from time steps
        y = y[np.argwhere(~np.isnan(x))[:,0]]
        x = x[np.argwhere(~np.isnan(x))[:,0]]
            
     
        if trim:
            x,y = trim_data(x,y)
            
        if diff:
            x,y = differentiate(x,y)                

        if rescale_x:
            x,y = rescale_times(x,y)
                                                                    
        if remove_outliers:
            x,y = remove_outliers_f(x,y)

        if independent:
            df_measurement = pd.DataFrame({'x':x, 'y':y})
            df_measurement['test'] = sheet_name
            names.append(sheet_name)
            #df = df.append(df_measurement)                
            df = pd.concat([df, df_measurement], ignore_index=True)
            
        else:
            #idxs = np.argwhere(x > np.append(x[1:], [0]))[:-1] + 1
            idxs = np.argwhere(x > np.append(x[1:], [0])) + 1
            idxs = np.append([0], idxs)
            for i in range(len(idxs)-1):
                df_measurement = pd.DataFrame({'x':x[idxs[i]:idxs[i+1]], 'y':y[idxs[i]:idxs[i+1]]})
                df_measurement['test'] = sheet_name + '_rep' + str(i+1) 
                names.append(sheet_name + '_rep' + str(i+1))
                #df = df.append(df_measurement)             
                df = pd.concat([df, df_measurement], ignore_index=True)

    """
    nans, nans, nas
    """
    # remove nans from measurement data
    #x = x[np.argwhere(~np.isnan(y))[:,0]]
    #y = y[np.argwhere(~np.isnan(y))[:,0]]

    # remove nans from a dataframe
    df.dropna(inplace=True)

    # remove tests that cannot be used to fit a cosinor model (e.g., have less than 3 measurements)
    if min_measurements > 0:
        to_remove = []
        for test in df.test.unique():
            if len(df[df.test  == test]) < min_measurements:
                to_remove.append(test)
        df = df[~df.test.isin(to_remove)]
        if to_remove:
            print(f'{",".join(to_remove)} removed due to small number of data!')
                                
            
    return df

def generate_test_data_group_random(N, name_prefix = "", characterize_data = False, amplitude=1, **kwargs):
    df = pd.DataFrame(columns=['test','x','y'], dtype=float)
    if characterize_data:
        df_params = pd.DataFrame(dtype=float)

    if not name_prefix:
        name_prefix = "test"

    for i in range(N):
        name = f"{name_prefix}_{i}"
        phases = [2*np.pi*np.random.random(), 2*np.pi*np.random.random(), 2*np.pi*np.random.random()]
        amplitudes = [amplitude,0.5*np.random.random(),0.5*np.random.random()]

        if characterize_data:
            df2, rhythm_params = generate_test_data(name=name, characterize_data = True, phase = phases, amplitudes = amplitudes,  **kwargs)
            #df = df.append(df2, ignore_index=True)
            df = pd.concat([df, df2], ignore_index=True)
            #df_params = df_params.append(rhythm_params, ignore_index=True, sort=False)
            df_params = df_add_row(df_params, rhythm_params)
        else:
            df2 = generate_test_data(name=name, phase = phases, amplitudes = amplitudes, **kwargs)
            #df = df.append(df2, ignore_index=True)
            df = pd.concat([df, df2], ignore_index=True)

    if characterize_data:
        return df, df_params
    else:
        return df



def generate_test_data_group(N, name_prefix = "", characterize_data = False, **kwargs):
    df = pd.DataFrame(columns=['test','x','y'], dtype=float)
    if characterize_data:
        df_params = pd.DataFrame(dtype=float)

    if not name_prefix:
        name_prefix = "test"

    for i in range(N):
        name = f"{name_prefix}_{i}"
       
        if characterize_data:
            df2, rhythm_params = generate_test_data(name=name, characterize_data = True, **kwargs)
            #df = df.append(df2, ignore_index=True)
            df = pd.concat([df, df2], ignore_index=True)
            #df_params = df_params.append(rhythm_params, ignore_index=True, sort=False)
            df_params = df_add_row(df_params, rhythm_params)
        else:
            df2 = generate_test_data(name=name, **kwargs)
            #df = df.append(df2, ignore_index=True)
            df = pd.concat([df, df2], ignore_index=True)

    if characterize_data:
        return df, df_params
    else:
        return df

def generate_test_data(n_components=1, period = 24, amplitudes = None, baseline = 0, lin_comp = 0, amplification = 0, phase = 0, min_time = 0, max_time = 48, time_step = 2, replicates = 1, independent = True, name="test", noise = 0, noise_simple = 1, characterize_data=False):
    df = pd.DataFrame(columns=['test','x','y'], dtype=float)
    x = np.arange(min_time, max_time+time_step, time_step)

    if amplitudes==None:
        amplitudes = np.array([1,1/2,1/3,1/4])
   
    periods = np.array([period, period/2, period/3, period/4])
   
    if (type(phase) == int) or (type(phase)==float):
        phases = np.array([phase, phase, phase, phase])
    else:        
        phases = np.array(phase)    

    for i in range(replicates):
        y = np.zeros(len(x))
        
        for j in range(n_components):
            y += amplitudes[j] * np.cos((x/periods[j])*np.pi*2 + phases[j])
   
        # if amplification < 0: oscillations are damped with time
        # if amplification > 0: oscillations are amplified with time
        # if amplification == 0: oscillations are sustained        
        y *= np.exp(amplification*x)
        
        # if lin_comp != 0: baseline is rising/decreasing with time
        y +=  lin_comp*x
        
        y += baseline
        
        if independent:
            test = name
        else:
            test = name + "_rep" + str(i+1)
        mu = 0
        sigma = noise
            
        NOISE = np.random.normal(mu, sigma, y.shape) 
        if noise_simple:
            y += NOISE
        else:
            # mutliplicative noise
            # sigma from 0 to 1; 
            # 0 ... no noise
            # 1 ... maximal noise
            """
            mu = 1
            sigma = noise            
            y *= np.random.normal(mu, sigma, y.shape) 
            """
            y *= (1 + NOISE)
        
        df2 = pd.DataFrame(columns=['test','x','y'], dtype=float)
        df2['x'] = x
        df2['y'] = y
        df2['test'] = test
        df = pd.concat([df, df2])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)

    if characterize_data:
        
        X_eval = np.linspace(0, 2*period, 1000)
        Y_eval = np.zeros(len(X_eval))
        for j in range(n_components):
            Y_eval += amplitudes[j] * np.cos((X_eval/periods[j])*np.pi*2 + phases[j])
        Y_eval += baseline
        rhythm_params = cosinor.evaluate_rhythm_params(X_eval, Y_eval, period=period)

        rhythm_params['lin_comp'] = lin_comp
        rhythm_params['amplification'] = amplification
        rhythm_params['period'] = period
        rhythm_params['test'] = test

        return df, rhythm_params
    else:
        return df

def read_csv(file_name, sep="\t"):
    
    df1 = pd.read_csv(file_name, sep=sep)

    try:
        header = np.array(list(map(lambda s: re.sub('[^0-9_]','', s), df1.columns[1:])))
        reps = int(header[-1].split('_')[1])
        
        t1 = int(header[0].split("_")[0])
        t2 = int(header[reps].split("_")[0])
        dt = t2-t1
        
        max_time = int(header[-1].split("_")[0])
        
        measurements = max_time//dt
                
        shuffle = np.array([(np.arange(0,measurements*reps+1,reps))+i for i in range(reps)]).flatten()
                
        x = list(map(lambda t: int(t.split('_')[0]), header[shuffle]))
    except:
        reps = 1
        shuffle = np.arange(0,len(df1.columns)-1)
        x = list(map(lambda s: int(re.sub('[^0-9]','', s)), df1.columns[1:]))


    Y = df1.iloc[:,shuffle+1].values
    names = df1.iloc[:,0].values

    df2 = pd.DataFrame(columns=['test','x','y'], dtype=float)
    for y, name in zip(Y, names):
        df_tmp = pd.DataFrame(columns=['test','x','y'], dtype=float)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = name

        df2 = pd.concat([df2, df_tmp])

    df2['x'] = df2['x'].astype(float)
    df2['y'] = df2['y'].astype(float)

    df2 = df2.dropna()

    return df2

def export(df, file_name, independent = True):
    tests = df.test.unique()
    if not independent:
        tests = list(set(map(lambda s:('_').join(s.split('_')[:-1]), tests)))
        tests.sort()
    
    with pd.ExcelWriter(file_name) as writer:  
        for test in tests:
            if independent:
                df[df.test == test][['x','y']].to_excel(writer, sheet_name=test, header=False, index=False)        
            else:
                df[df.test.str.startswith(test)][['x','y']].to_excel(writer, sheet_name=test, header=False, index=False)        
                


def export_csv(df, file_name, individual=False):
    export_JTK(df, file_name, individual=individual)

"""
    EXPORT FOR JTK
"""        
# descriptor: add additional descriptor file
# names: export only these names
# individual: rescale measurements
def export_JTK(df, file_name, descriptor = "", names = [], individual=False):
    if not names:
        names = df.test.unique()
    elif type(names) == str:
        names = [names]
        
    df_export = df[df.test.isin(names)].copy()
        
    df_export['x'] = df_export['x'].map(int)

    # merge multiple equal timepoints for a single measurement: only if individual!!!
    if individual:
        for name in names:
            df_name = df_export[df_export.test == name]
            x,y = np.array(df_name.x), np.array(df_name.y)
            x_unique = np.unique(x)
            y_unique = np.zeros(len(x_unique))
            for i in range(len(x_unique)):
                y_unique[i] = np.mean(y[x == x_unique[i]])
                
            df_export = df_export[df_export.test != name]
            df_tmp = pd.DataFrame({'x':x_unique, 'y':y_unique})
            df_tmp['test'] = name
            #df_export = df_export.append(df_tmp, ignore_index=True)
            df_export = pd.concat([df_export, df_tmp], ignore_index=True)
      
    reps = 0
        
    for name in names:
        x = np.array(df_export[df_export.test == name].x)
        r = sum(x[1:] - x[:-1] < 0)+1
        if reps < r:
            reps = r
                
        
        
    mintime = df_export['x'].min() 
    maxtime = df_export['x'].max()
    timestep = np.argmax(np.bincount(abs(np.array(df_export.x)[1:] - np.array(df_export.x)[:-1])))
    samples_in_one = (maxtime-mintime)//timestep + 1
               
    columns = ["gene"] + ["T"+str(i)+"_Rep"+str(j+1) for i in range(mintime, maxtime+1,timestep) for j in range(0,reps)]
    df = pd.DataFrame(columns=columns, dtype=float)
        
    x_full = np.arange(mintime, maxtime+1, timestep)
    x_fuller = np.array(list(range(mintime, maxtime+1, timestep)) * reps)
    y_full = np.zeros(len(x_full) * reps)
        
        
        
    for name in names:                          
        x,y = np.array(df_export[df_export.test == name].x), np.array(df_export[df_export.test == name].y)
                        
        #idxs = np.where(x[1:] - x[:-1] < 0)
        y_full[:] = np.nan
        
        for t in x_full:
            values = y[x == t]
            locs = np.where(t==x_fuller)[0]
            locs = locs[:len(values)]
            y_full[locs] = values
            
        """
        for r, i in enumerate(idxs):
            for j in range(samples_in_one):
                old_loc = 
                new_loc = samples_in_one*r + j
        """        
                    
                    
                    
                    
            
        sorted_values = []
        for i in range(samples_in_one):
            for j in range(reps):
                #print(i,j, i + (samples_in_one) * j)
                sorted_values.append(y_full[i + (samples_in_one) * j])
        df.loc[df.shape[0]] = [name] + sorted_values
        
    df.to_csv(file_name, sep="\t", index=False, na_rep='NA')
        
    if descriptor:
        f = open(descriptor, 'w')
        f.write(str(samples_in_one) + "\t")
        f.write(str(reps) + "\t")
        f.write(str(timestep) + "\n")
        f.close()

"""
    TRIM TRIM TRIM
"""            
def trim_data(x,y):
    t_start = x[0]
    t_end = y[0]
    x = x[1:]
    y = y[1:]
        
    if t_start != -1:
        y = y[np.logical_and(x >= t_start, x <= t_end)]
        x = x[np.logical_and(x >= t_start, x <= t_end)]
    
    return x,y



"""
    DIFF DIFF DIFF
"""            
def differentiate(x, y):
    # calculate the differentials
    y[:-1] = y[1:] - y[:-1]
    y = y[x <= np.append(x[1:], [0])]
    x = x[x <= np.append(x[1:], [0])]
    return x,y



 
"""
    RESCALE RESCALE RESCALE
"""
def rescale_times(x, y):
    i = 0
    while i < len(x):
        j = i + 1
        while (j < len(x)) and (x[j] == x[i]):
            j = j + 1
               

        n_same = j - i

        for k in range(i+1,j):
            x[k] = x[k] + (1/n_same)*(k-i)                   
        i = j
    
    return x,y
    
def remove_outliers_f(x, y):
    m, s = np.nanmean(y), np.nanstd(y)        
    idxs = np.logical_not(np.logical_or(y > m + 3*s, y < m - 3*s))
    #print(x[np.logical_not(idxs)], y[np.logical_not(idxs)])

    removed = len(x) - len(x[idxs])
    if removed:
        print(f"{removed} outliers removed.")
    
    return x[idxs], y[idxs]

def remove_outliers_df(df):

    for test in df.test.unique():
        x,y = df.loc[df.test == test, 'x'].values, df.loc[df.test == test, 'y'].values
        x,y = remove_outliers_f(x, y)
        df3 = pd.DataFrame()
        df3['x'] = x
        df3['y'] = y
        df3['test'] = test


        df2 = pd.concat([df2, df3], ignore_index=True)
    return df2

"""
RESCALE RESCALE RESCALE
"""
def rescale_to_median(x, y):
    new_x = []
    new_y = []

    i = 0
    while i < len(x):
        j = i + 1
        while (j < len(x)) and (x[j] == x[i]):
            j = j + 1
           
        new_x.append(x[i])
        new_y.append(np.median(y[i:j]))
                        
        i = j

    return np.array(new_x),np.array(new_y)


"""
EXPORT FOR COSINOR 2
"""  
def export_cosinor2(input_file_name, output_file_name, period = 24, trim=False, diff=False, remove_outliers=False, rescale_median = False, remove_lin_comp = False):        
    outputs = []

    xls_file = pd.ExcelFile(input_file_name)
    for sheet_name in xls_file.sheet_names:

        sheet = xls_file.parse(sheet_name, header=None)
        #print(sheet_name)

        M = np.array(sheet)

        # get x and y
        x = M[:,0].astype(float)
        y = M[:,1].astype(float)


        """
        nans, nans, nas
        """          
        # remove nans from time steps
        y = y[np.argwhere(~np.isnan(x))[:,0]]
        x = x[np.argwhere(~np.isnan(x))[:,0]]


        if trim:
           x,y = trim_data(x,y)

        if diff:
           x,y = differentiate(x,y)                

        if remove_outliers:
            x,y = remove_outliers_f(x,y)

        if rescale_median:
            x,y = rescale_to_median(x,y)
            min_x = min(x)
            max_x = max(x)
            full_x = np.arange(min_x, max_x+1)
        else:
            x,y = rescale_times(x,y)
            full_x = np.sort(np.unique(x))
                    
                    
        df = pd.DataFrame()
        df['time'] = full_x

        idxs = np.argwhere(x > np.append(x[1:], [0])) + 1
        idxs = np.append([0], idxs)
        
        
        
        for i in range(len(idxs)-1):
                curr_x = x[idxs[i]:idxs[i+1]]
                curr_y = y[idxs[i]:idxs[i+1]]
                
                if remove_lin_comp:
                    curr_x, curr_y = cosinor.remove_lin_comp(curr_x, curr_y, n_components = 1, period = period)
                
                full_y = np.zeros(len(full_x))
                full_y[:] = np.nan
                for cx in curr_x:                                      
                    full_y[full_x == cx] = curr_y[curr_x == cx]                                
                df['Subject '+str(i+1)] = full_y
        outputs.append((df, sheet_name))

    with pd.ExcelWriter(output_file_name) as writer:  
        for df, sheet_name in outputs:            
            #df.to_excel(writer, sheet_name=sheet_name, index=False)         
            df.transpose().to_excel(writer, sheet_name=sheet_name, header=False)         
            #df.transpose().to_excel(writer, sheet_name=sheet_name, index=False,header=False)         
        
