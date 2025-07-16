from numpy.ma import add
import pandas as pd
import numpy as np
np.seterr(divide='ignore')
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multi
from scipy.optimize import curve_fit
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import percentileofscore
from scipy.stats import circstd, circmean

import seaborn as sns

import copy
import itertools
from matplotlib.lines import Line2D
from random import sample

import os

import copy

from CosinorPy.helpers import df_add_row

#from skopt.space import Space
#from skopt.sampler import Lhs

   
def periodogram_df(df, folder = '', **kwargs):
    names = list(df.test.unique())
    names.sort()  

    for name in names:
        x, y = df[df.test == name].x.values, df[df.test == name].y.values
        if folder:
            save_to = os.path.join(folder, "per_" + name)            
        else:
            save_to = ""

        periodogram(x,y, save_to = save_to, name=name, **kwargs)


def periodogram(X, Y, per_type='per', sampling_f = '', logscale = False, name = '', save_to = '', prominent = False, max_per = 240):
    
    
    if per_type == 'per' or per_type == 'welch':
    
        X_u = np.unique(X)
        Y_u = []
        for x_u in X_u:
            #y_u.append(np.mean(y[t == x]))
            Y_u.append(np.median(Y[x_u == X]))
        
       
        
        if not sampling_f:
            sampling_f = 1/(X[1]-X[0])
        
        Y = Y_u
            
    if per_type == 'per':
        # Fourier
        f, Pxx_den = signal.periodogram(Y,sampling_f)
    elif per_type =='welch':
        # Welch
        f, Pxx_den = signal.welch(Y,sampling_f)
    elif per_type == 'lombscargle':
        # Lomb-Scargle
        min_per = 2
        #max_per = 50
        
        f = np.linspace(1/max_per, 1/min_per, 10)
        Pxx_den = signal.lombscargle(X, Y, f)
    else:
        print("Invalid option")
        return
        

    # significance
    # Refinetti et al. 2007
    p_t = 0.05
    
    N = len(Y)
    T = (1 - (p_t/N)**(1/(N-1))) * sum(Pxx_den) # threshold for significance
    
    if f[0] == 0:
        per = 1/f[1:]
        Pxx = Pxx_den[1:]
    else:
        per = 1/f
        Pxx = Pxx_den
    
    
    Pxx = Pxx[per <= max_per]
    per = per[per <= max_per]
    
    
    try:
        if logscale:
            plt.semilogx(per, Pxx, 'ko')
            plt.semilogx(per, Pxx, 'k--', linewidth=0.5)
            plt.semilogx([min(per), max(per)], [T, T], 'k--', linewidth=1)
        else:
            plt.plot(per, Pxx, 'ko')
            plt.plot(per, Pxx, 'k--', linewidth=0.5)
            plt.plot([min(per), max(per)], [T, T], 'k--', linewidth=1)
    except:
        print("Could not plot!")
        return

    peak_label = ''

    if prominent:    
        locs, heights = signal.find_peaks(Pxx, height = T)
        
        if any(locs):        
            heights = heights['peak_heights']
            s = list(zip(heights, locs))
            s.sort(reverse=True)
            heights, locs = zip(*s)
            
            heights = np.array(heights)
            locs = np.array(locs)
               
            peak_label = ', max peak=' + str(per[locs[0]])            
    
    else:
        locs = Pxx >= T
        if any(locs):          
            heights, locs = Pxx[locs], per[locs]                   
            HL = list(zip(heights, locs))
            HL.sort(reverse = True)
            heights, locs = zip(*HL)
            
            peak_label = ', peaks=\n'
            
            locs = locs[:11]
            for loc in locs[:-1]:
                peak_label += "{:.2f}".format(loc) + ','
            peak_label += "{:.2f}".format(locs[-1])
                

    plt.xlabel('period [hours]')
    plt.ylabel('PSD')
    plt.title(name + peak_label)
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()

def remove_lin_comp_df(df, n_components = 0, period = 24, summary_file=""):
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x.values,df[df['test']==test].y.values
        x,y,fit = remove_lin_comp(x,y,n_components=n_components, period=period, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        #df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            #df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2
    

def remove_lin_comp(X, Y, n_components = 0, period = 24, return_fit=False):
    
    X = np.array(X)
    Y = np.array(Y)

    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = True)
    model = sm.OLS(Y, X_fit)
    results = model.fit()

    CIs = results.conf_int()
    if type(CIs) != np.ndarray:
        CIs = CIs.values
    CI = CIs[1]
    #A = results.params[0]
    k = results.params[1]


    """      
    X_lin = np.zeros(X_fit.shape)
    X_lin[:,1] = X_fit[:,1]
    Y_lin = results.predict(X_lin)
    Y = Y-Y_lin
    """
    #Y_fit = results.predict(X_fir)
    #Y = Y - Y_fit

    
    
    #Y = Y - A - k*X
    if CI[0] * CI[1] > 0: # if both CIs have the same sign
        Y = Y - k*X
    
    if return_fit:
        fit = {}
        fit['k'] = results.params[1]
        fit['CI'] = CI
        fit['p'] = results.pvalues[1]
        fit['A'] = results.params[0]

        return X,Y,fit    
    """
    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = False)
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    plt.plot(X, results.fittedvalues, color="black")
    """
    
    return X, Y
    
# prepare the independent variables
def generate_independents(X, n_components = 3, period = 24, lin_comp = False, remove_lin_comp = False):
    if n_components == 0:
        X_fit = X       
        lin_comp = True
    else:
        for i in np.arange(n_components):
            n = i+1

            A = np.sin((X/(period/n))*np.pi*2)
            B = np.cos((X/(period/n))*np.pi*2)                        

            if not i:
                X_fit = np.column_stack((A, B))            
            else:
                X_fit = np.column_stack((X_fit, np.column_stack((A, B))))                
    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))    
        if remove_lin_comp:
            X_fit[:,0] = 0  
    
    X_fit = sm.add_constant(X_fit, has_constant='add')
    
    return X_fit

# prepare the independent variables for limorhyde
def generate_independents_compare(X1, X2, n_components1 = 3, period1 = 24, n_components2 = 3, period2 = 24, lin_comp = False, non_rhythmic=False, remove_lin_comp=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    X = np.concatenate((X1, X2))
    H_i = np.concatenate((H1, H2))
    X_i = H_i * X
   

    for i in np.arange(n_components1):
        n = i+1

        A = np.sin((X/(period1/n))*np.pi*2)        
        B = np.cos((X/(period1/n))*np.pi*2) 
        if not i:
            X_fit = np.column_stack((A, B))        
        else:
            X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
        
    if non_rhythmic:
        X_fit = np.column_stack((X_fit, H_i))                
    else:
        for i in np.arange(n_components2):
            n = i+1

            A_i = H_i * np.sin((X/(period2/n))*np.pi*2)        
            B_i = H_i * np.cos((X/(period2/n))*np.pi*2) 
        
               
            X_fit = np.column_stack((X_fit, np.column_stack((A_i, B_i))))
        
        X_fit = np.column_stack((X_fit, H_i))
        
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))    
        if remove_lin_comp:
            X_fit[:,0] = 0
            X_fit[:,1] = 0               

    X_fit = sm.add_constant(X_fit, has_constant='add')

    return X_fit


"""
*****************************
* start of finding the best *
*****************************
"""

def get_best_fits(df_results, criterium = 'R2_adj', reverse = False, n_components = []):
    df_best = pd.DataFrame(columns = df_results.columns, dtype=float)
    names = np.unique(df_results.test)
    
    for name in names:
        if n_components:
            for n_comp in n_components:
                if reverse:
                    M = df_results[(df_results.test == name) & (df_results.n_components == n_comp)][criterium].min()
                else:
                    M = df_results[(df_results.test == name) & (df_results.n_components == n_comp)][criterium].max()
                #df_best = df_best.append(df_results[(df_results.test == name) & (df_results.n_components == n_comp) & (df_results[criterium] == M)], ignore_index = True)
                df_best = pd.concat([df_best, df_results[(df_results.test == name) & (df_results.n_components == n_comp) & (df_results[criterium] == M)]], ignore_index = True)
        
        else:
            M = df_results[df_results.test == name][criterium].max()
            #df_best = df_best.append(df_results[(df_results.test == name) & (df_results[criterium] == M)], ignore_index = True)
            df_best = pd.concat([df_best,df_results[(df_results.test == name) & (df_results[criterium] == M)]], ignore_index = True)
    
    return df_best


def get_best_models_population(df, df_models, n_components = [1,2,3], lin_comp = False, criterium = 'RSS', reverse = True):    
    names = np.unique(df_models.test)   
    df_best = pd.DataFrame(columns = df_models.columns, dtype=float)
    df_models = get_best_fits(df_models, criterium = criterium, reverse = reverse, n_components=n_components)
    for test in names:
        n_points = df[df.test.str.startswith(test)].x.shape[0] # razlika med get_best_models in get_best_models_population
        df_test_models = df_models[df_models.test == test]
        df_test_models = df_test_models.sort_values(by=['n_components'])
    
        i = 0
        for new_row in df_test_models.iterrows():            
            if i == 0:
                best_row = new_row
                i = 1
            else:
                RSS_reduced = best_row[1].RSS
                RSS_full = new_row[1].RSS

                DF_reduced = n_points - (best_row[1].n_components * 2 + 1)
                DF_full = n_points - (new_row[1].n_components * 2 + 1)

                if lin_comp:
                    DF_reduced -= 1
                    DF_full -= 1                
                #print (test, old_row[1].n_components, new_row[1].n_components)
                if compare_models(RSS_reduced, RSS_full, DF_reduced, DF_full) < 0.05:
                    best_row = new_row
        #df_best = df_best.append(best_row[1], ignore_index=True)
        df_best = df_add_row(df_best, dict(best_row[1]))
    return df_best


# compare two models according to the F-test
# http://people.reed.edu/~jones/Courses/P24.pdf
# https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?reg_howtheftestworks.htm  
def get_best_models(df, df_models, n_components = [1,2,3], lin_comp = False, criterium='p', reverse = True):
       
    names = np.unique(df_models.test)   
    df_best = pd.DataFrame(columns = df_models.columns, dtype=float)
    df_models = get_best_fits(df_models, n_components = n_components, criterium=criterium, reverse = reverse)


    for test in names:  
        n_points = df[df.test == test].x.shape[0]
        df_test_models = df_models[df_models.test == test]
        df_test_models = df_test_models.sort_values(by=['n_components'])
        i = 0
        for new_row in df_test_models.iterrows():            
            if i == 0:
                best_row = new_row
                i = 1
            else:
                RSS_reduced = best_row[1].RSS
                RSS_full = new_row[1].RSS

                DF_reduced = n_points - (best_row[1].n_components * 2 + 1)
                DF_full = n_points - (new_row[1].n_components * 2 + 1)

                if lin_comp:
                    DF_reduced -= 1
                    DF_full -= 1                
                #print (test, old_row[1].n_components, new_row[1].n_components)
                if compare_models(RSS_reduced, RSS_full, DF_reduced, DF_full) < 0.05:
                    best_row = new_row
                   
        #df_best = df_best.append(best_row[1], ignore_index=True)
        df_best = df_add_row(df_best, dict(best_row[1]))
    return df_best

"""
***************************
* end of finding the best *
***************************
"""

"""
************
* plotting *
************
"""


def plot_data(df, names = [], folder = '', prefix = '', color='black'):
    if not names:
        names = np.unique(df.test) 
        
    for test in names:
        X, Y = np.array(df[df.test == test].x.values), np.array(df[df.test == test].y.values)  
        
        plt.plot(X,Y,'o', markersize=1, color=color)
        plt.title(test)
       
        #test = test.replace("$","")
       
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)               
        
       
        if folder:
            plt.savefig(os.path.join(folder, prefix+test+'.png'))
            plt.savefig(os.path.join(folder, prefix+test+'.pdf'))
            plt.close()
        else:
            plt.show()
    
    
def plot_data_pairs(df, names, folder = '', prefix ='', color1='black', color2='red'):
        
    for test1, test2 in names:
        X1, Y1 = df[df.test == test1].x.values, df[df.test == test1].y.values  
        X2, Y2 = df[df.test == test2].x.values, df[df.test == test2].y.values  
        
        plt.plot(X1,Y1,'o', color=color1, markersize=1, label=test1)
        plt.plot(X2,Y2,'o', color=color2, markersize=1, label=test2)
        plt.legend()
        plt.title(test1 + ' vs. ' + test2)
       
        if folder:            
            plt.savefig(os.path.join(folder,prefix+test1+'_'+test2+'.png'))
            plt.savefig(os.path.join(folder,prefix+test1+'_'+test2+'.pdf'))

            plt.close()
        else:
            plt.show()

def plot_components(X, Y, n_components = 3, period = 24, name = '', save_to = ''):
    
    A = np.sin((X/period)*np.pi*2)
    B = np.cos((X/period)*np.pi*2)
    C = np.sin((X/(period/2))*np.pi*2)
    D = np.cos((X/(period/2))*np.pi*2)
    E = np.sin((X/(period/3))*np.pi*2)
    F = np.cos((X/(period/3))*np.pi*2)
    #G = np.sin((X/(period/4))*np.pi*2)
    #H = np.cos((X/(period/4))*np.pi*2) 
    
    
    fig, axs = plt.subplots(n_components, 2, constrained_layout=True)
    fig.suptitle(name, fontsize=16)    
    
    
    
    axs[0,0].plot(A, Y,'.')
    axs[0,0].set(xlabel = 'sin((x/'+str(period)+') * 2$\pi$)')
       
    axs[0,1].plot(B, Y,'.')
    axs[0,1].set(xlabel = 'cos((x/'+str(period)+') * 2$\pi$)')
    
    
    if n_components >= 2:      
        axs[1,0].plot(C, Y,'.')
        axs[1,0].set(xlabel = 'sin((x/'+str(period/2)+') * 2$\pi$)')
        axs[1,1].plot(D, Y,'.')
        axs[1,1].set(xlabel = 'cos((x/'+str(period/2)+') * 2$\pi$)')
        
    if n_components == 3:        
        axs[2,0].plot(E, Y,'.')
        axs[2,0].set(xlabel = 'sin((x/'+str(period/3)+') * 2$\pi$)')
        axs[2,1].plot(F, Y,'.')
        axs[2,1].set(xlabel = 'cos((x/'+str(period/3)+') * 2$\pi$)')
  
    if n_components == 4:        
        axs[3,0].plot(E, Y,'.')
        axs[3,0].set(xlabel = 'sin((x/'+str(period/4)+') * 2$\pi$)')
        axs[3,1].plot(F, Y,'.')
        axs[3,1].set(xlabel = 'cos((x/'+str(period/4)+') * 2$\pi$)')
  
   
    for ax in axs.flat:
        ax.set(ylabel = 'y')

    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()


def plot_phases(acrs, amps, tests, period=24, colors = ("black", "red", "green", "blue"), folder = "", prefix="", legend=True, CI_acrs = [], CI_amps = [], linestyles = [], title = "", labels = []):#, plot_measurements = False, measurements=None):
    acrs = np.array(acrs, dtype = float)
    amps = np.array(amps, dtype = float)
    
    if colors and len(colors) < len(tests):
        colors += ("black",) * (len(tests)-len(colors))

    x = np.arange(0, 2*np.pi, np.pi/4)
    x_labels = list(map(lambda i: 'CT ' + str(i) + " ", list((x/(2*np.pi) * period).astype(int))))
    x_labels[1::2] = [""]*len(x_labels[1::2])

    ampM = np.max(amps)
    amps /= ampM
    
    acrs = -acrs
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_offset(0.5*np.pi)
    ax.set_theta_direction(-1) 
    lines = []

    for i, (acr, amp, test, color) in enumerate(zip(acrs, amps, tests, colors)):
  
        """
        if "LDL" in test:
            color = "#FF0000"
        elif "HDL" in test:
            color = "#0000FF"
        elif "CHOL" in test:
            color = "#00FF00"
        elif "control" in test.lower():
            color = "#000000"
        else:
            color = "#0000FF"            
        """
        if linestyles:
            #ax.plot([acr, acr], [0, amp], label=test, color=color, linestyle = linestyles[i])
            ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, alpha = 0.75, linewidth=2, linestyle = linestyles[i]) )
            lines.append(Line2D([0], [0], color=color, linewidth=2, linestyle=linestyles[i]))
        else:
            #ax.plot([acr, acr], [0, amp], label=test, color=color)
            ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, alpha = 0.75, linewidth=2) )
            lines.append(Line2D([0], [0], color=color, linewidth=2))

        #ax.plot([acr, acr], [0, amp], label=test, color=color)
    
        #ax.annotate("", xy=(acr, amp), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, linewidth=2) )
        
        if CI_acrs and CI_amps:
            amp_l, amp_u = np.array(CI_amps[i])/ampM
            amp_l = max(0, amp_l)
            amp_u = min(1, amp_u)
                   
            acr_l, acr_u = -np.array(CI_acrs[i])
             
            if acr_l - acr_u > 2*np.pi:
                plt.fill_between(np.linspace(0, np.pi*2, 1000), amp_l, amp_u, color=color, alpha=0.1)
            elif acr_u < acr_l:
                acr_l, acr_u = acr_u, acr_l
                plt.fill_between(np.linspace(acr_l, acr_u, 1000), amp_l, amp_u, color=color, alpha=0.1)

       

    ax.set_rmax(1)
    ax.set_rticks([0.5])  # Less radial ticks
    ax.set_yticklabels([""])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True)
    ax.set_facecolor('#f0f0f0')
       
    """
    for i, (acr, amp, test, color) in enumerate(zip(acrs, amps, tests, colors)):
        if plot_measurements:
            try:
                x,y = measurements
            except:
                df = measurements
                x,y=df[df.test == test].x, df[df.test == test].y
            plt.plot(x,y,'o',markersize=1, alpha = 0.75, color=color)
    """

    name = "_".join(tests)
    #ax.set_title(name, va='bottom')
    if title:
        ax.set_title(title, va='bottom')
    else:
        ax.set_title(name, va='bottom')

    if legend:
        if labels:
            plt.legend(lines, labels, bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0., frameon=False)
        else:
            plt.legend(lines, tests, bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0., frameon=False)
        #ax.legend()
    if folder:
        plt.savefig(os.path.join(folder,prefix+name+"_phase.pdf"))
        plt.savefig(os.path.join(folder,prefix+name+"_phase.png"))
        plt.close()
    else:
        plt.show()


### pivot a dataframe for heatmaps and PCA/tSNE plot
def pivot_data(df, merge_repeats = True, z_score = True, df_results=False, sort_by="p", ascending=True, filter_significant=False, dropnacols=False):
    if merge_repeats:
        # calculate means of repeats for a timepoint and for a test
        df_heatmap = df.groupby(["test","x"]).mean().reset_index()
    else:
        # separate repeats and denote them as ___rep0, ___rep1, ___rep2,...
        df_heatmap = pd.DataFrame(columns = df.columns)

        tests = df.test.unique()    
        for test in tests:
            df_test = df[df['test'] == test].copy()      
            X = df_test.x.values
            locs = list(np.where(X[:-1] > X[1:])[0]+1)
            locs += [len(X)]

            for i,(i1,i2) in enumerate(zip([0] + locs[:-1], locs)):
                df_test.iloc[i1:i2,2] = f"{test}___rep{i}"      
            #df_heatmap = df_heatmap.append(df_test, ignore_index=True)
            df_heatmap = pd.concat([df_heatmap, df_test], ignore_index=True)

    # convert timepoints to integers
    df_heatmap['x'] = df_heatmap['x'].astype(int)
    df_heatmap = df_heatmap.pivot("test", "x", "y")
    
    # scale data to obtain z_scores 
    if z_score: 
        Y = df_heatmap.values
        means = np.nanmean(Y, axis=1)
        means = np.transpose(np.broadcast_to(means, np.transpose(Y).shape))
        stds = np.nanstd(Y, axis=1)
        stds = np.transpose(np.broadcast_to(stds, np.transpose(Y).shape))
        Y = (Y-means)/stds
        df_heatmap.iloc[:,:] = Y

    # sort tests by significance, amplitude, acrophase, etc.
    if type(df_results) != bool:
        df_results = df_results.copy()
        if filter_significant:
            df_results = df_results[df_results['p(amplitude)']<0.05]
        
        
        #if sort_by == 'acrophase':
        #    df_results['acrophase'] = [a + 2*np.pi if a < 0 else a for a in df_results['acrophase']]        
        
        df_sorted = df_results.sort_values(by=[sort_by], ascending=ascending)                      
        
        sort_mapping = {name:val for val, name in enumerate(df_sorted.test.values)}        
        if not merge_repeats:
            test_orig = [test.split("__")[0] for test in df_heatmap.index]
        else:
            test_orig = df_heatmap.index        
        sorter = [sort_mapping[t] if t in sort_mapping else np.nan for t in test_orig]
        
        df_heatmap['sorter'] = sorter
        df_heatmap = df_heatmap.sort_values(by=['sorter','test'])        
        if filter_significant:
            df_heatmap = df_heatmap.dropna(axis=0)
        df_heatmap = df_heatmap.drop(columns=['sorter'])
        
    if dropnacols:
        df_heatmap = df_heatmap.dropna(axis=1)

    return df_heatmap


def plot_heatmap(df, clustermap=True, xlabel='Time [h]',  folder="", prefix="", **kwargs):

    df_heatmap = pivot_data(df, **kwargs)
   
    ##########
    # heatmap
    ax1 = sns.heatmap(df_heatmap,        
                    yticklabels=True, 
                    cbar=False,
                    linewidths = 0,
                    cmap="vlag")
    
    df_heatmap = df_heatmap.dropna()
    
    

    fig = plt.gcf()
    #x,y = fig.get_size_inches()
    fig.set_size_inches(10,len(df_heatmap)*0.5)
       
    ax1.set_ylabel("")
    ax1.set_xlabel(xlabel)
    
    if folder:
        plt.savefig(os.path.join(folder, prefix+'heatmap'+'.png'))
        plt.savefig(os.path.join(folder, prefix+'heatmap'+'.pdf'), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

        
    if not clustermap:
        return 
    
    #############
    # clustermap

    ax2 = sns.clustermap(df_heatmap, 
                yticklabels=True, 
                cbar=False,
                linewidths = 0,
                cmap="vlag")
    
    ax3 = ax2.ax_heatmap
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("")

    ax2.cax.set_visible(False)

    if folder:
        plt.savefig(os.path.join(folder, prefix+'clustermap'+'.png'))
        plt.savefig(os.path.join(folder, prefix+'clustermap'+'.pdf'), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()

# make groups for the PCA plot - can be used to plot different factor values with different colours
def make_groups(df, sep='-'):

    tests = df.test.unique()
    genes = set(map(lambda x:x.split(sep)[0], tests))
    cells = set(map(lambda x:x.split(sep)[1], tests))

    group_genes = {}
    for gene in genes:
        for test in tests:
            if gene in test:
                if gene in group_genes:
                    group_genes[gene].append(test)  
                else:
                    group_genes[gene] = [test]

    group_cells = {}
    for cell in cells:
        for test in tests:
            if cell in test:
                if cell in group_cells:
                    group_cells[cell].append(test)  
                else:
                    group_cells[cell] = [test]

    groups = {'genes': group_genes,
              'cells': group_cells}

    return groups

def plot_PCA(df, n_components = 2, tSNE=False, perplexity=10, groups={}, folder="", prefix="", **kwargs):
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except:
        print("sklearn needs to be installed for PCA plot")
        return()


    df_data = pivot_data(df, **kwargs)
    M = df_data.values
  
    
    if not tSNE:
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(M)
        pca_explained = pca.explained_variance_ratio_
    else:
        tSNE  = TSNE(n_components=n_components, perplexity = perplexity,  n_iter=5000)
        comps = tSNE.fit_transform(M)     
    
    tests = df_data.index                        
    for c in itertools.combinations(range(n_components), 2):
        i1 = c[0]
        i2 = c[1]
    
        if groups:
            for factor in groups:
                for group, test_group in groups[factor].items():        
                    tests_orig = [test.split("__")[0] for test in tests]                    
                    locs = np.isin(tests_orig, test_group)
                    comps1 = comps[locs,i1]
                    comps2 = comps[locs,i2]
                    plt.plot(comps1, comps2,"o", label=group)           
            
                plt.title(f'{factor}')
            
                if not tSNE:
                    plt.xlabel("PC"+str(i1+1) + " (" + str(round(100*pca_explained[i1],2))+"%)")
                    plt.ylabel("PC"+str(i2+1) + " (" + str(round(100*pca_explained[i2],2))+"%)")
                else:
                    plt.xlabel("PC"+str(i1+1))
                    plt.ylabel("PC"+str(i2+1))
                plt.legend()
                plt.gcf().set_size_inches(20,10)            
                if folder:
                    name = 'tSNE' if tSNE else 'PCA'                                        
                    plt.savefig(os.path.join(folder, f'{prefix}{name}_{factor}_PC{i1}_PC{i2}.png'))
                    plt.savefig(os.path.join(folder, f'{prefix}{name}_{factor}_PC{i1}_PC{i2}.pdf'), bbox_inches = 'tight')
                    plt.close()
                else:
                    plt.show()
            
        else:

            for i,test in enumerate(tests):
                x = comps[i,i1]
                y = comps[i,i2]

                plt.text(x+0.05,y+0.05,test)            
                plt.plot(x,y, 'o')


            if not tSNE:
                plt.xlabel("PC"+str(i1+1) + " (" + str(round(100*pca_explained[i1],2))+"%)")
                plt.ylabel("PC"+str(i2+1) + " (" + str(round(100*pca_explained[i2],2))+"%)")
            else:
                plt.xlabel("PC"+str(i1+1))
                plt.ylabel("PC"+str(i2+1))            
            plt.gcf().set_size_inches(20,10)
            
            if folder:
                name = 'tSNE' if tSNE else 'PCA'                    
                plt.savefig(os.path.join(folder, f'{prefix}{name}_PC{i1}_PC{i2}.png'))
                plt.savefig(os.path.join(folder, f'{prefix}{name}_PC{i1}_PC{i2}.pdf'), bbox_inches = 'tight')
                plt.close()
            else:
                plt.show()
            
            
            plt.show()


"""
*******************
* end of plotting *
*******************
"""

"""
*****************************
* start of fitting wrappers *
*****************************
"""


def fit_group(df, n_components = 2, period = 24, names = "", folder = '', prefix='', **kwargs):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'R2', 'R2_adj', 'log-likelihood', 'amplitude', 'acrophase', 'mesor', 'peaks', 'heights', 'troughs', 'heights2'], dtype=float)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not any(names):
        names = np.unique(df.test) 

    for test in names:
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                X, Y = df[df.test == test].x.values, df[df.test == test].y.values        
                if folder:                    
                    save_to = os.path.join(folder,prefix+test+'_compnts='+str(n_comps) +'_per=' + str(per))
                else:
                    save_to = ''
                                
                results, statistics, rhythm_param, _, _ = fit_me(X, Y, n_components = n_comps, period = per, name = test, save_to = save_to, **kwargs)
            
                try:
                    R2, R2_adj = results.rsquared,results.rsquared_adj
                except:
                    R2, R2_adj = np.nan, np.nan

                df_results = df_add_row(df_results,
                                        {'test': test, 
                                        'period': per,
                                        'n_components': n_comps,
                                        'p': statistics['p'], 
                                        'p_reject': statistics['p_reject'],
                                        'RSS': statistics['RSS'],
                                        'R2': R2, 
                                        'R2_adj': R2_adj,
                                        'ME': statistics['ME'],
                                        'resid_SE': statistics['resid_SE'],
                                        'log-likelihood': results.llf,        
                                        'amplitude': rhythm_param['amplitude'],
                                        'acrophase': rhythm_param['acrophase'],
                                        'mesor': rhythm_param['mesor'],
                                        'peaks': rhythm_param['peaks'],
                                        'heights': rhythm_param['heights'],
                                        'troughs': rhythm_param['troughs'],
                                        'heights2': rhythm_param['heights2']})
                if n_comps == 0:
                    break
    
    df_results.q = multi.multipletests(df_results.p, method = 'fdr_bh')[1]
    df_results.q_reject = multi.multipletests(df_results.p_reject, method = 'fdr_bh')[1]
    
    
    return df_results

def population_fit_group(df, n_components = 2, period = 24, folder = '', prefix='', names = [], **kwargs):

    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'RSS', 'amplitude', 'acrophase', 'mesor'], dtype=float)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not any(names):
        names = np.unique(df.test) 

    names = list(set(list(map(lambda x:x.split('_rep')[0], names))))
    names.sort()
    
    
    for name in set(names):
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                    
                    
                df_pop = df[df.test.str.startswith(name)]   

                if folder:                    
                    save_to=os.path.join(folder,prefix+name+'_compnts='+str(n_comps) +'_per=' + str(per))                    
                    _, statistics, statistics_params, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, save_to = save_to, **kwargs)
                else:                    
                    _, statistics, statistics_params, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, **kwargs)


                d = {'test': name, 
                    'period': per,
                    'n_components': n_comps,
                    'p': statistics['p'], 
                    'p_reject': statistics['p_reject'],
                    'RSS': statistics['RSS'],
                    'ME': statistics['ME'],
                    'resid_SE': statistics['resid_SE'],
                    'amplitude': rhythm_params['amplitude'],
                    'acrophase': rhythm_params['acrophase'],
                    'mesor': rhythm_params['mesor']}    
                
                
                ind_params_stats = statistics_params['ind_params_stats'] 
                for a,b in ind_params_stats.items():
                    d[a] = b

                df_results = df_add_row(df_results, d)
                if n_comps == 0:
                    break
    
    df_results.q = multi.multipletests(df_results.p, method = 'fdr_bh')[1]
    df_results.q_reject = multi.multipletests(df_results.p_reject, method = 'fdr_bh')[1]
        
    return df_results

"""
***************************
* end of fitting wrappers *
***************************
"""



"""
******************************
* start of fitting functions *
******************************
"""
    
def population_fit(df_pop, n_components = 2, period = 24, lin_comp= False, model_type = 'lin', 
                   plot = True, plot_measurements=True, plot_individuals=True, plot_margins=True, hold = False, save_to = '', x_label='', y_label='', 
                   return_individual_params = False, params_CI = False, params_CI_analysis = 'sampling', bootstrap_size = 1000, samples_per_param_CI=5, max_samples_CI = 1000, 
                   sampling_type = "LHS", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], color="black", **kwargs):    

    

    #if return_individual_params:
    ind_params = {}
    ind_params_stats = {}
    for param in parameters_to_analyse:
        ind_params[param] = []

        
    params = -1

    tests = df_pop.test.unique()
    k = len(tests)
    

    #X_test = np.linspace(0, 2*period, 1000)
    #X_fit_eval_params = generate_independents(X_test, n_components = n_components, period = period, lin_comp = lin_comp)
    #if lin_comp:
    #    X_fit_eval_params[:,1] = 0    
    
    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)
    min_Y = np.min(df_pop.y.values)
    max_Y = np.max(df_pop.y.values)


    if plot:
        if plot_measurements:
            X_plot = np.linspace(min(min_X,0), 1.1*max(max_X,period), 1000)
        else:
            X_plot = np.linspace(0, 1.1*period, 1000)

        X_plot_fits = generate_independents(X_plot, n_components = n_components, period = period, lin_comp = lin_comp)
        #if lin_comp:
        #    X_plot_fits[:,1] = 0   

    """
    min_X = 1000
    max_X = 0
    min_Y = 1000
    max_Y = 0
    min_X_test = np.min(X_test)
    """
    min_Y_test = 1000
    max_Y_test = 0
    
    
    for test in tests:
        x,y = df_pop[df_pop.test == test].x.values, df_pop[df_pop.test == test].y.values
        
        """
        min_X = min(min_X, np.min(x))
        max_X = max(max_X, np.max(x))
        
        min_Y = min(min_Y, np.min(y))
        max_Y = max(max_Y, np.max(y))
        """

        results, statistics, rhythm_params, X_test, Y_test, model = fit_me(x, y, n_components = n_components, period = period, plot = False, return_model = True, lin_comp=lin_comp, **kwargs)
        X_fit_eval_params = generate_independents(X_test, n_components = n_components, period = period, lin_comp = lin_comp, remove_lin_comp=True)
        if lin_comp:
            X_fit_eval_params[:,1] = 0    

        #if return_individual_params:
        Y_eval_params = results.predict(X_fit_eval_params)    
        rhythm_ind_params = evaluate_rhythm_params(X_test, Y_eval_params, period=period)    
        for param in parameters_to_analyse:
            ind_params[param].append(rhythm_ind_params[param])
            
        if (plot and plot_individuals):
            #Y_eval_params = results.predict(X_fit_eval_params)            
            Y_plot_fits = results.predict(X_plot_fits)            
            if (plot and plot_individuals):
                if not hold:
                    plt.plot(X_plot,Y_plot_fits,color=color, alpha=0.25, label=test)
                else:
                    plt.plot(X_plot,Y_plot_fits,color=color, alpha=0.25)
            
            min_Y_test = min(min_Y_test, np.min(Y_plot_fits))
            max_Y_test = max(max_Y_test, np.max(Y_plot_fits))
            
        
        if plot and plot_measurements:
            plt.plot(x,y,'o', color=color, markersize=1)

        if type(params) == int:
            params = results.params
            if plot and plot_margins:
                #_, lowers, uppers = wls_prediction_std(results, exog=X_fit_eval_params, alpha=0.05)     
                Y_plot_fits_all = Y_plot_fits
        else:
            params = np.vstack([params, results.params])
            if plot and plot_margins:
                #_, l, u = wls_prediction_std(results, exog=X_fit_eval_params, alpha=0.05)    
                #lowers = np.vstack([lowers, l])
                #uppers = np.vstack([uppers, u])
                Y_plot_fits_all = np.vstack([Y_plot_fits_all, Y_plot_fits])
    


    # parameter statistics: means, variances, standard deviations, confidence intervals, p-values
    #http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    if k > 1:
        means = np.mean(params, axis=0)
        variances = np.sum((params-np.mean(params, axis=0))**2, axis = 0)/(k-1) # np.var(params, axis=0) # isto kot var z ddof=k-1
        sd = variances**0.5
        se = sd/((k-1)**0.5)
        T0 = means/se
        p_values = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        t = abs(stats.t.ppf(0.05/2,df=k-1))
        lower_CI = means - ((t*sd)/((k-1)**0.5))
        upper_CI = means + ((t*sd)/((k-1)**0.5))        
        results.initialize(model, means)


        # analysis of amplitude, mesor, acrophase...
        for ind_param in parameters_to_analyse:
            ind_vals = np.array(ind_params[ind_param])
            
            if ind_param in parameters_angular:
                p_means = project_acr(circmean(ind_vals, high = np.pi, low = -np.pi))
                p_sd = circstd(ind_vals, high = np.pi, low = -np.pi)
            else:
                p_means = np.mean(ind_vals)
                p_variances = np.sum((ind_vals-np.mean(ind_vals))**2)/(k-1)
                p_sd = p_variances**0.5
            
            p_se = p_sd/((k-1)**0.5)
            p_T0 = p_means/p_se
            ind_param_p = 2 * (1 - stats.t.cdf(abs(p_T0), k-1))
            t = abs(stats.t.ppf(0.05/2,df=k-1))
            ind_param_lower_CI = p_means - ((t*p_sd)/((k-1)**0.5))
            ind_param_upper_CI = p_means + ((t*p_sd)/((k-1)**0.5))        
           
            ind_params_stats[f'mean({ind_param})'] = p_means
            ind_params_stats[f'p({ind_param})'] = ind_param_p
            ind_params_stats[f'CI({ind_param})'] = [ind_param_lower_CI, ind_param_upper_CI]          

    else:
        means = params
        sd = np.zeros(len(params))
        sd[:] = np.nan
        se = np.zeros(len(params))
        se[:] = np.nan
        lower_CI = means
        upper_CI = means
        p_values = np.zeros(len(params))
        p_values[:] = np.nan

        for ind_param in parameters_to_analyse:
            ind_params_stats[f'mean({ind_param})'] = ind_params[ind_param][0]
            ind_params_stats[f'p({ind_param})'] = np.nan
            ind_params_stats[f'CI({ind_param})'] = [np.nan, np.nan_to_num]

    x,y = df_pop.x, df_pop.y
    xy = list(zip(x,y))
    xy.sort()
    x,y = zip(*xy)
    x,y = np.array(x), np.array(y)
    X_fit = generate_independents(x, n_components = n_components, period = period, lin_comp = lin_comp)
    Y_fit = results.predict(X_fit)
    
    Y_eval_params = results.predict(X_fit_eval_params) 
    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params, period=period)
        
    if plot:      
        pop_name = "_".join(test.split("_")[:-1])  

        Y_plot_fits = results.predict(X_plot_fits)            
        
        if not hold:
            plt.plot(X_plot,Y_plot_fits, color=color, label="population fit")
        else:            
            plt.plot(X_plot,Y_plot_fits, color=color, label=pop_name)
        plt.legend()
        if x_label:
            plt.xlabel(x_label)    
        else:
            plt.xlabel('time [h]')

        if y_label:
            plt.ylabel(y_label)    
        else:
            plt.ylabel('measurements')
    
        min_Y_test = min(min_Y_test, np.min(Y_eval_params))
        max_Y_test = max(max_Y_test, np.max(Y_eval_params))
    
        

    if plot and plot_margins:
        if k == 1:
            _, lower, upper = wls_prediction_std(results, exog=X_plot_fits, alpha=0.05)        
        else:
            #lower = np.mean(lowers, axis=0)
            #upper = np.mean(uppers, axis=0)
            var_Y = np.var(Y_plot_fits_all, axis=0, ddof = k-1)
            sd_Y = var_Y**0.5
            lower = Y_plot_fits - ((t*sd_Y)/((k-1)**0.5))
            upper = Y_plot_fits + ((t*sd_Y)/((k-1)**0.5))
        plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)                   

    if plot: 
        if plot_measurements:
            if model_type == 'lin':
                plt.axis([min(min_X,0), 1.1*max(max_X,period), 0.9*min(min_Y, min_Y_test), 1.1*max(max_Y, max_Y_test)])
            else:
                plt.axis([min(min_X,0), max_X, 0.9*min(min_Y, min_Y_test), 1.1*max(max_Y, max_Y_test)])
            
        else:
            plt.axis([0, period, min_Y_test*0.9, max_Y_test*1.1])
        
       
    statistics = calculate_statistics(x, y, Y_fit, n_components, period, lin_comp)   
    statistics_params = {'values': means,
                        'SE': se,
                        'CI': (lower_CI, upper_CI),
                        'p-values': p_values,
                        'ind_params_stats': ind_params_stats} 

    if plot:
        #pop_name = "_".join(test.split("_")[:-1])
        if not hold:
            plt.title(pop_name + ', p-value=' + "{0:.5f}".format(statistics['p']))

            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()

    
    if params_CI:
        if params_CI_analysis == 'bootstrap':
            ind_params_stats = population_eval_params_bootstrap(X_test, X_fit_eval_params, model, results, params, bootstrap_size=bootstrap_size, bootstrap_type='std', t_test=True, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period=period)               
        elif params_CI_analysis == 'sampling':
            ind_params_stats = population_eval_params_CI(X_test, X_fit_eval_params, results, statistics_params, rhythm_params, samples_per_param=samples_per_param_CI, max_samples = max_samples_CI, k=k, sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period=period)                
        
        rhythm_params.update(ind_params_stats)
        
    if return_individual_params:        
        return params, statistics, statistics_params, rhythm_params, results, ind_params

    else:
        return params, statistics, statistics_params, rhythm_params, results

def fit_me(X, Y, n_components = 2, period = 24, lin_comp = False, model_type = 'lin', alpha = 0, name = '', save_to = '', plot=True, plot_residuals=False, plot_measurements=True, plot_margins=True, return_model = False, color = False, plot_phase = True, hold=False, x_label = "", y_label = "", rescale_to_period=False, bootstrap=False, bootstrap_size=1000, bootstrap_type="std", params_CI = False, samples_per_param_CI=5, max_samples_CI = 1000, sampling_type="LHS", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase']):
    #print(lin_comp)

    """
    ###
    # prepare the independent variables
    ###
    """
    

    """
    if n_components == 0:
        X_fit = X
        X_fit_test = X_test
        lin_comp = True
    else:
        for i in np.arange(n_components):
            n = i+1

            A = np.sin((X/(period/n))*np.pi*2)
            B = np.cos((X/(period/n))*np.pi*2)                
            A_test = np.sin((X_test/(period/n))*np.pi*2)
            B_test = np.cos((X_test/(period/n))*np.pi*2)

            if not i:
                X_fit = np.column_stack((A, B))
                X_fit_test = np.column_stack((A_test, B_test))     
            else:
                X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
                X_fit_test = np.column_stack((X_fit_test, np.column_stack((A_test, B_test))))
    """

    X_fit = generate_independents(X, n_components=n_components, period=period, lin_comp=lin_comp)    
    
    #X_fit_eval_params = X_fit_test
    
    #if lin_comp and n_components:
    #    X_fit = np.column_stack((X, X_fit))
    #    X_fit_eval_params = np.column_stack((np.zeros(len(X_test)), X_fit_test))
    #    X_fit_test = np.column_stack((X_test, X_fit_test))                              


    #X_fit = sm.add_constant(X_fit, has_constant='add')
    #X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    #X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')
    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':        
        #model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(disp=0)
    elif model_type =='gen_poisson':
        #model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(disp=0)
    elif model_type == 'nb':        
        # https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4
        # https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/#cameron
        # if not alpha:
            
        #     train_model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        #     train_results = train_model.fit()

        #     df_train = pd.DataFrame()
        #     df_train['Y'] = Y
        #     df_train['mu'] = train_results.mu
        #     df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['mu'])**2 - x['Y']) / x['mu'], axis=1)
        #     ols_expr = """AUX_OLS_DEP ~ mu - 1"""
        #     aux_olsr_results = smf.ols(ols_expr, df_train).fit()

        #     alpha=aux_olsr_results.params[0]
        #model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(disp=0)
    else:
        print("Invalid option")
        return

    
    if model_type =='lin':
        Y_fit = results.fittedvalues
    else:
        Y_fit = results.predict(X_fit)
        
    
    if model_type in ['lin', 'poisson', 'nb']:
        statistics = calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp)
        if model_type in ['poisson', 'nb']:
            statistics['count'] = np.sum(Y)                                
    else:
        RSS = sum((Y - Y_fit)**2)
        p = results.llr_pvalue
        statistics = {'p':p, 'RSS':RSS, 'count': np.sum(Y)}
    
    #Y_test = results.predict(X_fit_test)
    X_test = np.linspace(0, 2*period, 10000)
    X_fit_test = generate_independents(X_test, n_components=n_components, period=period, lin_comp=lin_comp, remove_lin_comp = True)
    Y_fit_test = results.predict(X_fit_test)
    rhythm_params = evaluate_rhythm_params(X_test, Y_fit_test, period=period)
    if lin_comp:
        rhythm_params['lin_comp'] = results.params[1]
        CIs = results.conf_int()
        if type(CIs) != np.ndarray:
            rhythm_params['CI(lin_comp)'] = CIs.values[1]
        else:
            rhythm_params['CI(lin_comp)'] = CIs[1]
        rhythm_params['p(lin_comp)'] = results.pvalues[1]
        #print(rhythm_params['p(lin_comp)'])
            
    """
    ###
    # plot
    ###
    """
         
    if plot:

        if plot_measurements:
            min_X = np.min(X)
            max_X = np.max(X)
        else:
            min_X = 0
            max_X = period
                
        X_plot = np.linspace(min_X, max_X, 1000)
        X_plot_fits = generate_independents(X_plot, n_components=n_components, period=period, lin_comp=lin_comp)
        Y_plot = results.predict(X_plot_fits)
        

        ###
        if not color:
            color = 'black'

        if plot_measurements:        
            if not hold:                             
                plt.plot(X,Y, 'o', markersize=1, label = 'data', color=color)
            else:
                plt.plot(X,Y, 'o', markersize=1, color=color)
                
        if not hold:
            plt.plot(X_plot, Y_plot, label = 'fit', color=color)
        else:
            plt.plot(X_plot, Y_plot, label = name, color=color)
        
        # plot measurements
        if plot_measurements:
            if rescale_to_period:
                X = X % period

            if model_type == 'lin':               
                plt.axis([min_X, max_X, 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])
            else:
                plt.axis([min_X, max_X, 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])
        else:
            plt.axis([min_X, max_X, min(Y_plot)*0.9, max(Y_plot)*1.1])
        
        if name: 
            plt.title(name)
        """
        if model_type == 'lin':
            if name: 
                plt.title(name + ', p-value=' + "{0:.5f}".format(statistics['p']))
            else:
                plt.title('p-value=' + "{0:.5f}".format(statistics['p']))
        else:
            if name:
                plt.title(name + ', p-value=' + '{0:.3f}'.format(statistics['p']) + ' (n='+str(statistics['count'])+ ')')            
            else:
                plt.title('p-value=' + '{0:.3f}'.format(statistics['p']) + ' (n='+str(statistics['count'])+ ')')
        """
        if x_label:
            plt.xlabel(x_label)
        else:
            plt.xlabel('Time [h]')
        
        if y_label:
            plt.ylabel(y_label)
        elif model_type == 'lin':
            plt.ylabel('Measurements')
        else:
            plt.ylabel('Count')
        
        # plot confidence intervals
        if plot_margins:
            if model_type == 'lin':
                _, lower, upper = wls_prediction_std(results, exog=X_plot_fits, alpha=0.05)
             
                if color:
                    plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)
                else:
                    plt.fill_between(X_plot, lower, upper, color='#888888', alpha=0.1)
            else: 
                # calculate and draw plots from the combinations of parameters from the  95 % confidence intervals of assessed parameters

                res2 = copy.deepcopy(results)
                params = res2.params
                CIs = results.conf_int()
                if type(CIs) != np.ndarray:
                    CIs = CIs.values
                
                #N = 512
                N = 1024
                
                if n_components == 1:                    
                    N2 = 10
                elif n_components == 2:
                    N2 = 8
                else:                                   
                    N2 = 10 - n_components 
                             
                P = np.zeros((len(params), N2))
                                
                for i, CI in enumerate(CIs):                    
                    P[i,:] = np.linspace(CI[0], CI[1], N2)

                n_param_samples = P.shape[1]**P.shape[0] 
                N = n_param_samples #min(max_samples_CI, n_param_samples)
            
                if n_param_samples < 10**6:
                    params_samples = np.random.choice(n_param_samples, size=N, replace=False)
                else:
                    params_samples = my_random_choice(max_val=n_param_samples, size=N)

                for i,idx in enumerate(params_samples): 
            
                    p = lazy_prod(idx, P)
            
                    res2.initialize(results.model, p)            
                    Y_test_CI = res2.predict(X_plot_fits)

                                                      
                    if plot and plot_margins:
                        if color and color != '#000000':
                            plt.plot(X_plot, Y_test_CI, color=color, alpha=0.05)
                        else:
                            plt.plot(X_plot, Y_test_CI, color='#888888', alpha=0.05)
                            
        


        if not hold:
            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()
            if plot_residuals:
                resid = results.resid
                sm.qqplot(resid)
                plt.title(name)
                if save_to:
                    plt.savefig(save_to+'_resid.pdf', bbox_inches='tight')
                    plt.savefig(save_to+'_resid.png')                
                    plt.close()
                else:
                    plt.show()
            
            if plot_phase:
                per = rhythm_params['period']
                amp = rhythm_params['amplitude']
                phase = rhythm_params['acrophase']
                if save_to:
                    folder = os.path.join(*os.path.split(save_to)[:-1])                                        
                    plot_phases([phase], [amp], [name], period=per, folder=folder)
                else:
                    plot_phases([phase], [amp], [name], period=per)#, plot_measurements=True, measurements=[X,Y])

    if bootstrap:
        eval_params_bootstrap(X, X_fit, X_test, X_fit_test, Y,  model_type = model_type, rhythm_params=rhythm_params, bootstrap_size=bootstrap_size, bootstrap_type=bootstrap_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period=period)
        
    if params_CI:
        eval_params_CI(X_test, X_fit_test, results, rhythm_params, samples_per_param = samples_per_param_CI, max_samples = max_samples_CI, k=len(X), sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period=period)

    if return_model: 
        return results, statistics, rhythm_params, X_test, Y_fit_test, model
    else:    
        return results, statistics, rhythm_params, X_test, Y_fit_test


"""
****************************
* end of fitting functions *
****************************
"""

"""
***********************
* start of assessment *
***********************
"""

# rhythm params
def evaluate_rhythm_params(X,Y, project_acrophase=True, period=0):
    #plt.plot(X,Y)
    #plt.show()    

    m = min(Y)
    M = max(Y)    
    A = M - m
    MESOR = m + A/2
    AMPLITUDE = abs(A/2)
    
    PHASE = 0
    PHASE_LOC = 0
        
    H = M - 0.01*M if M >= 0 else M + 0.01*M
    locs, heights = signal.find_peaks(Y, height = H)
    heights = heights['peak_heights'] 

    
    
    if len(locs) >= 2:
        period2 = X[locs[1]] - X[locs[0]]
        period2 = int(round(period2))
    else:
        period2 = np.nan
    
    if not period:
        period = period2

    if len(locs) >= 1:
       PHASE = X[locs[0]]
       PHASE_LOC = locs[0]

    if period:
        ACROPHASE = phase_to_radians(PHASE, period)
        if project_acrophase:
            ACROPHASE = project_acr(ACROPHASE)
    else:
        ACROPHASE = np.nan
  
    
    # peaks and heights
    #Y = Y[X < 24]
    #X = X[X < 24]
    locs, heights = signal.find_peaks(Y, height = MESOR)
    heights = heights['peak_heights'] 

    peaks = X[locs]
    heights = Y[locs]
    
    idxs1 = peaks <= period
    peaks = peaks[idxs1]
    heights = heights[idxs1]

    Y2 = M - Y    
    locs2, heights2 = signal.find_peaks(Y2, height = MESOR-m)
    heights2 = heights2['peak_heights'] 

    troughs = X[locs2]
    heights2 = Y[locs2]

    idxs2 = troughs <= period
    troughs = troughs[idxs2]
    heights2 = heights2[idxs2]

    # rhythm_params
    return {'period':period, 'amplitude':AMPLITUDE, 'acrophase':ACROPHASE, 'mesor':MESOR, 'peaks': peaks, 'heights': heights, 'troughs': troughs, 'heights2': heights2, 'max_loc': PHASE_LOC, 'period2':period2}
    
def calculate_statistics(X, Y, Y_fit, n_components, period, lin_comp = False):
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean())**2)
    RSS = sum((Y - Y_fit)**2)

    n_params = n_components * 2 + 1
    if lin_comp:
        n_params += 1            
    N = Y.size

    F = (MSS/(n_params - 1)) / (RSS/(N - n_params)) 
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)
    #print("p-value(Cornelissen): {}".format(p))
    
    # statistics of GOF according to Cornelissen (eqs (14) - (15))
    # TODO: ali bi bilo potrebno popraviti za lumicycle - ko je več zaporednih meritev v eni časovni točki?
    #X_periodic = (X % period).astype(int)
    X_periodic = np.round_(X % period,2)                                    
    
    X_unique = np.unique(X_periodic)
    n_T = len(X_unique)
    
    SSPE = 0
    for x in X_unique:
        Y_i_avg = np.mean(Y[X_periodic == x])
        SSPE += sum((Y[X_periodic == x] - Y_i_avg)**2)
    SSLOF = RSS-SSPE
    #print('RSS: ', RSS)
    #print('SSPE: ', SSPE)
    #print('SSLOF: ', SSLOF)
    if lin_comp:
        try:
            F = (SSLOF/(n_T-1-(2*n_components + 1)))/(SSPE/(N-n_T))
            p_reject = 1 - stats.f.cdf(F, n_T-1-(2*n_components + 1), N-n_T)
        except:
            F = np.nan
            p_reject = np.nan    
    else:    
        try:
            F = (SSLOF/(n_T-1-2*n_components))/(SSPE/(N-n_T))
            p_reject = 1 - stats.f.cdf(F, n_T-1-2*n_components, N-n_T)
        except:
            F = np.nan
            p_reject = np.nan
        
    
    # Another measure that describes goodnes of fit
    # How well does the curve describe the data?
    # signal to noise ratio
    # fitted curve: signal
    # noise: 
    stdev_data = np.std(Y, ddof = 1)
    stdev_fit = np.std(Y_fit, ddof = 1)
    SNR = stdev_fit / stdev_data
    
    # Standard Error of residuals, margin of error
    # https://stats.stackexchange.com/questions/57746/what-is-residual-standard-error
    DoF = N - n_params
    resid_SE = np.sqrt(RSS/DoF)
    # https://scientificallysound.org/2017/05/16/independent-t-test-python/
    # https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/margin-of-error/
    critical_value = stats.t.ppf(1-0.025, DoF)
    ME = critical_value * resid_SE
    
    
    return {'p':p, 'p_reject':p_reject, 'SNR':SNR, 'RSS': RSS, 'resid_SE': resid_SE, 'ME': ME}


"""
*********************
* end of assessment *
*********************
"""

"""
*****************************
* start of compare wrappers *
*****************************
"""

# compare pairs using a given number of components and period
# analysis - options (from best to worst) (ADDITIONAL ANALYSIS)
# - bootstrap1: independent bootstrap analysis
# - CI1: independent analysis of confidence intervals of two models
# - bootstrap2: bootstrap analysis of a merged model
# - CI2: analysis of confidence intervals of a merged model
def compare_pairs_limo(df, pairs, n_components = 3, period = 24, folder = "", prefix = "", analysis = "", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):
    
    if analysis not in ("", "CI1", "bootstrap1", "CI2", "bootstrap2"):
        print("Invalid option")
        return

    columns = ['test', 'period',  'n_components', 'p', 'q', 'p params', 'q params', 'p(F test)', 'q(F test)']
    if analysis:
        for param in parameters_to_analyse:
            #if param not in ("amplitude", "acrophase"): # these two are already included
            columns += [f'd_{param}']
            columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
    
    df_results = pd.DataFrame(columns = columns)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
                
    for test1, test2 in pairs: 
        for per in period:
            for n_comps in n_components:                                
                if folder:                                       
                    save_to = os.path.join(folder,prefix + test1 + '-' + test2 + '_per=' + str(per) + '_comps=' + str(n_comps))
                else:
                    save_to = ''
                
                #pvalues, params, results = compare_pair_df_extended(df, test1, test2, n_components = n_comps, period = per, lin_comp = lin_comp, model_type = model_type, alpha=alpha, save_to = save_to, plot_measurements=plot_measurements)
                #p_overall, pvalues, params, _ = compare_pair_df_extended(df, test1, test2, n_components = n_comps, period = per, save_to = save_to, **kwargs)
                
                p_overall, p_params, p_F, _, _, rhythm_params = compare_pair_df_extended(df, test1, test2, n_components = n_comps, period = per, save_to = save_to, additional_analysis = analysis, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, **kwargs)
                
                
                d = {}
                d['test'] = test1 + ' vs. ' + test2
                d['period'] = per
                d['n_components'] = n_comps

                d['d_amplitude'] = rhythm_params['d_amplitude']
                d['d_acrophase'] = rhythm_params['d_acrophase']

                d['p'] = p_overall
                d['p params'] = p_params
                d['p(F test)'] = p_F
      
                if analysis:
                     for param in parameters_to_analyse:
                        d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                        d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                        d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                        d[f'q(d_{param})'] = np.nan           
                    
                #df_results = df_results.append(d, ignore_index=True)
                df_results = df_add_row(df_results, d)
  
    df_results['q'] = multi.multipletests(df_results['p'], method = 'fdr_bh')[1]
    
    df_results['q params'] = multi.multipletests(df_results['p params'], method = 'fdr_bh')[1]
    df_results['q(F test)'] = multi.multipletests(df_results['p(F test)'], method = 'fdr_bh')[1]

    
    if analysis:
        for param in parameters_to_analyse:
            df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]             
        
    return df_results

# compare pairs using the best models as stored in df_best_models
# Basic analysis: fist analysis according to LymoRhyde (Singer:2019). Extended with the extra sum-of-squares F test that compares two nested models
# compare pairs with the presumption that the same model is used in both cases 
# the same model: the same period and the same number of cosinor components
#
# analysis - options (from best to worst)
# - bootstrap1: independent bootstrap analysis
# - CI1: independent analysis of confidence intervals of two models
# - bootstrap2: bootstrap analysis of a merged model
# - CI2: analysis of confidence intervals of a merged model
def compare_pairs_best_models_limo(df, df_best_models, pairs, folder = "", prefix = "", analysis = "", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):

    if analysis not in ("", "CI1", "bootstrap1", "CI2", "bootstrap2"):
        print("Invalid option")
        return

    columns = ['test', 'period1',  'n_components1', 'period2',  'n_components2', 'p', 'q', 'p params', 'q params', 'p(F test)', 'q(F test)']


    if analysis:     
        for param in parameters_to_analyse:
            #if param not in ("amplitude", "acrophase"): # these two are already included
            columns += [f'd_{param}']
            columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
    
    df_results = pd.DataFrame(columns = columns)  
   
    
    for test1, test2 in pairs:
        model1 = df_best_models[df_best_models["test"] == test1].iloc[0]
        model2 = df_best_models[df_best_models["test"] == test2].iloc[0]
    
        n_components1 = model1.n_components
        n_components2 = model2.n_components
    
        period1 = model1.period
        period2 = model2.period


        # if models have different number of components always start with the simpler model    
        # model is simpler if number of components is smaller
        if n_components1 > n_components2:
            test1, test2 = test2, test1
            n_components1, n_components2 = n_components2, n_components1
            period1, period2 = period2, period1
        
        if folder:            
            save_to = os.path.join(folder, prefix + test1 + '-' + test2 + '_per1=' + str(period1) + '_comps1=' + str(n_components1) + '_per1=' + str(period2) + '_comps1=' + str(n_components2))
        else:
            save_to = ''
                
        p_overall, p_params, p_F, params, _, rhythm_params = compare_pair_df_extended(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, save_to = save_to, additional_analysis = analysis, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, **kwargs)
        
        d = {}
        d['test'] = test1 + ' vs. ' + test2
        d['period1'] = period1
        d['n_components1'] = n_components1
        d['period2'] = period2
        d['n_components2'] = n_components2

        d['d_amplitude'] = rhythm_params['d_amplitude']
        d['d_acrophase'] = rhythm_params['d_acrophase']

        d['p'] = p_overall
        d['p params'] = p_params                
        d['p(F test)'] = p_F
                         
        if analysis:
            for param in parameters_to_analyse:
                d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                d[f'q(d_{param})'] = np.nan           
            
            #df_results = df_results.append(d, ignore_index=True)


        #d['CI(d_amplitude)'] = rhythm_params['CI(d_amplitude)']
        #d['p(d_amplitude)'] = rhythm_params['p(d_amplitude)']
        #d['CI(d_acrophase)'] = rhythm_params['CI(d_acrophase)']
        #d['p(d_acrophase)'] = rhythm_params['p(d_acrophase)']
       
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
    
    df_results['q'] = multi.multipletests(df_results['p'], method = 'fdr_bh')[1]
    df_results['q params'] = multi.multipletests(df_results['p params'], method = 'fdr_bh')[1]     
    df_results['q(F test)'] = multi.multipletests(df_results['p(F test)'], method = 'fdr_bh')[1]

    if analysis:
        for param in parameters_to_analyse:
            df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]

    return df_results



# compare pairs using a given number of components and period
# analysis - options (from best to worst)
# - bootstrap: independent bootstrap analysis
# - CI: independent analysis of confidence intervals of two models
# if you want to increase the speed specify df_results_extended in which for all analysed models confidence intervals for amplitude and acrophase are given - result of cosinor.analyse_models
def diff_p_t_test_from_CI(X1, X2, CI1, CI2, DoF, angular = False):
    dX = X2 - X1
    if angular:
        dX = project_acr(dX)

    t = abs(stats.t.ppf(0.05/2,df=DoF)) 
    

    dev1 = (CI1[1] - CI1[0])/2
    dev2 = (CI2[1] - CI2[0])/2
    
    if angular:
        dev1 = abs(project_acr(dev1))
        dev2 = abs(project_acr(dev2))
    else:
        dev1 = abs(dev1)
        dev2 = abs(dev2)
    
    dev = dev1+dev2
    se = (dev1 + dev2)/t
    
    CI = [dX-dev, dX+dev]
    T0 = dX/se
    p_val = 2 * (1 - stats.t.cdf(abs(T0), DoF))

    return dX, p_val, CI



def compare_pairs(df, pairs, n_components = 3, period = 24, analysis = "bootstrap", df_results_extended = pd.DataFrame(columns=["test"]), parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], lin_comp = False, **kwargs): 
    
    if (analysis != "CI") and (analysis != "bootstrap"):
        print("Invalid option")
        return

    columns = ['test', 'period',  'n_components', 'p1', 'p2', 'q1', 'q2']
    
    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'd_{param}']
        columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
    
    df_results = pd.DataFrame(columns = columns)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
                
    for test1, test2 in pairs: 
        for per in period:
            for n_comps in n_components:                                
                d = {}
                d['test'] = test1 + ' vs. ' + test2
                d['period'] = per
                d['n_components'] = n_comps

                single_params = {}
                if (test1 in list(df_results_extended['test'])) and (test2 in list(df_results_extended['test'])):
                    try:
                        res1 = dict(df_results_extended[(df_results_extended['test'] == test1) & (df_results_extended['n_components'] == n_comps) & (df_results_extended['period'] == per)].iloc[0])
                        res2 = dict(df_results_extended[(df_results_extended['test'] == test2) & (df_results_extended['n_components'] == n_comps) & (df_results_extended['period'] == per)].iloc[0])
                       
                        single_params["test1"] = {}
                        single_params["test2"] = {}

                        for param in parameters_to_analyse:
                            single_params["test1"][f'CI({param})'] =  res1[f'CI({param})']
                            single_params["test2"][f'CI({param})'] =  res2[f'CI({param})']                        
                        
                    except:
                        pass
                
                if analysis == "CI":
                    rhythm_params = compare_pair_CI(df, test1, test2, n_components = n_comps, period = per, single_params=single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, lin_comp = lin_comp, **kwargs)                    
                elif analysis == "bootstrap":
                    rhythm_params = compare_pair_bootstrap(df, test1, test2, n_components = n_comps, period = per, single_params=single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, lin_comp = lin_comp, **kwargs)                    

                for param in parameters_to_analyse:
                    d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                    d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                    d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                    d[f'q(d_{param})'] = np.nan        

                d['p1'] = rhythm_params['statistics1']['p']
                d['p2'] = rhythm_params['statistics2']['p']
                
                if lin_comp:
                    rp1 = rhythm_params['rhythm_params1']
                    rp2 = rhythm_params['rhythm_params2']
                    d['d_lin_comp'], d['p(d_lin_comp)'], d['CI(d_lin_comp)'] = diff_p_t_test_from_CI(rp1['lin_comp'], rp2['lin_comp'], rp1['CI(lin_comp)'], rp2['CI(lin_comp)'], rhythm_params['DoF'])


                #df_results = df_results.append(d, ignore_index=True)
                df_results = df_add_row(df_results, d)
  
    df_results['q1'] = multi.multipletests(df_results['p1'], method = 'fdr_bh')[1]
    df_results['q2'] = multi.multipletests(df_results['p2'], method = 'fdr_bh')[1]
    
    for param in parameters_to_analyse:
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]        

    if lin_comp:
        param = "lin_comp"
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]        

    return df_results


# compare pairs using the best models as stored in df_best_models
# each member of a pair uses its own model
# analysis - options (from best to worst)
# - bootstrap: independent bootstrap analysis
# - CI: independent analysis of confidence intervals of two models
# if you want to increase the speed specify df_results_extended in which for all analysed models confidence intervals for amplitude and acrophase are given - result of cosinor.analyse_best_models
def compare_pairs_best_models(df, df_best_models, pairs, analysis = "bootstrap", df_results_extended = pd.DataFrame(columns=["test"]), parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], lin_comp=False, **kwargs):
    if (analysis != "CI") and (analysis != "bootstrap"):
        print("Invalid option")
        return

    columns = ['test', 'period1', 'n_components1', 'period2', 'n_components2', 'p1', 'p2', 'q1', 'q2']
     

    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'd_{param}']
        columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
    
    df_results = pd.DataFrame(columns = columns)

    for test1, test2 in pairs:
        model1 = df_best_models[df_best_models["test"] == test1].iloc[0]
        model2 = df_best_models[df_best_models["test"] == test2].iloc[0]
    
        n_components1 = model1.n_components
        n_components2 = model2.n_components
    
        period1 = model1.period
        period2 = model2.period
       
        d = {}
        d['test'] = test1 + ' vs. ' + test2
        d['period1'] = period1
        d['n_components1'] = n_components1
        d['period2'] = period2        
        d['n_components2'] = n_components2


        single_params = {}
        if (test1 in list(df_results_extended['test'])) and (test2 in list(df_results_extended['test'])):
            try:
                res1 = dict(df_results_extended[(df_results_extended['test'] == test1) & (df_results_extended['n_components'] == n_components1) & (df_results_extended['period'] == period1)].iloc[0])
                res2 = dict(df_results_extended[(df_results_extended['test'] == test2) & (df_results_extended['n_components'] == n_components2) & (df_results_extended['period'] == period2)].iloc[0])
                
                single_params["test1"] = {}
                single_params["test2"] = {}

                for param in parameters_to_analyse:
                    single_params["test1"][f'CI({param})'] =  res1[f'CI({param})']
                    single_params["test2"][f'CI({param})'] =  res2[f'CI({param})']                        
                
            except:
                pass

        
        if analysis == "CI":
            rhythm_params = compare_pair_CI(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, single_params = single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, lin_comp = lin_comp, **kwargs)
        elif analysis == "bootstrap":
            rhythm_params = compare_pair_bootstrap(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, single_params = single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, lin_comp = lin_comp, **kwargs)

        for param in parameters_to_analyse:
            d[f'd_{param}'] =  rhythm_params[f'd_{param}']
            d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
            d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
            d[f'q(d_{param})'] = np.nan        

        d['p1'] = rhythm_params['statistics1']['p']
        d['p2'] = rhythm_params['statistics2']['p']

        if lin_comp:
            rp1 = rhythm_params['rhythm_params1']
            rp2 = rhythm_params['rhythm_params2']
            d['d_lin_comp'], d['p(d_lin_comp)'], d['CI(d_lin_comp)'] = diff_p_t_test_from_CI(rp1['lin_comp'], rp2['lin_comp'], rp1['CI(lin_comp)'], rp2['CI(lin_comp)'], rhythm_params['DoF'])

        
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)


    df_results['q1'] = multi.multipletests(df_results['p1'], method = 'fdr_bh')[1]
    df_results['q2'] = multi.multipletests(df_results['p2'], method = 'fdr_bh')[1]
    
    for param in parameters_to_analyse:
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1] 
    
    if lin_comp:
        param = "lin_comp"
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]        


    return df_results



# compare pairs using a given number of components and period
# analysis - options (from best to worst)
# - CI: independent analysis of confidence intervals of two models
# - permutation: permutation/randomisation test
# if you want to increase the speed specify df_results_extended in which for all analysed models confidence intervals for amplitude and acrophase are given - result of cosinor.analyse_models_population
def compare_pairs_population(df, pairs, n_components = 3, period = 24, folder = "", prefix = "", analysis = "CI", lin_comp= False, model_type = 'lin', df_results_extended = pd.DataFrame(columns=["test"]), parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):
           
    if (analysis != "CI") and (analysis != "permutation"):
        print("Invalid option")
        return
    
    columns = ['test', 'period',  'n_components', 'p1', 'p2', 'q1', 'q2']
    

 
    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'd_{param}']
        if analysis == "CI":
            columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
        else:
            columns += [f'p(d_{param})', f'q(d_{param})'] # permutation test does not assess the confidence intervals

    df_results = pd.DataFrame(columns = columns)

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
                
    for test1, test2 in pairs: 
        for per in period:
            for n_comps in n_components:                                                                
                df_pop1 = df[df.test.str.startswith(test1)] 
                df_pop2 = df[df.test.str.startswith(test2)] 

                _, statistics1, _, rhythm_params1, _ = population_fit(df_pop1, n_components = n_comps, period = per, plot = False, lin_comp = lin_comp, model_type = model_type)
                _, statistics2, _, rhythm_params2, _ = population_fit(df_pop2, n_components = n_comps, period = per, plot = False, lin_comp = lin_comp, model_type = model_type)

               
                d = {}
                d['test'] = test1 + ' vs. ' + test2
                d['period'] = per
                d['n_components'] = n_comps

                d['d_amplitude'] = rhythm_params2['amplitude'] - rhythm_params1['amplitude']
                d['d_acrophase'] = project_acr(rhythm_params2['acrophase'] - rhythm_params1['acrophase'])

                d['p1'] = statistics1['p']
                d['p2'] = statistics2['p']
                

                if analysis == "CI":
                    single_params = {}
                    if (test1 in list(df_results_extended['test'])) and (test2 in list(df_results_extended['test'])):
                        try:
                            res1 = dict(df_results_extended[(df_results_extended['test'] == test1) & (df_results_extended['n_components'] == n_comps) & (df_results_extended['period'] == per)].iloc[0])
                            res2 = dict(df_results_extended[(df_results_extended['test'] == test2) & (df_results_extended['n_components'] == n_comps) & (df_results_extended['period'] == per)].iloc[0])
                            
                            single_params["test1"] = {}
                            single_params["test2"] = {}

                            for param in parameters_to_analyse:
                                single_params["test1"][f'CI({param})'] =  res1[f'CI({param})']
                                single_params["test2"][f'CI({param})'] =  res2[f'CI({param})'] 
                        except:
                            pass

                    rhythm_params = compare_pair_population_CI(df, test1, test2, n_components=n_comps, period=per, lin_comp = lin_comp, model_type = model_type, single_params = single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)

                    for param in parameters_to_analyse:
                        d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                        d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                        d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                        d[f'q(d_{param})'] = np.nan

                elif analysis == "permutation":
                    rhythm_params = permutation_test_population_approx(df, [(test1,test2)], n_components=n_comps, period=per, plot=False, lin_comp = lin_comp, model_type = model_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
                    for param in parameters_to_analyse:
                        d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                        #d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                        d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                        d[f'q(d_{param})'] = np.nan
               
                    
                #df_results = df_results.append(d, ignore_index=True)
                df_results = df_add_row(df_results, d)
  
    df_results['q1'] = multi.multipletests(df_results['p1'], method = 'fdr_bh')[1]
    df_results['q2'] = multi.multipletests(df_results['p2'], method = 'fdr_bh')[1]
    
    for param in parameters_to_analyse:
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]     
        
    return df_results

# compare pairs using the best models as stored in best_models
# analysis - options (from best to worst)
# - CI: independent analysis of confidence intervals of two models
# - permutation: permutation/randomisation test
# if you want to increase the speed specify df_results_extended in which for all analysed models confidence intervals for amplitude and acrophase are given - result of cosinor.analyse_best_models_population
def compare_pairs_best_models_population(df, df_best_models, pairs, folder = "", prefix = "", analysis = "CI",  df_results_extended = pd.DataFrame(columns=["test"]), parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):

    if (analysis != "CI") and (analysis != "permutation"):
        print("Invalid option")
        return

    columns = ['test', 'period1', 'n_components1', 'period2', 'n_components2', 'p1', 'p2', 'q1', 'q2']
    
    
    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'd_{param}']
        if analysis == "CI":
            columns += [f'CI(d_{param})', f'p(d_{param})', f'q(d_{param})']
        else:
            columns += [f'p(d_{param})', f'q(d_{param})'] # permutation test does not assess the confidence intervals
    
    df_results = pd.DataFrame(columns = columns)

    for test1, test2 in pairs:
        model1 = df_best_models[df_best_models["test"] == test1].iloc[0]
        model2 = df_best_models[df_best_models["test"] == test2].iloc[0]
    
        n_components1 = model1.n_components
        n_components2 = model2.n_components
        
        period1 = model1.period
        period2 = model2.period

        p1 = model1.p
        p2 = model2.p

        q1 = model1.q
        q2 = model2.q

        d_amplitude = model2.amplitude - model1.amplitude
        d_acrophase = project_acr(model2.acrophase - model1.acrophase)

        d = {}
        d['test'] = test1 + ' vs. ' + test2
        d['period1'] = period1
        d['n_components1'] = n_components1
        d['period2'] = period2
        d['n_components2'] = n_components2

        d['d_amplitude'] = d_amplitude
        d['d_acrophase'] = d_acrophase
        
        d['p1'] = p1
        d['p2'] = p2
        d['q1'] = q1
        d['q2'] = q2
              
        if analysis == "CI":
            single_params = {}
            if (test1 in list(df_results_extended['test'])) and (test2 in list(df_results_extended['test'])):
                try:
                    res1 = dict(df_results_extended[(df_results_extended['test'] == test1) & (df_results_extended['n_components'] == n_components1) & (df_results_extended['period'] == period1)].iloc[0])
                    res2 = dict(df_results_extended[(df_results_extended['test'] == test2) & (df_results_extended['n_components'] == n_components2) & (df_results_extended['period'] == period2)].iloc[0])
                    
                    single_params["test1"] = {}
                    single_params["test2"] = {}

                    for param in parameters_to_analyse:
                        single_params["test1"][f'CI({param})'] =  res1[f'CI({param})']
                        single_params["test2"][f'CI({param})'] =  res2[f'CI({param})'] 
                except:
                    pass

            rhythm_params = compare_pair_population_CI(df, test1, test2, n_components=n_components1, period=period1, n_components2=n_components2, period2=period2, single_params = single_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)

            for param in parameters_to_analyse:
                d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                d[f'q(d_{param})'] = np.nan

        elif analysis == "permutation":
            rhythm_params = permutation_test_population_approx(df, [(test1,test2)], n_components=n_components1, period=period1, n_components2=n_components2, period2=period2, plot=False, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
            for param in parameters_to_analyse:
                d[f'd_{param}'] =  rhythm_params[f'd_{param}']
                #d[f'CI(d_{param})'] = rhythm_params[f'CI(d_{param})']
                d[f'p(d_{param})'] = rhythm_params[f'p(d_{param})']
                d[f'q(d_{param})'] = np.nan
                
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
    
    for param in parameters_to_analyse:
        df_results[f'q(d_{param})'] = multi.multipletests(df_results[f'p(d_{param})'], method = 'fdr_bh')[1]     

    return df_results

"""
***************************
* end of compare wrappers *
***************************
"""

#def compare_pair_df_extended(df, test1, test2, n_components = 3, period = 24, n_components2 = None, period2 = None, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot=True, plot_measurements=True, plot_residuals=False, plot_margins=True, x_label = '', y_label = '', bootstrap = False, bootstrap_independent = False, bootstrap_type="std", bootstrap_size=1000, params_CI = False, params_CI_independent = False, samples_per_param_CI=5, max_samples_CI = 1000, sampling_type="LHS"):
# additional analysis - options (from best to worst)
# - bootstrap1: independent bootstrap analysis
# - CI1: independent analysis of confidence intervals of two models
# - bootstrap2: bootstrap analysis of a merged model
# - CI2: analysis of confidence intervals of a merged model
def compare_pair_df_extended(df, test1, test2, n_components = 3, period = 24, n_components2 = None, period2 = None, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot=True, plot_measurements=True, plot_residuals=False, plot_margins=True, x_label = '', y_label = '', additional_analysis = "", bootstrap_type="std", bootstrap_size=1000, samples_per_param_CI=5, max_samples_CI = 1000, sampling_type="LHS", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase']):
       
    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1
        
    
    df_pair = df[(df.test == test1) | (df.test == test2)].copy()
    df_pair['h_i'] = 0
    df_pair.loc[df_pair.test == test2, 'h_i'] = 1
    
    
    X = df_pair.x
    Y = df_pair.y
    H_i = df_pair.h_i
    
    """
    ###
    # prepare the independent variables
    ###
    """
    X_i = H_i * X 

    for i in np.arange(n_components1):
        n = i+1

        A = np.sin((X/(period1/n))*np.pi*2)        
        B = np.cos((X/(period1/n))*np.pi*2) 
        if not i:
            X_fit = np.column_stack((A, B))        
        else:
            X_fit = np.column_stack((X_fit, np.column_stack((A, B))))
        
    if non_rhythmic:
        X_fit = np.column_stack((X_fit, H_i))
        idx_params = np.array([-1])
    else:
        for i in np.arange(n_components2):
            n = i+1

            A_i = H_i * np.sin((X/(period2/n))*np.pi*2)        
            B_i = H_i * np.cos((X/(period2/n))*np.pi*2) 
        
               
            X_fit = np.column_stack((X_fit, np.column_stack((A_i, B_i))))
        
        X_fit = np.column_stack((X_fit, H_i))
        
        # idx_params = [3, 4] # n = 1
        # idx_params = [5, 6, 7, 8] # n = 2
        # idx_params = [7, 8, 9, 10, 11, 12] # n = 3
        # idx_params = [9, 10, 11, 12, 13, 14, 15, 16] # n = 4
        
        #strt = 2*n_components + 1
        #stp = strt + 2*n_components + 1

        strt = -2
        stp = strt - 2*n_components2 - 1
        idx_params = np.arange(strt, stp, -1)


           
        
    if lin_comp:
        X_fit = np.column_stack((X_i, X_fit))
        X_fit = np.column_stack((X, X_fit))
        idx_params = np.array(idx_params) + 2                                
    
    X_fit = sm.add_constant(X_fit, has_constant='add')

    """
    ###
    # fit
    ###
    """       
    if model_type == 'lin':
        model = sm.OLS(Y, X_fit)
        results = model.fit()
    elif model_type == 'poisson':
        #model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(disp=0)
    elif model_type =='gen_poisson':
        #model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit)
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(disp=0)
    elif model_type == 'nb':
        
        # if not alpha:
        #     train_model = sm.GLM(Y, X_fit, family=sm.families.Poisson())
        #     train_results = train_model.fit()

        #     df_train = pd.DataFrame()
        #     df_train['Y'] = Y
        #     df_train['mu'] = train_results.mu
        #     df_train['AUX_OLS_DEP'] = df_train.apply(lambda x: ((x['Y'] - x['mu'])**2 - x['Y']) / x['mu'], axis=1)
        #     ols_expr = """AUX_OLS_DEP ~ mu - 1"""
        #     aux_olsr_results = smf.ols(ols_expr, df_train).fit()

        #     alpha=aux_olsr_results.params[0]
        

        # model = sm.GLM(Y, X_fit, family=sm.families.NegativeBinomial(alpha=alpha))
        
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(disp=0)
    else:
        print("Invalid option")
        return    
   
    """
    ###
    # plot
    ###
    """
    
    
    ###
    if plot and plot_measurements:
        plt.plot(df_pair[df_pair.test == test1].x.values, df_pair[df_pair.test == test1].y.values, 'ko', markersize=1, alpha = 0.75)
        plt.plot(df_pair[df_pair.test == test2].x.values, df_pair[df_pair.test == test2].y.values, 'ro', markersize=1, alpha = 0.75)
    #plt.plot(X, results.fittedvalues, label = 'fit')
    
    if model_type =='lin':
        Y_fit = results.fittedvalues        
        p_overall = results.f_pvalue
    else:
        Y_fit = results.predict(X_fit)
        p_overall = results.llr_pvalue
     
    
    X1 = X[H_i == 0]
    #Y_fit1 = Y_fit[H_i == 0]
    #L1 = list(zip(X1,Y_fit1))
    #L1.sort()
    #X1,Y_fit1 = list(zip(*L1))  
    X2 = X[H_i == 1]
    #Y_fit2 = Y_fit[H_i == 1]
    #L2 = list(zip(X2,Y_fit2))
    #L2.sort()
    #X2,Y_fit2 = list(zip(*L2))  
    
    
    
    #plt.plot(X1, Y_fit1, 'k', label = 'fit '+test1)    
    #plt.plot(X2, Y_fit2, 'r', label = 'fit '+test2)    

    ### F-test
    # for nested models
    # using extra-sum-of-squares F test
    # in a similar way as described in CYCLOPS
    # https://www.pnas.org/content/114/20/5312#sec-8
    # https://www.pnas.org/content/pnas/suppl/2017/04/20/1619320114.DCSupplemental/pnas.201619320SI.pdf?targetid=nameddest%3DSTXT

    n_params_full = len(results.params)
    n_params_small = n_params_full - len(idx_params) 
    N = len(Y)

    r_small = fit_me(X, Y, n_components, period, lin_comp=lin_comp, model_type=model_type, alpha=alpha, plot=False, x_label = x_label, y_label = y_label)
    RSS_small = r_small[1]['RSS']
    RSS_full = sum((Y - Y_fit)**2)

    DoF_small = N-n_params_small
    DoF_full = N-n_params_full

    """
    print('RSS_small: ', RSS_small)
    print('RSS_full: ', RSS_full)
    print('n_small, dof: ', n_params_small, DoF_small)
    print('n_full, dof: ', n_params_full, DoF_full)
    """
    p_f = compare_models(RSS_small, RSS_full, DoF_small, DoF_full)

    
    if plot:
         ### plot with higher density   
        n_points = 1000
        max_P = max(period1, period2)
        X_full = np.linspace(min(min(X1),min(X2)), max(max_P, max(max(X1), max(X2))), n_points)
        
        X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components1, period1 = period1, n_components2 = n_components2, period2 = period2, lin_comp= lin_comp)
        
        H_i = X_fit_full[:,-1]
        locs = H_i == 0

        #Y_fit_full = results.predict(X_fit_full)
        #plt.plot(X_full, Y_fit_full[0:n_points], 'k', label = test1)    
        #plt.plot(X_full, Y_fit_full[n_points:], 'r', label = test2)    
        
        Y_fit_full1 = results.predict(X_fit_full[locs])
        Y_fit_full2 = results.predict(X_fit_full[~locs])

        
        plt.plot(X_full, Y_fit_full1, 'k', label = test1)    
        plt.plot(X_full, Y_fit_full2, 'r', label = test2)    
        
        if model_type == 'lin' and plot_margins:
            _, lower, upper = wls_prediction_std(results, exog=X_fit_full[locs], alpha=0.05)
            plt.fill_between(X_full, lower, upper, color='black', alpha=0.1)   
            _, lower, upper = wls_prediction_std(results, exog=X_fit_full[~locs], alpha=0.05)
            plt.fill_between(X_full, lower, upper, color='red', alpha=0.1)

    
        ### end of plot with higher density
        
        
        #p = min(results.pvalues[idx_params])
        #plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p))
        plt.title(test1 + ' vs. ' + test2 + ', p-value=' + "{0:.5f}".format(p_f))
        plt.xlabel('time [h]')
        plt.ylabel('measurements')
        plt.legend()
        
        #fig = plt.gcf()
        #fig.set_size_inches(11,8)
        
        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
        
        if plot_residuals:
            
            resid = results.resid
            sm.qqplot(resid)
            plt.title(test1 + ' vs. ' + test2)
            save_to_resid = save_to.split(".")[0] + '_resid' + save_to.split(".")[1]
            if save_to:
                plt.savefig(save_to_resid)
                plt.close()
            else:
                plt.show()
        

    #p_values = list(results.pvalues[idx_params]) + [p_f]
    pvalues = (results.pvalues)
    
    if type(pvalues) != np.ndarray:
        pvalues = pvalues.values
    p_params = np.nanmin(pvalues[idx_params.astype(int)])
        
    # eval rhythm parameters
    n_points = 1000
    max_P = max(2*period1, 2*period2)
    X_full = np.linspace(0, max_P, n_points)
    X_fit_full = generate_independents_compare(X_full, X_full, n_components1 = n_components1, period1 = period1, n_components2 = n_components2, period2 = period2, lin_comp= lin_comp, remove_lin_comp=True)
    H_i = X_fit_full[:,-1]
    locs = H_i == 0

    Y_fit_full1 = results.predict(X_fit_full[locs])
    Y_fit_full2 = results.predict(X_fit_full[~locs])

    # rhythm_params
    rhythm_params1 = evaluate_rhythm_params(X_full, Y_fit_full1, period=period1)
    rhythm_params2 = evaluate_rhythm_params(X_full, Y_fit_full2, period=period2)
    

    rhythm_params = {'amplitude1': rhythm_params1['amplitude'],
                     'amplitude2': rhythm_params2['amplitude'],
                     'd_amplitude': rhythm_params2['amplitude']-rhythm_params1['amplitude'],
                     'acrophase1': rhythm_params1['acrophase'],
                     'acrophase2': rhythm_params2['acrophase'],
                     'd_acrophase': project_acr(rhythm_params2['acrophase']-rhythm_params1['acrophase']),
                     'mesor1': rhythm_params1['mesor'],
                     'mesor2': rhythm_params2['mesor'],
                     'd_mesor': rhythm_params2['mesor']-rhythm_params1['mesor']}

    
    if additional_analysis == "CI1":
        compare_pair_CI(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2 = period2, samples_per_param_CI = samples_per_param_CI, max_samples_CI = max_samples_CI, sampling_type=sampling_type, rhythm_params=rhythm_params, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular)
    elif additional_analysis == "bootstrap1":
        compare_pair_bootstrap(df, test1, test2, n_components = n_components1, period = period1, n_components2 = n_components2, period2=period2, rhythm_params=rhythm_params, bootstrap_size=bootstrap_size, bootstrap_type=bootstrap_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular)
    elif additional_analysis == "CI2":
        eval_params_diff_CI(X_full, X_fit_full, locs, results, rhythm_params = rhythm_params, samples_per_param = samples_per_param_CI, max_samples = max_samples_CI, k = len(X), sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period1=period1, period2=period2)
    elif additional_analysis == "bootstrap2":
        eval_params_diff_bootstrap(X, X_fit, X_full, X_fit_full, Y, model_type, locs, rhythm_params = rhythm_params, bootstrap_size = bootstrap_size, bootstrap_type = bootstrap_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, period1=period1, period2=period2)    
    elif additional_analysis == "":
        pass
    else:
        print("Invalid option")          

    if additional_analysis:  
        for param in parameters_to_analyse:
            d_param = rhythm_params2[param] - rhythm_params1[param]
            if param in parameters_angular:
                d_param = project_acr(d_param)
                rhythm_params[f'd_{param}'] = d_param    
        
  
    return (p_overall, p_params, p_f, results.params[idx_params], results, rhythm_params)


def plot_df_models(df, df_models, folder ="", **kwargs):
    for row in df_models.iterrows():
        test = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        X, Y = df[df.test == test].x.values, df[df.test == test].y.values   
        
        if folder:            
            save_to = os.path.join(folder,test+'_compnts='+str(n_components) +'_per=' + str(period))
        else:
            save_to = ""
        
        fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = save_to, plot=True, **kwargs)

"""
******************************
* start of analysis wrappers *
******************************
"""
# perform a more detailed analysis of the models that were identified to be the best, interesting... in previous analyses
# analysis - options (from best to worst)
# - bootstrap
# - CI: analysis of confidence intervals of regression coefficients
def analyse_models(df, n_components = 3, period = 24, plot = False, folder = "", analysis = "bootstrap", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], lin_comp = False, **kwargs):
    params_CI = False
    bootstrap = False
    if analysis == "CI":
        params_CI = True
    elif analysis == "bootstrap":
        bootstrap = True
    else:
        print("Invalid option") 
        return
    
    columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject']#, 'amplitude', 'acrophase']
    
    if not lin_comp:
        parameters_to_analyse_ext = parameters_to_analyse
    else:
        parameters_to_analyse_ext = parameters_to_analyse + ['lin_comp']

    for param in parameters_to_analyse_ext:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'{param}']
        columns += [f'CI({param})', f'p({param})', f'q({param})']
            

    df_results_extended = pd.DataFrame(columns = columns)  
    
    save_to = "" # for figures

    

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
                
    for test in df.test.unique():
        for per in period:
            for n_comps in n_components:     
                X, Y = df[df.test == test].x.values, df[df.test == test].y.values   
                
                if plot and folder:
                    save_to = os.path.join(folder,test+'_compnts='+str(n_comps) +'_per=' + str(per))
                    
                _, statistics, rhythm_params, _, _ = fit_me(X, Y, n_components = n_comps, period = per, name = test, save_to = save_to, plot=plot, bootstrap=bootstrap, params_CI = params_CI, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, lin_comp = lin_comp, **kwargs)
                
                #if sparse_output:
                #    row = dict(row[1][['test', 'per', 'n_comps', 'p', 'q', 'p_reject', 'q_reject', 'amplitude', 'acrophase', 'mesor']])
                #else:
                row = {'test': test,
                    'period': per,
                    'n_components': n_comps,
                    'p': statistics['p'],
                    'q': np.nan,
                    'p_reject': statistics['p_reject'], 
                    'q_reject': np.nan, 
                    'amplitude': rhythm_params['amplitude'], 
                    'acrophase': rhythm_params['acrophase']}
                
                for param in parameters_to_analyse_ext:
                    row[f'{param}'] =  rhythm_params[f'{param}']
                    row[f'CI({param})'] = rhythm_params[f'CI({param})']
                    row[f'p({param})'] = rhythm_params[f'p({param})']
                    row[f'q({param})'] = np.nan   

                #df_results_extended = df_results_extended.append(row, ignore_index=True, sort=False)
                df_results_extended = df_add_row(df_results_extended, row)

    df_results_extended['q'] = multi.multipletests(df_results_extended['p'], method = 'fdr_bh')[1]
    df_results_extended['q_reject'] = multi.multipletests(df_results_extended['p_reject'], method = 'fdr_bh')[1]    
    
    for param in parameters_to_analyse_ext:
        df_results_extended[f'q({param})'] = multi.multipletests(df_results_extended[f'p({param})'], method = 'fdr_bh')[1]

    return df_results_extended    

# perform a more detailed analysis of the models that were identified to be the best, interesting... in previous analyses
# analysis - options (from best to worst)
# - bootstrap
# - CI: analysis of confidence intervals of regression coefficients
def analyse_best_models(df, df_models, sparse_output = True, plot = False, folder = "", analysis = "bootstrap", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], lin_comp = False, **kwargs):
    params_CI = False
    bootstrap = False
    if analysis == "CI":
        params_CI = True
    elif analysis == "bootstrap":
        bootstrap = True
    else:
        print("Invalid option") 
        return
    
    columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject']
    if not lin_comp:
        parameters_to_analyse_ext = parameters_to_analyse
    else:
        parameters_to_analyse_ext = parameters_to_analyse + ['lin_comp']

    for param in parameters_to_analyse_ext:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'{param}']
        columns += [f'CI({param})', f'p({param})', f'q({param})']

    df_results_extended = pd.DataFrame(columns = columns)  

    
    if sparse_output:
        df_models = df_models[['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'amplitude', 'acrophase']]

    save_to = "" # for figures

    for row in df_models.iterrows():        

        test = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        X, Y = df[df.test == test].x.values, df[df.test == test].y.values   
        
        if plot and folder:
            save_to = os.path.join(folder,test+'_compnts='+str(n_components) +'_per=' + str(period))
            
        _, _, rhythm_params, _, _ = fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = save_to, plot=plot, bootstrap=bootstrap, params_CI = params_CI, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, lin_comp = lin_comp, **kwargs)
        
        row = dict(row[1])
        
        for param in parameters_to_analyse_ext:
            row[f'{param}'] =  rhythm_params[f'{param}']
            row[f'CI({param})'] = rhythm_params[f'CI({param})']
            row[f'p({param})'] = rhythm_params[f'p({param})']
            row[f'q({param})'] = np.nan   

        #df_results_extended = df_results_extended.append(row, ignore_index=True, sort=False)
        df_results_extended = df_add_row(df_results_extended, row)
        
    for param in parameters_to_analyse_ext:               
        df_results_extended[f'q({param})'] = multi.multipletests(df_results_extended[f'p({param})'], method = 'fdr_bh')[1]

    return df_results_extended    

# perform a more detailed analysis of the models that were identified to be the best, interesting... in previous analyses
# the only option supported is the CI anaylsis: analysis of confidence intervals of regression coefficients
def analyse_models_population(df, n_components = 3, period = 24, plot=False, folder = "", prefix="",  parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):
    columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject']#, 'amplitude', 'acrophase']
    
    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'{param}']
        columns += [f'CI({param})', f'p({param})', f'q({param})']

    df_results_extended = pd.DataFrame(columns = columns) 
    
    
    save_to = "" # for figures
       
    names = np.unique(df.test) 
    names = list(set(list(map(lambda x:x.split('_rep')[0], names))))
    names.sort()

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]

    for name in names:
        for n_comps in n_components:
            for per in period:      
        
                df_pop = df[df.test.str.startswith(name)] 
                    
                if plot and folder:
                    save_to=os.path.join(folder,prefix+name+'_compnts='+str(n_comps) +'_per=' + str(per)) 
            

                _, statistics, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, plot = plot, save_to = save_to, params_CI = True, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, **kwargs)  
                                
                row = {'test': name,
                    'period': per,
                    'n_components': n_comps,
                    'p': statistics['p'],
                    'q': np.nan,
                    'p_reject': statistics['p_reject'], 
                    'q_reject': np.nan, 
                    'amplitude': rhythm_params['amplitude'], 
                    'acrophase': rhythm_params['acrophase']}
                
                for param in parameters_to_analyse:
                    row[f'{param}'] =  rhythm_params[f'{param}']
                    #row[f'mean({param})'] =  rhythm_params[f'mean({param})']
                    row[f'CI({param})'] = rhythm_params[f'CI({param})']
                    row[f'p({param})'] = rhythm_params[f'p({param})']
                    row[f'q({param})'] = np.nan   

                                   
                #df_results_extended = df_results_extended.append(row, ignore_index=True, sort=False)
                df_results_extended = df_add_row(df_results_extended, row)

    df_results_extended['q'] = multi.multipletests(df_results_extended['p'], method = 'fdr_bh')[1]
    df_results_extended['q_reject'] = multi.multipletests(df_results_extended['p_reject'], method = 'fdr_bh')[1]    
    
    for param in parameters_to_analyse:
        df_results_extended[f'q({param})'] = multi.multipletests(df_results_extended[f'p({param})'], method = 'fdr_bh')[1]

    return df_results_extended    



# perform a more detailed analysis of the models that were identified to be the best, interesting... in previous analyses
# the only option supported is the CI anaylsis: analysis of confidence intervals of regression coefficients
def analyse_best_models_population(df, df_models, sparse_output = True, plot=False, folder = "", prefix="", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):
    columns = ['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject']#, 'amplitude', 'acrophase']
    
    if 'lin_comp' in kwargs and kwargs['lin_comp'] and 'lin_comp' not in parameters_to_analyse:
        parameters_to_analyse += ['lin_comp']
    
    for param in parameters_to_analyse:
        #if param not in ("amplitude", "acrophase"): # these two are already included
        columns += [f'{param}']
        columns += [f'CI({param})', f'p({param})', f'q({param})']

    df_results_extended = pd.DataFrame(columns = columns) 

    if sparse_output:
        df_models = df_models[['test', 'period', 'n_components', 'p', 'q', 'p_reject', 'q_reject', 'amplitude', 'acrophase']]

    save_to = "" # for figures

    for row in df_models.iterrows():        

        name = row[1].test
        n_comps = row[1].n_components
        per = row[1].period
        df_pop = df[df.test.str.startswith(name)] 
               
        if plot and folder:
            save_to=os.path.join(folder,prefix+name+'_compnts='+str(n_comps) +'_per=' + str(per)) 

        _, _, _, rhythm_params, _ = population_fit(df_pop, n_components = n_comps, period = per, plot = plot, save_to = save_to, params_CI = True, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, **kwargs)  
                        
        row = dict(row[1])
        
        for param in parameters_to_analyse:
            row[f'{param}'] =  rhythm_params[f'{param}']
            row[f'CI({param})'] = rhythm_params[f'CI({param})']
            row[f'p({param})'] = rhythm_params[f'p({param})']
            row[f'q({param})'] = np.nan   

            
        #df_results_extended = df_results_extended.append(row, ignore_index=True, sort=False)
        df_results_extended = df_add_row(df_results_extended, row)

    for param in parameters_to_analyse:
        df_results_extended[f'q({param})'] = multi.multipletests(df_results_extended[f'p({param})'], method = 'fdr_bh')[1]

    return df_results_extended    



"""
****************************
* end of analysis wrappers *
****************************
"""

    
def plot_tuples_best_models(df, df_best_models, tuples, colors = ['black', 'red'], folder = '', **kwargs):
    
    
    for T in tuples:
        min_x = 1000
        max_x = -1000
        min_y = 1000
        max_y = -1000


        for test, color in zip(T, colors):
            model = df_best_models[df_best_models["test"] == test].iloc[0]
            n_components = model.n_components
            period = model.period
            X, Y = df[df.test == test].x.values, df[df.test == test].y.values  

            min_x = min(min(X), min_x)
            if 'plot_measurements' in kwargs and kwargs['plot_measurements'] == False:
                max_x = max(max(X % period), max_x)
            else:
                max_x = max(max(X), max_x)


            min_y = min(min(Y), min_y)
            max_y = max(max(Y), max_y)

            fit_me(X, Y, n_components = n_components, period = period, name = test, save_to = "", plot_residuals = False, hold=True, color = color, **kwargs)
        
        plt.title(" + ".join(T))
        
                
        plt.axis([min(min_x,0), max_x, 0.9*min_y, 1.1*max_y])


        plt.legend()

        if folder:            
            save_to = os.path.join(folder,"+".join(T)+"_"+str(period)+"_"+str(n_components))    
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
        else:
            plt.show()
        plt.close()

def plot_tuples_best_population(df, df_best_models, tuples, colors = ['black', 'red'], folder = '', **kwargs):
    
    
    for T in tuples:
        min_x = 1000
        max_x = -1000
        min_y = 1000
        max_y = -1000


        for test, color in zip(T, colors):
            model = df_best_models[df_best_models["test"] == test].iloc[0]
            n_components = model.n_components
            period = model.period
            df_pop = df[df.test.str.startswith(test)] 

            X, Y = np.array(df_pop.x), np.array(df_pop.y)  


            min_x = min(min(X), min_x)
            if 'plot_measurements' in kwargs and kwargs['plot_measurements'] == False:
                max_x = max(max(X % period), max_x)
            else:
                max_x = max(max(X), max_x)
            min_y = min(min(Y), min_y)
            max_y = max(max(Y), max_y)

            population_fit(df_pop, n_components = n_components, period = period, save_to = "", hold=True, color = color, **kwargs)
        
        plt.title(" + ".join(T))
        
                
        plt.axis([min(min_x,0), max_x, 0.9*min_y, 1.1*max_y])
        
        plt.legend()

        if folder:            
            save_to = os.path.join(folder,"+".join(T)+"_"+str(period)+"_"+str(n_components))   
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
        else:
            plt.show()
        plt.close()

def plot_tuples_models(df, tuples, n_components = 2, period = 24, colors = ['black', 'red'], folder = '', **kwargs):
    
    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]

    for per in period:
        for n_comps in n_components:


            for T in tuples:
                min_x = 1000
                max_x = -1000
                min_y = 1000
                max_y = -1000


                for test, color in zip(T, colors):
                    X, Y = df[df.test == test].x.values, df[df.test == test].y.values  

                    min_x = min(min(X), min_x)
                    if 'plot_measurements' in kwargs and kwargs['plot_measurements'] == False:
                        max_x = max(max(X % per), max_x)
                    else:
                        max_x = max(max(X), max_x)


                    min_y = min(min(Y), min_y)
                    max_y = max(max(Y), max_y)

                    fit_me(X, Y, n_components = n_comps, period = per, name = test, save_to = "", plot_residuals = False, hold=True, color = color, **kwargs)
                
                plt.title(" + ".join(T))
                
                        
                plt.axis([min(min_x,0), max_x, 0.9*min_y, 1.1*max_y])


                plt.legend()

                if folder:            
                    save_to = os.path.join(folder,"+".join(T)+"_"+str(per)+"_"+str(n_comps))    
                    plt.savefig(save_to+'.png')
                    plt.savefig(save_to+'.pdf')
                else:
                    plt.show()
                plt.close()

def plot_tuples_population(df, tuples, n_components = 2, period = 24, colors = ['black', 'red'], folder = '', **kwargs):
    
    
    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]

    for per in period:
        for n_comps in n_components:


            for T in tuples:
                min_x = 1000
                max_x = -1000
                min_y = 1000
                max_y = -1000


                for test, color in zip(T, colors):
                    df_pop = df[df.test.str.startswith(test)] 

                    X, Y = np.array(df_pop.x), np.array(df_pop.y)  


                    min_x = min(min(X), min_x)
                    if 'plot_measurements' in kwargs and kwargs['plot_measurements'] == False:
                        max_x = max(max(X % per), max_x)
                    else:
                        max_x = max(max(X), max_x)
                    min_y = min(min(Y), min_y)
                    max_y = max(max(Y), max_y)

                    population_fit(df_pop, n_components = n_comps, period = per, save_to = "", hold=True, color = color, **kwargs)
                
                plt.title(" + ".join(T))
                
                        
                plt.axis([min(min_x,0), max_x, 0.9*min_y, 1.1*max_y])
                
                plt.legend()

                if folder:            
                    save_to = os.path.join(folder,"+".join(T)+"_"+str(per)+"_"+str(n_comps))      
                    plt.savefig(save_to+'.png')
                    plt.savefig(save_to+'.pdf')
                else:
                    plt.show()
                plt.close()



def plot_df_models_population(df, df_models, folder="", model_type="lin"):
    for row in df_models.iterrows():
        pop = row[1].test
        n_components = row[1].n_components
        period = row[1].period
        #X, Y = np.array(df[df.test == test].x), np.array(df[df.test == test].y)  
        df_pop = df[df.test.str.startswith(pop)]
        if folder:            
            save_to = os.path.join(folder, pop+'_pop_compnts='+str(n_components) +'_per=' + str(period))
        else:
            save_to = ""
        population_fit(df_pop, n_components = n_components, period = period, model_type = model_type, save_to = save_to)

def compare_models(RSS1, RSS2, DF1, DF2):
    if DF2 < DF1:
        F = ((RSS1 - RSS2)/(DF1 - DF2))/(RSS2/DF2)
        return 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1)/(DF2 - DF1))/(RSS1/DF1)
        return 1 - stats.f.cdf(F, DF2 - DF1, DF1)

def compare_models_LR_test(ll1, ll2, K1, K2):
    #ll1 = m1.llf
    #ll2 = m2.llf 
    LR = -2*(ll1-ll2)
    #print("LR",LR)
    
    #K1 = len(m1.params)
    #K2 = len(m2.params)
    
    DoF = K2-K1
    p = stats.chi2.sf(LR, DoF)
    
    return p



def ct_response(y, mu):
    return ((y-mu)**2 - y) / mu

def ct_test(count, poiss_results):

    mu = poiss_results.mu
    y = count
    ct = ct_response(y, mu)

    ct_data=pd.DataFrame()
    ct_data['ct_resp'] = ct
    ct_data['mu'] = mu
    ct_results = smf.ols('ct_resp ~ mu - 1', ct_data).fit()
    alpha_ci95 = ct_results.conf_int(0.05).loc['mu']
    print('\nC-T dispersion test: alpha = {:5.3f}, 95% CI = ({:5.3f}, {:5.3f})'.format(ct_results.params[0], alpha_ci95.loc[0], alpha_ci95.loc[1]))
    alpha = ct_results.params[0]
    
    return alpha       


def compare_ANOVA(df, pairs, n_components = 3, period = 24):
    # https://pythonfordatascience.org/anova-python/
    # http://www.statistik.si/storitve/statisticne-analize-testi/anova-analiza-variance/
    # https://www.youtube.com/watch?v=-yQb_ZJnFXw
    P = []

    for test1, test2 in pairs: 
        """
        df2 = df[(df['test'] == test1) | (df['test'] == test2)].copy()
        df2['A'] = np.sin((df2['x']/period)*np.pi*2)
        df2['B'] = np.cos((df2['x']/period)*np.pi*2)
        if n_components >= 2:
            df2['C'] = np.sin((df2['x']/(period/2))*np.pi*2)
            df2['D'] = np.cos((df2['x']/(period/2))*np.pi*2)
        if n_components >= 3:
            df2['E'] = np.sin((df2['x']/(period/3))*np.pi*2)
            df2['F'] = np.cos((df2['x']/(period/3))*np.pi*2)
        """
        P.append(stats.f_oneway(df['y'][df['test'] == test1], df['y'][df['test'] == test2]).pvalue)

        #results = smf.ols('y ~ test', data = df[(df['test'] == test1) | (df['test'] == test2)]).fit()
        #print(results.summary())

    return multi.multipletests(P, method = 'fdr_bh')[1]

#https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
#https://pythonfordatascience.org/anova-2-way-n-way/
def compare_ANOVA2(df, pairs):
    P = []
	
    for test1, test2 in pairs:
        data = df[(df['test'] == test1) | (df['test'] == test2)]
        formula = 'y ~ x + test + x:test'
        model = smf.ols(formula, data).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        
        P.append(aov_table['PR(>F)']['x:test'])  
        #P.append(model.pvalues[-1])
	
	
    return multi.multipletests(P, method = 'fdr_bh')[1]
	
#def test_phase(X1, Y1, X2, Y2, phase, period = 0, test1 = '', test2 = ''):
#    X2 -= phase
#    if period:
#        X1 %= period
#        X2 %= period
    


"""
Permutation test - does not work as well as it should. 
Problem: when you move an individual from the first population to 
the second one, rhythmicity is collapsed.

N ... number of permutations (if omitted, all permutations are used)

Procedure:

- for each permutation...
-- build permuted population 1 (pop1_perm) and permuted population 2 (pop2_perm)
-- build a cosinor model for pop1_perm and pop2_perm
-- evaluate rhythmicity params for pop1_perm and pop2_perm
-- evalute differences for rhythmicity params between pop1_perm and pop2_perm
-- add differences to a list
- calculate percentile score of the difference for rhythmicity params between population 1 and population 2 
"""
"""
def permutation_test_population(df, pairs, period = 24, n_components = 2, lin_comp = False, model_type = 'lin', N = None):#, N=10=, permutations=[]):
    
    
    df_results = pd.DataFrame(columns = ['pair', "d_amp", "p_d_amp", "d_acr", "p_d_acr", "d_mesor", "p_d_mesor"], dtype=float)

    for pair in pairs:

        

        df_pop1 = df[df.test.str.startswith(pair[0])] 
        df_pop2 = df[df.test.str.startswith(pair[1])] 

        _, statistics1, _, rhythm_params1, _ = population_fit(df_pop1, n_components = n_components, period = period, lin_comp= lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False)
        _, statistics2, _, rhythm_params2, _ = population_fit(df_pop2, n_components = n_components, period = period, lin_comp= lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False)

        p1, amplitude1, acrophase1, mesor1 = statistics1['p'], rhythm_params1['amplitude'], rhythm_params1['acrophase'], rhythm_params1['mesor']
        p2, amplitude2, acrophase2, mesor2 = statistics2['p'], rhythm_params2['amplitude'], rhythm_params2['acrophase'], rhythm_params2['mesor']

        #if p1 > 0.05 or p2 > 0.05:
        #    print(pair, "rhythmicity in one is not significant")
        #    continue

        d_amp = abs(amplitude1 - amplitude2)
        d_acr = abs(acrophase1 - acrophase2)
        d_mesor = abs(mesor1 - mesor2)
        amps, acrs, mesors = [], [], [] #[d_amp], [d_acr], [d_mesor]

        tests1 = list(df_pop1.test.unique())
        tests2 = list(df_pop2.test.unique())
        #n_pop1 = len(tests1)
        #n_pop2 = len(tests2)

        #tests = np.array(tests1 + tests2)
        
        permutations = generate_permutations_all(tests1, tests2)

        if N:
            permutations = np.array(list(permutations))
            if N < len(permutations):
                idxs = np.random.choice(np.arange(len(permutations)), size=N, replace=False)
                permutations = permutations[idxs]
            else:
                idxs = np.random.choice(np.arange(len(permutations)), size=N, replace=True)  
                permutations = permutations[idxs]


        #print(permutations)

        for perm1, perm2 in permutations:
            df_test1 = df[df.test.isin(perm1)]
            df_test2 = df[df.test.isin(perm2)]

            # could as well only permute the parameters of the models
            _, statistics_test1, _, rhythm_params_test1, _ = population_fit(df_test1, n_components = n_components, period = period, lin_comp = lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False)
            _, statistics_test2, _, rhythm_params_test2, _ = population_fit(df_test2, n_components = n_components, period = period, lin_comp = lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False)

            p_test1, amplitude_test1, acrophase_test1, mesor_test1 = statistics_test1['p'], rhythm_params_test1['amplitude'], rhythm_params_test1['acrophase'], rhythm_params_test1['mesor']
            p_test2, amplitude_test2, acrophase_test2, mesor_test2 = statistics_test2['p'], rhythm_params_test2['amplitude'], rhythm_params_test2['acrophase'], rhythm_params_test2['mesor']

            if p_test1 <= 0.05 and p_test2 <= 0.05:
                d_amp_test = abs(amplitude_test1 - amplitude_test2)
                d_acr_test = abs(acrophase_test1 - acrophase_test2)
                d_mesor_test = abs(mesor_test1 - mesor_test2)
            else:
                d_amp_test, d_acr_test, d_mesor_test = 0, 0, 0

           

            amps.append(d_amp_test)
            acrs.append(d_acr_test)
            mesors.append(d_mesor_test)
        
        p_d_amp = 1 - percentileofscore(amps, d_amp, 'rank')/100
        p_d_acr = 1 - percentileofscore(acrs, d_acr, 'rank')/100
        p_d_mesor = 1 - percentileofscore(mesors, d_mesor, 'rank')/100
        
        d = {"pair": tuple(pair),
             "d_amp": d_amp, 
             "p_d_amp": p_d_amp, 
             "d_acr": d_acr, 
             "p_d_acr": p_d_acr, 
             "d_mesor": d_mesor, 
             "p_d_mesor": p_d_mesor}
        
        df_results = df_results.append(d, ignore_index=True)


    return df_results
"""
"""
generate random permutations of two populations
"""
def generate_permutations(n_pop1, n_pop2, N):
    n = n_pop1 + n_pop2
    permutations = []
    
    for _ in range(N):
        R = np.random.permutation(n)
        permutations.append((R[:n_pop1], R[n_pop1:]))

    return permutations

"""
generate all possible permutations of two populations. Presumption: populations should be of equal sizes.
"""
def generate_permutations_all(pop1, pop2):
    n_pop1 = len(pop1)
    #n_pop2 = len(pop2)

    permutations = set()

    full = set(pop1 + pop2)

    for i in range(1,n_pop1):
        p1 = itertools.combinations(pop1,i)
        p2 = itertools.combinations(pop2,n_pop1-i)

        X = list(itertools.product(p1,p2))

        # flatten
        for i in range(len(X)):
            X[i] = [a for b in X[i] for a in b]

        for x in X:
            x.sort()
            y = list(set(full)-set(x))
            y.sort()
            z = [tuple(x), tuple(y)]
            z.sort()
            permutations.add(tuple(z))


    return(permutations)


"""
* only approximative
* rhythm params should be calculated for each population on the population mean cosinor
* in this case, we evaluate rhythm params as means of rhythm params of each individual 
(only approximately equals mean of rhythm params from the population)

N ... number of permutations (if omitted, all permutations are used)

Procedure:

- for each permutation...
-- build permuted population 1 (pop1_perm) and permuted population 2 (pop2_perm)
-- calculate means of rhythmicity params for pop1_perm and pop2_perm
-- evalute differences for rhythmicity params between pop1_perm and pop2_perm
-- add differences to a list
- calculate percentile score of the difference for rhythmicity params between population 1 and population 2

"""

def permutation_test_population_approx(df, pairs, period = 24, n_components = 2, n_components2 = None, period2 = None, N = None, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):#, N=10=, permutations=[]):
    
    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1

    columns = ['pair']
    for param in parameters_to_analyse:
        columns += [f'd_{param}', f'p(d_{param})']

    df_results = pd.DataFrame(columns = columns, dtype=float)

    for pair in pairs:

        df_pop1 = df[df.test.str.startswith(pair[0])] 
        df_pop2 = df[df.test.str.startswith(pair[1])] 

        _, statistics1, _, _, _, ind_params1= population_fit(df_pop1, n_components = n_components1, period = period1, return_individual_params = True, **kwargs)#lin_comp= lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False, return_individual_params=True)
        _, statistics2, _, _, _, ind_params2 = population_fit(df_pop2, n_components = n_components2, period = period2, return_individual_params = True, **kwargs)#lin_comp= lin_comp, model_type = model_type, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False, return_individual_params=True)

        p1 = statistics1['p']
        p2 = statistics2['p']

        #if p1 > 0.05 or p2 > 0.05:
        #    print(pair, ": rhythmicity in one is not significant", sep="")
        #    continue

        mean_params1 = {}
        mean_params2 = {}
        ind_params_all = {}
        d_params = {}
        d_params_permute = {}
        # equations below only present an approximation
        for param in parameters_to_analyse:        
            if param in parameters_angular:
                mean_params1[param] = project_acr(circmean(ind_params1[param], high = np.pi, low = -np.pi))
                mean_params2[param] = project_acr(circmean(ind_params2[param], high = np.pi, low = -np.pi))
                d_params[param] = project_acr(mean_params2[param] - mean_params1[param])
            else:
                mean_params1[param] = np.mean(ind_params1[param])
                mean_params2[param] = np.mean(ind_params2[param])
                d_params[param] = mean_params2[param] - mean_params1[param]
            
            ind_params_all[param] = np.append(ind_params1[param], ind_params2[param])                  
            d_params_permute[param] = []

       

        n1 = len(list(df_pop1.test.unique()))
        n2 = len(list(df_pop2.test.unique()))


        permutations = np.array(list(generate_permutations_all(list(range(n1)), list(range(n1,n1+n2)))))

        if N:
            if N < len(permutations):
                idxs = np.random.choice(np.arange(len(permutations)), size=N, replace=False)
                permutations = permutations[idxs]
            else:
                idxs = np.random.choice(np.arange(len(permutations)), size=N, replace=True)  
                permutations = permutations[idxs]
        
        for perm1, perm2 in permutations:
            perm1 = np.array(perm1)
            perm2 = np.array(perm2)

            for param in parameters_to_analyse:     
                if param in parameters_angular:
                    test1 = project_acr(circmean(ind_params_all[param][perm1], high = np.pi, low = -np.pi))
                    test2 = project_acr(circmean(ind_params_all[param][perm2], high = np.pi, low = -np.pi))
                    d_test = project_acr(test2 - test1)
                else:
                    test1 = np.mean(ind_params_all[param][perm1])
                    test2 = np.mean(ind_params_all[param][perm2])
                    d_test = test2 - test1

                d_params_permute[param].append(d_test)
        
        p_d = {}
        d = {"pair": tuple(pair)}
        for param in parameters_to_analyse: 
            p_d[param] = 1 - percentileofscore(np.abs(d_params_permute[param]), np.abs(d_params[param]), 'rank')/100
            
            d[f'd_{param}'] = d_params[param]
            d[f'p(d_{param})'] = p_d[param]
        
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
    
    if len(pairs) == 1:
        return d
    else:
        return df_results

# eval parameters using bootstrap
# bootstrap type should be set to either std (CI = X+-1.96*STD(X)) or percentile (CI = [2.5th percentile, 97.5th percentile])
def eval_params_bootstrap(X, X_fit, X_test, X_fit_eval_params, Y, model_type, rhythm_params, bootstrap_size=1000, bootstrap_type='std', t_test=True, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], period=24):       
    # generate and evaluate bootstrap samples
    params_bs = {}
    for param in parameters_to_analyse:
        params_bs[param] = np.zeros(bootstrap_size)
    
    idxs = np.arange(len(X))

    for i in range(bootstrap_size):
        
        idxs_bs = np.random.choice(idxs, len(idxs), replace=True)
        Y_bs, X_fit_bs = Y[idxs_bs], X_fit[idxs_bs]

        if model_type == 'lin':
            model_bs = sm.OLS(Y_bs, X_fit_bs)
            results_bs = model_bs.fit()
            
            ## https://python.hotexamples.com/examples/statsmodels.genmod.generalized_linear_model/GLM/fit_constrained/python-glm-fit_constrained-method-examples.html            
            #model_bs = sm.GLM(Y_bs, X_fit_bs)
            #constr = "const>-1"
            #results_bs = model_bs.fit_constrained(constr)
            


        elif model_type == 'poisson':
            #model_bs = sm.GLM(Y_bs, X_fit_bs, family=sm.families.Poisson())
            model_bs = statsmodels.discrete.discrete_model.Poisson(Y_bs, X_fit_bs)
            results_bs = model_bs.fit(disp=0)
        elif model_type =='gen_poisson':
            #model_bs = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y_bs, X_fit_bs)
            model_bs = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y_bs, X_fit_bs, p=1)
            results_bs = model_bs.fit(disp=0)
        elif model_type == 'nb':
            #model_bs = sm.GLM(Y_bs, X_fit_bs, family=sm.families.NegativeBinomial(alpha=alpha))
            model_bs = statsmodels.discrete.discrete_model.NegativeBinomialP(Y_bs, X_fit_bs, p=1)
            results_bs = model_bs.fit(disp=0)

        #Y_test_bs = results_bs.predict(X_fit_test)        
        Y_eval_params_bs = results_bs.predict(X_fit_eval_params)
        rhythm_params_bs = evaluate_rhythm_params(X_test, Y_eval_params_bs, period=period)
            
        
        """
        if rhythm_params_bs['amplitude'] > np.max(Y_eval_params_bs):
            
        
            print(results_bs.summary())

            plt.plot(X[idxs_bs], Y_bs,'.')
            plt.plot(X_test, Y_eval_params_bs)
            plt.show()
        """

        # remove the fits that exhibit divergence
        for param in parameters_to_analyse:
            if (abs(rhythm_params_bs['amplitude']) > (np.max(Y)-np.min(Y))) or ((rhythm_params_bs['period2']) and (rhythm_params_bs['period2'] < rhythm_params_bs['period2'])):
                params_bs[param][i] = np.nan                         
            else:    
                #plt.plot(X_test, Y_eval_params_bs, alpha=0.5)
                params_bs[param][i] = rhythm_params_bs[param]                 

    #plt.show()
    # analyse bootstrap samples
    DoF = bootstrap_size - len(results_bs.params)
    n_params = len(results_bs.params)
    rhythm_params['DoF'] = DoF    

    for param in parameters_to_analyse:
        if param in parameters_angular:
            angular = True
        else:
            angular = False
    
        sample_bs = params_bs[param]
        mean, p_val, CI = bootstrap_statistics(sample_bs, angular=angular, bootstrap_type = bootstrap_type, t_test= t_test, n_params=n_params)

        rhythm_params[f'{param}_bootstrap'] = mean
        rhythm_params[f'CI({param})'] = CI
        rhythm_params[f'p({param})'] = p_val
    
    return rhythm_params


# eval rhythmicity parameter differences using bootstrap in a combination with limorhyde
# bootstrap type should be set to either std (CI = X+-1.96*STD(X)) or percentile (CI = [2.5th percentile, 97.5th percentile])
def eval_params_diff_bootstrap(X, X_fit, X_full, X_fit_full, Y, model_type, locs, rhythm_params, bootstrap_size, bootstrap_type, t_test=True, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], period1=24, period2=24):    
    params_bs = {}
    for param in parameters_to_analyse:
        params_bs[param] = np.zeros(bootstrap_size)

    idxs = np.arange(len(X.values))

    for i in range(bootstrap_size):
        
        idxs_bs = np.random.choice(idxs, len(idxs), replace=True)
        Y_bs, X_fit_bs  = Y.values[idxs_bs], X_fit[idxs_bs]            

        if model_type == 'lin':                    
            model_bs = sm.OLS(Y_bs, X_fit_bs)
            results_bs = model_bs.fit()
        elif model_type == 'poisson':
            #model_bs = sm.GLM(Y_bs, X_fit_bs, family=sm.families.Poisson())
            model_bs = statsmodels.discrete.discrete_model.Poisson(Y_bs, X_fit_bs)
            results_bs = model_bs.fit(disp=0)
        elif model_type =='gen_poisson':
            #model_bs = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y_bs, X_fit_bs)
            model_bs = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y_bs, X_fit_bs, p=1)
            results_bs = model_bs.fit(disp=0)
        elif model_type == 'nb':
            #model_bs = sm.GLM(Y_bs, X_fit_bs, family=sm.families.NegativeBinomial(alpha=alpha))
            model_bs = statsmodels.discrete.discrete_model.NegativeBinomialP(Y_bs, X_fit_bs, p=1)
            results_bs = model_bs.fit(disp=0)

        Y_fit_full1_bs = results_bs.predict(X_fit_full[locs])
        Y_fit_full2_bs = results_bs.predict(X_fit_full[~locs])

        rhythm_params1_bs = evaluate_rhythm_params(X_full, Y_fit_full1_bs, period=period1)
        rhythm_params2_bs = evaluate_rhythm_params(X_full, Y_fit_full2_bs, period=period2)
    
        for param in parameters_to_analyse:
            params_bs[param][i] = rhythm_params2_bs[param] - rhythm_params1_bs[param]
            if param in parameters_angular:
                params_bs[param][i] = params_bs[param][i]#project_acr(params_bs[param][i])
    
    # analyse bootstrap samples
    DoF = bootstrap_size - len(results_bs.params)
    n_params = len(results_bs.params)
    rhythm_params['DoF'] = DoF

    for param in parameters_to_analyse:
        if param in parameters_angular:
            angular = True
        else:
            angular = False
    
        sample_bs = params_bs[param]
        mean, p_val, CI = bootstrap_statistics(sample_bs, angular=angular, bootstrap_type = bootstrap_type, t_test= t_test, n_params=n_params)

        rhythm_params[f'{param}_bootstrap'] = mean
        rhythm_params[f'CI({param})'] = CI
        rhythm_params[f'p({param})'] = p_val
   
    return rhythm_params


# compare two pairs independently using bootstrap
def compare_pair_bootstrap(df, test1, test2, n_components = 1, period = 24, n_components2 = None, period2 = None, bootstrap_size=1000, bootstrap_type="std", t_test = True, rhythm_params = {}, single_params = {}, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):    

    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1   
    
    X1 = df[(df.test == test1)].x.values
    Y1 = df[(df.test == test1)].y.values
    X2 = df[(df.test == test2)].x.values
    Y2 = df[(df.test == test2)].y.values
    
    if single_params:
        run_bootstrap = False
    else:
        run_bootstrap = True

    
    res1, statistics1, rhythm_params1, _, _ = fit_me(X1, Y1, n_components = n_components1, period = period1, plot = False, bootstrap=run_bootstrap, bootstrap_size=bootstrap_size, bootstrap_type = bootstrap_type, **kwargs)
    res2, statistics2, rhythm_params2, _, _ = fit_me(X2, Y2, n_components = n_components2, period = period2, plot = False, bootstrap=run_bootstrap, bootstrap_size=bootstrap_size, bootstrap_type = bootstrap_type, **kwargs)

    rhythm_params['rhythm_params1'] = rhythm_params1
    rhythm_params['rhythm_params2'] = rhythm_params2

    rhythm_params['statistics1'] = statistics1
    rhythm_params['statistics2'] = statistics2

    p1 = statistics1['p']
    p2 = statistics2['p']

    #if p1 > 0.05 or p2 > 0.05:
    #    print("rhythmicity in one is not significant")
    #    #return

    d_params = {}
    
    for param in parameters_to_analyse:        
        d_params[param] = rhythm_params2[param] - rhythm_params1[param]
        if param in parameters_angular:
            d_params[param] = project_acr(d_params[param])
        
    
    CI1 = {}
    CI2 = {}
    if not single_params:
        for param in parameters_to_analyse:            
            CI1[param] = rhythm_params1[f'CI({param})']
            CI2[param] = rhythm_params2[f'CI({param})']
    else:
        for param in parameters_to_analyse:
            CI1[param] = single_params['test1'][f'CI({param})']
            CI2[param] = single_params['test2'][f'CI({param})']

    # DoF
    #k = len(X1) + len(X2)
    n_params1 = len(res1.params)
    n_params2 = len(res2.params)
    n_params = n_params1 + n_params2
    DoF = 2*bootstrap_size - n_params
    rhythm_params['DoF'] = DoF

    DoF1 = bootstrap_size - n_params1
    DoF2 = bootstrap_size - n_params2

    if t_test:
        n_devs = abs(stats.t.ppf(0.05/2,df=DoF))  
    else:
        n_devs = 1.96

    # statistics
    for param in parameters_to_analyse:        
        angular = True if param in parameters_angular else False
        se_param = get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = t_test, angular=angular, CI_type = bootstrap_type, n1 = bootstrap_size, n2 = bootstrap_size, DoF = DoF) 
        d_param = d_params[param]

        rhythm_params[f'd_{param}'] = d_param

        if param in parameters_angular:
            rhythm_params[f'CI(d_{param})'] =  get_acrophase_CI(d_param, n_devs*se_param)
        else:
            rhythm_params[f'CI(d_{param})'] = [d_param - n_devs*se_param, d_param + n_devs*se_param]        

        if t_test:
            rhythm_params[f'p(d_{param})'] = get_p_t_test(d_param, se_param, DoF)
        else:
            rhythm_params[f'p(d_{param})'] = get_p_z_test(d_param, se_param)    

    return rhythm_params

"""
def eval_from_Y_CI(X, Y, Y_l, Y_u, rhythm_params):
    loc = rhythm_params['max_loc']
    dev_params = {}
    
    m_min = min(Y_l)
    m_max = min(Y_u)
    M_min = max(Y_l)
    M_max = max(Y_u)

    amp_min = M_min - m_max
    amp_max = M_max - m_min
    dev_amp = abs(amp_max - amp_min)/2

    mes_min = m_min + amp_min/2
    mes_max = m_max + amp_max/2
    dev_mes = abs(mes_max - mes_min)/2

    eps = 0.01
    idx = loc
    #print(Y[loc])
    #print("***")

    plt.plot(X,Y)
    plt.plot(X,Y_l)
    plt.plot(X,Y_u)
    plt.show()

    found = True
    while idx < len(Y):
        #print(Y_u[idx])
        if abs(Y_u[idx] - Y[loc]) <= eps:        
            break        
        idx += 1              
    else:
        #print("***")
        idx = loc
        while idx >=0:
            #print(Y_u[idx])
            if abs(Y_u[idx] - Y[loc]) <= eps:
                break
            idx -= 1
        else:
            found = False
            dev_acr = np.pi

    if found:
        loc_max = idx
        dev_phase = abs(X[loc_max] - X[loc])
        dev_acr = abs(project_acr(phase_to_radians(dev_phase, rhythm_params['period'])))
    
    acr_min = rhythm_params['acrophase'] - dev_acr
    acr_max = rhythm_params['acrophase'] + dev_acr

    #print(acr_min, rhythm_params['acrophase'], acr_max)
    #print(acrophase_to_hours(acr_min), acrophase_to_hours(rhythm_params['acrophase']), acrophase_to_hours(acr_max))

    rhythm_params['CI(amplitude)'] = [amp_min, amp_max]
    rhythm_params['CI(acrophase)'] = [acr_min, acr_max]
    rhythm_params['CI(mesor)'] = [mes_min, mes_max]

    dev_params['mesor'] = dev_mes
    dev_params['amplitude'] = dev_amp
    dev_params['acrophase'] = dev_acr

    return dev_params  
""" 


# sample the parameters from the confidence interval, builds a set of models and assesses the rhythmicity parameters confidence intervals   
def eval_params_CI(X_test, X_fit_test, results, rhythm_params, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], samples_per_param=5, max_samples = 1000, t_test=True, k=0, sampling_type="LHS", period=24):
      
    res2 = copy.deepcopy(results)
    params = res2.params
    n_params=len(params)
    DoF = k - n_params
    rhythm_params['DoF'] = DoF

    CIs = results.conf_int()
    if type(CIs) != np.ndarray:
        CIs = CIs.values
                   
    P = np.zeros((len(params), samples_per_param))
    for i, CI in enumerate(CIs):                    
        P[i,:] = np.linspace(CI[0], CI[1], samples_per_param)

    mean_params = {}
    dev_params = {}
    for param in parameters_to_analyse:
        mean_params[param] = rhythm_params[param]
        if param in parameters_angular:
            mean_params[param] = project_acr(mean_params[param])
        dev_params[param] = 0.0
   
    
    if not sampling_type:
        n_param_samples = P.shape[1]**P.shape[0] 
        N = min(max_samples, n_param_samples)
        if n_param_samples < 10**6:
            params_samples = np.random.choice(n_param_samples, size=N, replace=False)
        else:
            params_samples = my_random_choice(max_val=n_param_samples, size=N)
    else:
        params_samples = generate_samples(sampling_type, CIs, max_samples)
        if not params_samples:
            print("Invalid sampling type")
            return 

    for i,idx in enumerate(params_samples):     
        if not sampling_type:
            p = lazy_prod(idx, P)
        else: # if sampling_type is defined (e.g., LHS)
            p = params_samples[i]
    
        res2.initialize(results.model, p)            
        Y_test_CI = res2.predict(X_fit_test)      
    
        rhythm_params_CI = evaluate_rhythm_params(X_test, Y_test_CI, period=period)

        for param in parameters_to_analyse:
            dev_tmp = mean_params[param] - rhythm_params_CI[param]

            if np.isnan(dev_tmp):
                continue

            if param in parameters_angular:                
                dev_tmp = np.abs(project_acr(dev_tmp))
            else:            
                dev_tmp = np.abs(dev_tmp)

            if dev_tmp > dev_params[param]:
                dev_params[param] = dev_tmp
        
        for param in parameters_to_analyse:
            if param in parameters_angular:
                rhythm_params[f'CI({param})'] = get_acrophase_CI(mean_params[param], dev_params[param])
            else:
                rhythm_params[f'CI({param})'] = [mean_params[param] - dev_params[param], mean_params[param] + dev_params[param]]
    """
    else:
        _, Y_l, Y_u = wls_prediction_std(results, exog=X_fit_test, alpha=0.05)
        Y = results.predict(X_fit_test)
        dev_params = eval_from_Y_CI(X_test, Y, Y_l, Y_u, rhythm_params)           
    """

    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF))  
    else:
        t = 1.96

    for param in parameters_to_analyse:
        se_param = dev_params[param]/t   
        if t_test:
            rhythm_params[f'p({param})'] = get_p_t_test(mean_params[param], se_param, DoF)            
        else:
            rhythm_params[f'p({param})'] = get_p_z_test(mean_params[param], se_param)    
  
    return rhythm_params 

# eval rhythmicity parameter differences using parameter confidence intervals and limorhyde
def eval_params_diff_CI(X_full, X_fit_full, locs, results, rhythm_params, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], samples_per_param=5, max_samples=1000, t_test=True, k=0, sampling_type="LHS", period1=24, period2=24):

    res2 = copy.deepcopy(results)
    params = res2.params
    n_params = len(params)
    DoF = k-n_params
    rhythm_params['DoF'] = DoF
    CIs = results.conf_int()
    if type(CIs) != np.ndarray:
        CIs = CIs.values
                
    P = np.zeros((len(params), samples_per_param))
    for i, CI in enumerate(CIs):                    
        P[i,:] = np.linspace(CI[0], CI[1], samples_per_param)

    mean_params = {}
    dev_params = {}
    for param in parameters_to_analyse:
        mean_params[param] = rhythm_params[f'd_{param}']
        if param in parameters_angular:
            mean_params[param] = project_acr(mean_params[param])
        dev_params[param] = 0.0

    if not sampling_type:
        n_param_samples = P.shape[1]**P.shape[0] 
        N = min(max_samples, n_param_samples)
        if n_param_samples < 10**6:
            params_samples = np.random.choice(n_param_samples, size=N, replace=False)
        else:
            params_samples = my_random_choice(max_val=n_param_samples, size=N)
    else:
        params_samples = generate_samples(sampling_type, CIs, max_samples)
        if not params_samples:
            print("Invalid sampling type")
            return 

    for i,idx in enumerate(params_samples):     
        if not sampling_type:
            p = lazy_prod(idx, P)
        else: # if lhs
            p = params_samples[i]
    
        res2.initialize(results.model, p)        

        Y_fit_CI1 = res2.predict(X_fit_full[locs])
        Y_fit_CI2 = res2.predict(X_fit_full[~locs])

        rhythm_params1_CI = evaluate_rhythm_params(X_full, Y_fit_CI1, period=period1)
        rhythm_params2_CI = evaluate_rhythm_params(X_full, Y_fit_CI2, period=period2)

        for param in parameters_to_analyse:            
            d_param = rhythm_params2_CI[param] - rhythm_params1_CI[param]
            dev_tmp = mean_params[param] - d_param

            if np.isnan(dev_tmp):
                continue

            if param in parameters_angular:                
                dev_tmp = np.abs(project_acr(dev_tmp))
            else:            
                dev_tmp = np.abs(dev_tmp)

            if dev_tmp > dev_params[param]:
                dev_params[param] = dev_tmp

    # statistics
    for param in parameters_to_analyse:
        if param in parameters_angular:
            rhythm_params[f'CI(d_{param})'] = get_acrophase_CI(mean_params[param], dev_params[param])
        else:
            rhythm_params[f'CI(d_{param})'] = [mean_params[param] - dev_params[param], mean_params[param] + dev_params[param]]
   
    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF))             
    else:
        t = 1.96

    for param in parameters_to_analyse:
        se_param = dev_params[param]/t   
        if t_test:
            rhythm_params[f'p(d_{param})'] = get_p_t_test(mean_params[param], se_param, DoF)            
        else:
            rhythm_params[f'p(d_{param})'] = get_p_z_test(mean_params[param], se_param) 

    return rhythm_params

# eval parameters using bootstrap
# bootstrap type should be set to either std (CI = X+-1.96*STD(X)) or percentile (CI = [2.5th percentile, 97.5th percentile])
def population_eval_params_bootstrap(X_test, X_fit_eval_params, model, results, model_params, bootstrap_size=100, bootstrap_type='std', t_test=True, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], period=24):           
    
    model_bs = copy.deepcopy(model) # precaution
    results_bs = copy.deepcopy(results) # precaution
        
    # generate and evaluate bootstrap samples
    params_bs = {}
    for param in parameters_to_analyse:
        params_bs[param] = np.zeros(bootstrap_size)
        
    n_models = model_params.shape[0] # number of models
    n_params  = model_params.shape[1] # number of model params

    for i in range(bootstrap_size):

        idxs = np.random.choice(n_models, n_models)

        model_params_sample = model_params[idxs,:] #resampling
                
        params_means = np.mean(model_params_sample, axis=0)
        results_bs.initialize(model_bs, params_means)
                
        Y_eval_params = results_bs.predict(X_fit_eval_params) 
        rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params, period=period)
        

        for param in parameters_to_analyse:
            params_bs[param][i] = rhythm_params[param]


        """
        # remove the fits that exhibit divergence
        for param in parameters_to_analyse:
            if (abs(rhythm_params_bs['amplitude']) > (np.max(Y)-np.min(Y))) or ((rhythm_params_bs['period2']) and (rhythm_params_bs['period2'] < rhythm_params_bs['period2'])):
                params_bs[param][i] = np.nan                         
            else:    
                #plt.plot(X_test, Y_eval_params_bs, alpha=0.5)
                params_bs[param][i] = rhythm_params_bs[param]                 
        """

    #print(params_bs)

    # analyse bootstrap samples
    DoF = bootstrap_size - n_params    
    rhythm_params['DoF'] = DoF    

    for param in parameters_to_analyse:
        if param in parameters_angular:
            angular = True
        else:
            angular = False
    
        sample_bs = params_bs[param]
        mean, p_val, CI = bootstrap_statistics(sample_bs, angular=angular, bootstrap_type = bootstrap_type, t_test= t_test, n_params=n_params)

        rhythm_params[f'{param}_bootstrap'] = mean
        rhythm_params[f'CI({param})'] = CI
        rhythm_params[f'p({param})'] = p_val
    
    return rhythm_params



# sample the parameters from the confidence interval, builds a set of models and assesses the rhythmicity parameters confidence intervals   
def population_eval_params_CI(X_test, X_fit_eval_params, results, statistics_params, rhythm_params, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], samples_per_param=5, max_samples = 1000, t_test = True, k=0, sampling_type="LHS", period=24): 
    res2 = copy.deepcopy(results)
    rhythm_params = copy.deepcopy(rhythm_params)
    params = res2.params    
    DoF = k-1
    rhythm_params['DoF'] = DoF
    CIs = statistics_params['CI']
    CIs = list(zip(*CIs))
                
    P = np.zeros((len(params), samples_per_param))
    for i, CI in enumerate(CIs):                    
        P[i,:] = np.linspace(CI[0], CI[1], samples_per_param)

    mean_params = {}
    dev_params = {}
    for param in parameters_to_analyse:
        if param in parameters_angular:
            mean_params[param] = project_acr(rhythm_params[param])
        else:
            mean_params[param] = rhythm_params[param]

        dev_params[param] = 0.0

    if not sampling_type:
        n_param_samples = P.shape[1]**P.shape[0] 
        N = min(max_samples, n_param_samples)        
        if n_param_samples < 10**6:
            params_samples = np.random.choice(n_param_samples, size=N, replace=False)
        else:
            params_samples = my_random_choice(max_val=n_param_samples, size=N)
    else:         
        params_samples = generate_samples(sampling_type, CIs, max_samples)
        if not params_samples:
            print("Invalid sampling type")
            return 

    for i,idx in enumerate(params_samples):     
        if not sampling_type:
            p = lazy_prod(idx, P)
        else: # if lhs
            p = params_samples[i]  

        res2.initialize(results.model, p)            
        Y_test_CI = res2.predict(X_fit_eval_params)
    
        rhythm_params_CI = evaluate_rhythm_params(X_test, Y_test_CI, period=period)

        for param in parameters_to_analyse:
            dev_tmp = mean_params[param] - rhythm_params_CI[param]
            if np.isnan(dev_tmp):
                continue

            if param in parameters_angular:                
                dev_tmp = np.abs(project_acr(dev_tmp))
            else:            
                dev_tmp = np.abs(dev_tmp)

            if dev_tmp > dev_params[param]:
                dev_params[param] = dev_tmp
            
            
    # statistics
    for param in parameters_to_analyse:
        if param in parameters_angular:
            rhythm_params[f'CI({param})'] = get_acrophase_CI(mean_params[param], dev_params[param])
        else:
            rhythm_params[f'CI({param})'] = [mean_params[param] - dev_params[param], mean_params[param] + dev_params[param]]
        
    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF))  
    else:
        t = 1.96

    for param in parameters_to_analyse:
        se_param = dev_params[param]/t   
        if t_test:
            rhythm_params[f'p({param})'] = get_p_t_test(mean_params[param], se_param, DoF)            
        else:
            rhythm_params[f'p({param})'] = get_p_z_test(mean_params[param], se_param)            
        
    #print(rhythm_params)
    return rhythm_params

# compare two population fit pairs independently
def compare_pair_population_CI(df, test1, test2, n_components = 1, period = 24, n_components2 = None, period2 = None, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], single_params = {}, t_test = True, **kwargs):
      
    rhythm_params = {}

    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1
            
    df_pop1 = df[df.test.str.startswith(test1)] 
    df_pop2 = df[df.test.str.startswith(test2)] 

    if single_params:
        run_params_CI = False # fit_me is called without sampling
    else:
        run_params_CI = True # fit_me is called with sampling

    _, statistics1, _, rhythm_params1, _ = population_fit(df_pop1, n_components = n_components1, period = period1, plot = False,plot_measurements=False, plot_individuals=False, plot_margins=False, params_CI = run_params_CI, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
    _, statistics2, _, rhythm_params2, _ = population_fit(df_pop2, n_components = n_components2, period = period2, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False, params_CI = run_params_CI, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)

    rhythm_params['rhythm_params1'] = rhythm_params1
    rhythm_params['rhythm_params2'] = rhythm_params2

    rhythm_params['statistics1'] = statistics1
    rhythm_params['statistics2'] = statistics2

    p1 = statistics1['p']
    p2 = statistics2['p']

    #if p1 > 0.05 or p2 > 0.05:
    #    print("rhythmicity in one is not significant")
    #    #return

    d_params = {}
    
    for param in parameters_to_analyse:        
        d_params[param] = rhythm_params2[param] - rhythm_params1[param]
        if param in parameters_angular:
            d_params[param] = project_acr(d_params[param])        
    
    CI1 = {}
    CI2 = {}
    if not single_params:
        for param in parameters_to_analyse:            
            CI1[param] = rhythm_params1[f'CI({param})']
            CI2[param] = rhythm_params2[f'CI({param})']
    else:
        for param in parameters_to_analyse:
            CI1[param] = single_params['test1'][f'CI({param})']
            CI2[param] = single_params['test2'][f'CI({param})']
        

    # DoF
    k1 = len(df_pop1.test.unique())
    k2 = len(df_pop2.test.unique()) 
    k = len(df_pop1.test.unique()) + len(df_pop2.test.unique()) 
    DoF = k - 2
    DoF1 = k1 - 1
    DoF2 = k2 - 1
    rhythm_params['DoF'] = DoF

    # statistics
    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF)) 
    else:
        t = 1.96

    for param in parameters_to_analyse:        
        angular = True if param in parameters_angular else False
        se_param = get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = t_test, angular=angular, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF) 
        d_param = d_params[param]

        rhythm_params[f'd_{param}'] = d_param

        if param in parameters_angular:
            rhythm_params[f'CI(d_{param})'] =  get_acrophase_CI(d_param, t*se_param)
        else:
            rhythm_params[f'CI(d_{param})'] = [d_param - t*se_param, d_param + t*se_param]        
  
        if t_test:
            rhythm_params[f'p(d_{param})'] = get_p_t_test(d_param, se_param, DoF)
        else:
            rhythm_params[f'p(d_{param})'] = get_p_z_test(d_param, se_param)
    
    return rhythm_params


# compare two population fit pairs independently - new variant that calculates CI amplitudes, CI acrophases etc. from individual models
def compare_pair_population_CI_new(df, test1, test2, n_components = 1, period = 24, n_components2 = None, period2 = None, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], t_test = True, **kwargs):
      
    rhythm_params = {}

    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1
            
    df_pop1 = df[df.test.str.startswith(test1)] 
    df_pop2 = df[df.test.str.startswith(test2)] 

    _, statistics1, statistics_params1, rhythm_params1, _ = population_fit(df_pop1, n_components = n_components1, period = period1, plot = False,plot_measurements=False, plot_individuals=False, plot_margins=False, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
    _, statistics2, statistics_params2, rhythm_params2, _ = population_fit(df_pop2, n_components = n_components2, period = period2, plot = False, plot_measurements=False, plot_individuals=False, plot_margins=False, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)

    rhythm_params['rhythm_params1'] = rhythm_params1
    rhythm_params['rhythm_params2'] = rhythm_params2

    rhythm_params['statistics1'] = statistics1
    rhythm_params['statistics2'] = statistics2

    #p1 = statistics1['p']
    #p2 = statistics2['p']

    #if p1 > 0.05 or p2 > 0.05:
    #    print("rhythmicity in one is not significant")
    #    #return

    d_params = {}
    
    for param in parameters_to_analyse:        

        rhythm_params1[param] = statistics_params1['ind_params_stats'][f'mean({param})'] #override the joint cosinor model
        rhythm_params2[param] = statistics_params2['ind_params_stats'][f'mean({param})'] #override the joint cosinor model

        d_params[param] = rhythm_params2[param] - rhythm_params1[param]
        if param in parameters_angular:
            d_params[param] = project_acr(d_params[param])        
    
    CI1 = {}
    CI2 = {}
    for param in parameters_to_analyse:            
        CI1[param] = statistics_params1['ind_params_stats'][f'CI({param})']
        CI2[param] = statistics_params2['ind_params_stats'][f'CI({param})']
        

    # DoF
    k1 = len(df_pop1.test.unique())
    k2 = len(df_pop2.test.unique()) 
    k = len(df_pop1.test.unique()) + len(df_pop2.test.unique()) 
    DoF = k - 2
    DoF1 = k1 - 1
    DoF2 = k2 - 1
    rhythm_params['DoF'] = DoF

    # statistics
    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF)) 
    else:
        t = 1.96

    for param in parameters_to_analyse:        
        angular = True if param in parameters_angular else False
        se_param = get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = t_test, angular=angular, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF) 
        d_param = d_params[param]

        rhythm_params[f'd_{param}'] = d_param

        if param in parameters_angular:
            rhythm_params[f'CI(d_{param})'] =  get_acrophase_CI(d_param, t*se_param)
        else:
            rhythm_params[f'CI(d_{param})'] = [d_param - t*se_param, d_param + t*se_param]        
  
        if t_test:
            rhythm_params[f'p(d_{param})'] = get_p_t_test(d_param, se_param, DoF)
        else:
            rhythm_params[f'p(d_{param})'] = get_p_z_test(d_param, se_param)
    
    return rhythm_params


# compare two pairs independently
def compare_pair_CI(df, test1, test2, n_components = 1, period = 24, n_components2 = None, period2 = None, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], samples_per_param_CI=5, max_samples_CI = 1000, t_test = True, sampling_type="LHS", rhythm_params = {}, single_params = {}, **kwargs):    
    
    n_components1 = n_components
    period1 = period
    if not n_components2:
        n_components2 = n_components1
    if not period2:
        period2 = period1   
    
    X1 = df[(df.test == test1)].x.values
    Y1 = df[(df.test == test1)].y.values
    X2 = df[(df.test == test2)].x.values
    Y2 = df[(df.test == test2)].y.values

    if single_params:
        run_params_CI = False # fit_me is called without sampling
    else:
        run_params_CI = True # fit_me is called with sampling
    
    res1, statistics1, rhythm_params1, _, _ = fit_me(X1, Y1, n_components = n_components1, period = period1, plot = False, params_CI = run_params_CI, samples_per_param_CI = samples_per_param_CI, max_samples_CI=max_samples_CI, sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
    res2, statistics2, rhythm_params2, _, _ = fit_me(X2, Y2, n_components = n_components2, period = period2, plot = False, params_CI = run_params_CI, samples_per_param_CI = samples_per_param_CI, max_samples_CI=max_samples_CI, sampling_type=sampling_type, parameters_to_analyse = parameters_to_analyse, parameters_angular = parameters_angular, **kwargs)
    
    rhythm_params['rhythm_params1'] = rhythm_params1
    rhythm_params['rhythm_params2'] = rhythm_params2

    rhythm_params['statistics1'] = statistics1
    rhythm_params['statistics2'] = statistics2

    #p1 = statistics1['p']
    #p2 = statistics2['p']

    #if p1 > 0.05 or p2 > 0.05:
    #    print("rhythmicity in one is not significant")
    #    #return

    d_params = {}
    for param in parameters_to_analyse:        
        d_params[param] = rhythm_params2[param] - rhythm_params1[param]
        if param in parameters_angular:
            d_params[param] = project_acr(d_params[param])
        
    CI1 = {}
    CI2 = {}
    if not single_params:
        for param in parameters_to_analyse:            
            CI1[param] = rhythm_params1[f'CI({param})']
            CI2[param] = rhythm_params2[f'CI({param})']
    else:
        for param in parameters_to_analyse:
            CI1[param] = single_params['test1'][f'CI({param})']
            CI2[param] = single_params['test2'][f'CI({param})']
     

    # DoF
    k1 = len(X1)
    k2 = len(X2)
    k = len(X1) + len(X2)
    n_params = len(res1.params)+len(res2.params)
    n_params1 = len(res1.params)
    n_params2 = len(res2.params)
    DoF = k - n_params
    DoF1 = k1 - n_params1
    DoF2 = k2 - n_params2
    rhythm_params['DoF'] = DoF
    
    # statistics
    if t_test:
        t = abs(stats.t.ppf(0.05/2,df=DoF))          
    else:
        t = 1.96

    for param in parameters_to_analyse:        
        angular = True if param in parameters_angular else False
        se_param = get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = t_test, angular=angular, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF) 
        d_param = d_params[param]

        rhythm_params[f'd_{param}'] = d_param

        if param in parameters_angular:
            rhythm_params[f'CI(d_{param})'] =  get_acrophase_CI(d_param, t*se_param)
        else:
            rhythm_params[f'CI(d_{param})'] = [d_param - t*se_param, d_param + t*se_param]        
  
        if t_test:
            rhythm_params[f'p(d_{param})'] = get_p_t_test(d_param, se_param, DoF)
        else:
            rhythm_params[f'p(d_{param})'] = get_p_z_test(d_param, se_param)

    return rhythm_params

"""
**************************
* other helper functions *
**************************
"""


# returns an idx-th element from the cartesian product of the rows within L
def lazy_prod(idx, L):
    
    p = np.zeros(len(L))
    
    for i,l in enumerate(L):
        p[i] = l[idx % len(l)]
        idx //= len(l)

    return p

# choice n_param_samples values from the interval [0, max_val) without replacements - less memory consumption than np.random.choice
def my_random_choice(max_val, size):
    if max_val < size:
        return []
    
    S = np.zeros(size, dtype=np.int64)
    S[:] = -1
    
    for i in range(size):
        while True:
            r = np.random.randint(0, max_val, dtype=np.int64)
            if r not in S:
                S[i] = r
                break
    return S

# convert phase from time units to angles in radians
def phase_to_radians(phase, period=24):
    phase_rads = (-(phase/period)*2*np.pi) % (2*np.pi)
    if phase_rads > 0:
        phase_rads -= 2*np.pi
    return phase_rads

# convert phase angles in radians to time units
def acrophase_to_hours(acrophase, period=24):
    acrophase = project_acr(acrophase)
    hours = -period * acrophase/(2*np.pi)
    if hours < 0:
        hours += period
    return hours

# project acrophase to the interval [-pi, pi]
def project_acr(acr):
    acr %= (2*np.pi)
    if acr > np.pi:
        acr -= 2*np.pi
    elif acr < -np.pi:
        acr += 2*np.pi       
    return acr

# generate samples from the intervals using lating hypercube sampling and its variants
# intervals define the dimensionality of the space (number of intervals) and lower and upper bounds
# size defines the number of samples to generate
# uses scikit-optimize library
# https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
def generate_samples(sampling_type, intervals, size):
    
    try:
        from skopt.space import Space
        from skopt.sampler import Lhs
    except:
        print("Cannot import skopt")
        return

    space = Space(intervals)

    if sampling_type == "LHS":
        lhs = Lhs(lhs_type="classic", criterion=None)
    elif sampling_type == "centered":
        lhs = Lhs(lhs_type="centered", criterion=None)
    elif sampling_type == "maximin":
        lhs = Lhs(criterion="maximin", iterations=10000)
    elif sampling_type == "ratio":
        lhs = Lhs(criterion="ratio", iterations=10000)
    else:
        return None

    return lhs.generate(space.dimensions, size)

def get_acrophase_CI(mean_acr, dev_acr):
    return [mean_acr-np.abs(dev_acr), mean_acr+np.abs(dev_acr)]

# get standard errors of difference from CIs of two variables
# https://calcworkshop.com/confidence-interval/difference-in-means/    
def get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = True, angular=False, pooled = True, CI_type = "std", n1=0, n2=0, DoF=0, biased=False):
    if angular:
        dev1 = abs(project_acr(CI1[1] - CI1[0]))/2
        dev2 = abs(project_acr(CI2[1] - CI2[0]))/2
    else:
        dev1 = abs(CI1[1] - CI1[0])/2
        dev2 = abs(CI2[1] - CI2[0])/2
    
    if t_test:
        t1 = abs(stats.t.ppf(0.05/2,df=DoF1))
        t2 = abs(stats.t.ppf(0.05/2,df=DoF2))  
    else:
        t1 = 1.96
        t2 = 1.96

    se1 = dev1/t1
    se2 = dev2/t2

    var1, var2 = se1**2, se2**2
    se = (var1 + var2)**0.5  

    if CI_type == "se" and pooled:
        if not DoF:
            DoF = DoF1 + DoF2

        if biased:
            var1 = var1 * (n1)
            var2 = var2 * (n2)
        else:
            var1 = var1 * (n1 + 1)
            var2 = var2 * (n2 + 1)

        F = var1/var2 if var1 > var2 else var2/var1
        t = abs(stats.t.ppf(0.05,df=DoF))
        # pooled variance        
        if F <= t:
            if biased:                
                sp = (((n1-1) * var1 + (n2-1) * var2)/(n1 + n2 - 2))**0.5                
                se = sp * (((1/n1) + (1/n2))**0.5)
            else:
                sp = (((n1-2) * var1 + (n2-2) * var2)/(n1 + n2 - 4))**0.5
                se = sp * (((1/(n1-1)) + (1/(n2-1)))**0.5)
            
    return se

# z-test for parameter significance
def get_p_z_test(X, se_X):
    p_val = 2 * stats.norm.cdf(-np.abs(X/se_X))
    return p_val

# t-test for parameter significance
def get_p_t_test(X, se_X, DoF):
    T0 = X/se_X
    p_val = 2 * (1 - stats.t.cdf(abs(T0), DoF))
    return p_val

# when the number of samples is large, the 0.05 significance level should be decreased
# calculate_significance_level allows you to define a significance level in such cases
# N: number of samples
# kwargs should include:
# n_params: number of params in a model
# OR
# n_components: number of components in a cosinor model 
# optional: lin_comp (bool): additional linear component
# by default the function returns a significance level for the F-test used in a regression
# if return_T is True, the function returns a significance level for the T-test
# for the explanation of background and references see https://davegiles.blogspot.com/2019/10/everythings-significant-when-you-have.html
def calculate_significance_level(N, **kwargs):
    F = np.log(N)
    
    if 'n_params' in kwargs:
        n_params = kwargs['n_params']
    elif 'n_components' in kwargs:
        n_components = kwargs['n_components']
        n_params = n_components * 2 + 1

        if 'lin_comp' in kwargs and kwargs['lin_comp']:
            n_params += 1
    else:
        print('At least n_params or n_components need to be specified.')
        return

    dof1 = n_params-1

    if 'return_T' in kwargs and kwargs['return_T']:
        alpha_T = 1 - stats.t.cdf(np.sqrt(F), dof1)
        return alpha_T
    else:
        dof2 = N - n_params
        alpha_F = 1 - stats.f.cdf(F, dof1, dof2)
        return alpha_F

#########################################################
# calculate confidence intervals and bootstrap p-values #
#########################################################
# return mean, p_val, CI
def bootstrap_statistics(sample_bs, angular=False, bootstrap_type = "std", t_test=True, n_params=0):
        
    sample_bs = sample_bs[~np.isnan(sample_bs)]  

    DoF = len(sample_bs) - n_params #bootstrap_size - len(results_bs.params)    
    if t_test:
        n_devs = abs(stats.t.ppf(0.05/2,df=DoF))  
    else:
        n_devs = 1.96


    # SE or STD?
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1255808/
    # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    if angular:
        mean = project_acr(circmean(sample_bs, high = np.pi, low = -np.pi))
        std = circstd(sample_bs, high = np.pi, low = -np.pi)

        if bootstrap_type == "se":
            se = std/(len(sample_bs)-1)**0.5
            #se_params[param] = std_phase/(len(phases))**0.5
        elif bootstrap_type == "std":
            se = std
        elif bootstrap_type == "percentile":
                # percentiles are used for the calculation of standard error. Confidence intervals are evaluated on the basis of the lower/upper percentile with the largest deviance from the mean
                # https://math.stackexchange.com/questions/1756425/is-it-possible-to-calculate-the-xth-percentile-of-a-collection-of-wind-vectors
                cos_phases = np.cos(sample_bs)
                cos_ci_l = np.percentile(cos_phases,2.5)
                cos_ci_h = np.percentile(cos_phases,97.5)
                ci_l = np.arccos(cos_ci_l)
                ci_h = np.arccos(cos_ci_h)   
                d_phase_l = project_acr(mean - ci_l)
                d_phase_u = project_acr(ci_h - mean)
                dev_phase = np.nanmax([np.abs(d_phase_l), np.abs(d_phase_u)])
                se = dev_phase/n_devs      
    else:                           
        mean = np.nanmean(sample_bs)

        if bootstrap_type == "se":
            se = stats.sem(sample_bs, nan_policy='omit')                
        elif bootstrap_type == "std":
            se = np.nanstd(sample_bs)
        elif bootstrap_type == "percentile":
            # percentiles are used for the calculation of standard error. Confidence intervals are evaluated on the basis of the lower/upper percentile with the largest deviance from the mean
            ci_l = np.percentile(sample_bs,2.5)
            ci_h = np.percentile(sample_bs,97.5)
            dev = np.nanmax([np.abs(mean-ci_l), np.abs(mean-ci_h)])
            se = dev/n_devs

    if angular: 
        CI = get_acrophase_CI(mean, n_devs*se)
    else:
        CI = [mean - n_devs*se, mean + n_devs*se]

    if t_test:    
        p_val = get_p_t_test(mean, se, DoF)
    else:
        p_val = get_p_z_test(mean, se)         
    
    return mean, p_val, CI
        

        