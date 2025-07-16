from CosinorPy import cosinor
import numpy as np
np.seterr(divide='ignore')
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#from scipy.stats import ncf, f, norm # use stats.ncf, stats.f, stats.norm instead
import scipy.stats as stats
import statsmodels.stats.multitest as multi
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
import os

from CosinorPy.helpers import df_add_row

def prepare_df(X1, Y1, X2, Y2):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    Y = np.concatenate((Y1, Y2))    
    X = np.concatenate((X1, X2))    
    H = np.concatenate((H1, H2))

    data = pd.DataFrame()
    
    data['x'] = X
    data['y'] = Y
    data['group'] = H
    
    return data

def plot_pair(data, results, test1='1', test2='2', plot_measurements=True, save_to='', plot_dense=True, period=24, plot_margins = True, legend=True):
   
    X = data.x.values
    Y = data.y.values
    H = data.group
    
    color1 = "black"
    """
    if "LDL" in test2:
        color2 = "#FF0000"
    elif "HDL" in test2:
        color2 = "#0000FF"
    elif "CHOL" in test2:
        color2 = "#00FF00"
    else:
        color2 = "#FF0000"
    """
    color2 = "red"

    if plot_dense:
        X_fit = np.linspace(min(X), max(X), 100)
        H1_fit = np.zeros(len(X_fit))
        H2_fit = np.ones(len(X_fit))
        
        X_fit = np.append(X_fit, X_fit)
        H_fit = np.append(H1_fit, H2_fit)
        
        rrr_fit= np.cos(2*np.pi*X_fit/period)
        sss_fit = np.sin(2*np.pi*X_fit/period)
        
        data = pd.DataFrame()
    
        data['x'] = X_fit
        
        
        data['group'] = H_fit
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        
        Y_fit = results.predict(data)       
        
        if plot_margins:
            X_fit_M = np.linspace(min(X), max(X), 100)
            rrr_fit_M= np.cos(2*np.pi*X_fit_M/period)
            sss_fit_M = np.sin(2*np.pi*X_fit_M/period)
          
            D1 = np.column_stack((H1_fit, rrr_fit_M, sss_fit_M, H1_fit*rrr_fit_M, H1_fit*sss_fit_M))
            D1 = sm.add_constant(D1, has_constant='add')
            _, lower, upper = wls_prediction_std(results, exog=D1, alpha=0.05)
            plt.fill_between(X_fit_M, lower, upper, color=color1, alpha=0.1)   
        
            D2 = np.column_stack((H2_fit, rrr_fit_M, sss_fit_M, H2_fit*rrr_fit_M, H2_fit*sss_fit_M))
            D2 = sm.add_constant(D2, has_constant='add')
            _, lower, upper = wls_prediction_std(results, exog=D2, alpha=0.05)
            plt.fill_between(X_fit_M, lower, upper, color=color2, alpha=0.1)   
        
        
    else:
        H_fit = H
        X_fit = X
        Y_fit = results.fittedvalues    
        
    
    X1 = X[H == 0]
    Y1 = Y[H == 0]
    
    X_fit1 = X_fit[H_fit == 0]
    X_fit2 = X_fit[H_fit == 1]
    Y_fit1 = Y_fit[H_fit == 0]
    
    L1 = list(zip(X_fit1,Y_fit1))
    L1.sort()
    X_fit1,Y_fit1 = list(zip(*L1))  
    
    X2 = X[H == 1]
    Y2 = Y[H == 1]
    Y_fit2 = Y_fit[H_fit == 1]
    
    L2 = list(zip(X_fit2,Y_fit2))
    L2.sort()
    X_fit2,Y_fit2 = list(zip(*L2))  
    
    if plot_measurements:
        plt.plot(X1, Y1, 'o', color=color1, markersize=1)
        plt.plot(X2, Y2, 'o', color=color2, markersize=1)
    plt.plot(X_fit1, Y_fit1, color=color1, label = test1)    
    plt.plot(X_fit2, Y_fit2, color=color2, label = test2)
    plt.xlabel('time [h]')    
    plt.ylabel('measurements')
    
    if legend:
        plt.legend()
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()    
   
def plot_single(data, results, test='', plot_measurements=True, save_to='', plot_dense=True, plot_margins = True, period=24):
   
    X = data.x.values
    Y = data.y.values
    
    #if 'control' in test.lower():
    #    color = "black"
    #else:
    #    color = "blue"
    
    color = "black"
        
    if plot_dense:
        X_fit = np.linspace(min(X), max(X), 1000)
        
        rrr_fit= np.cos(2*np.pi*X_fit/period)
        sss_fit = np.sin(2*np.pi*X_fit/period)
        
        data = pd.DataFrame()
            
        #data['x'] = X_fit
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        #data['Intercept'] = 1
        
        Y_fit = results.predict(data)      
        
        D = np.column_stack((rrr_fit, sss_fit))
        D = sm.add_constant(D, has_constant='add')

        if plot_margins:
            _, lower, upper = wls_prediction_std(results, exog=D, alpha=0.05)
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)   
            #plt.fill_between(X_fit, lower, upper, color='#888888', alpha=0.1)   

        
    else:
        X_fit = X
        Y_fit = results.fittedvalues    
        
    
    L1 = list(zip(X_fit,Y_fit))
    L1.sort()
    X_fit,Y_fit = list(zip(*L1))  
    
    if plot_measurements:
        plt.plot(X, Y, 'o', markersize=1, color=color)
    
    plt.plot(X_fit, Y_fit, color=color, label = test)    
    
    if plot_measurements:
        plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_fit)), 1.1*max(max(Y), max(Y_fit))])
    else:
        plt.axis([min(X_fit), 1.1*max(X), min(Y_fit)*0.9, max(Y_fit)*1.1])
        
    plt.title(test + ', p-value=' + "{0:.5f}".format(results.f_pvalue))
    
    if save_to:
        plt.savefig(save_to+'.pdf')
        plt.savefig(save_to+'.png')
        plt.close()
    else:
        plt.show()    
   

def generate_test_data(x, beta_s, beta_r, period, plot_on = True):
    y = beta_r * np.cos(2*np.pi*x/period) + beta_s *  np.sin(2*np.pi*x/period)
    
    if plot_on:
        plt.plot(x,y)
        plt.show()
    
    #amp, acr = amp_acr(beta_s, beta_r, corrected = True)
    #print("rrr (beta)", beta_r, "sss (gamma):", beta_s)
    #print("amp", amp, "acr", acr)
    
    return x,y
    

def amp_acr(beta_s, beta_r, corrected = True):
    amp = (beta_s**2 + beta_r**2)**(1/2)
    
    #print("rrr (beta)", beta_r, "sss (gamma):", beta_s)
    
    if corrected:
        
        if type(beta_s) != np.ndarray:
            beta_s = np.array([beta_s])
            beta_r = np.array([beta_r])
        
        
        acr = np.zeros(len(beta_s))
        # acr corrected according to cosinor2
        for i in range(len(beta_s)):
            rrr = beta_r[i]
            sss = beta_s[i]
            
            if (rrr > 0) and (sss > 0):
                acr[i] = 0 + (-1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.arctan(sss / rrr)
            elif (rrr > 0) and (sss < 0):
                acr[i] = 2*(-1)*np.pi + (1*np.arctan(np.abs(sss / rrr)))                
                #acr[i] = (1*np.arctan(sss / rrr))
            elif (rrr < 0) and (sss > 0):
                acr[i] = np.pi*(-1) + (1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.pi*(-1) + (1*np.arctan(sss / rrr))
            else:
                acr[i] = np.pi*(-1)+(-1*np.arctan(np.abs(sss / rrr)))
                #acr[i] = np.pi + np.arctan(sss / rrr)
                        
            #acr[i] %= 2*np.pi
            #if acr[i] < 0:                           
            #    acr[i] = acr[i] + 2*np.pi
                #print(acr)
        if type(amp) != np.ndarray:
            acr = acr[0]            
        
        #acr = np.arctan2(beta_s, beta_r)
        #acr = np.arctan2(beta_r, beta_s)
        #acr = np.abs(acr)
                   
           
    else:
        acr = np.arctan(beta_s / beta_r)
        
       
    return amp, acr

def evaluate_cosinor(x, MESOR, amp, acr, period):
    return MESOR + amp*np.cos((2*np.pi*x/period) + acr)

def population_fit_group(df, period = 24, save_folder='', **kwargs):    
    df_cosinor1_fits = pd.DataFrame(columns = ['test', 'p', 'q', 'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 'mesor', 'p(mesor)', 'q(mesor)', 'CI(mesor)', 'acrophase', 'p(acrophase)', 'q(acrophase)', 'CI(acrophase)'], dtype=float)
    
    names = df.test.unique()
    names = list(set(map(lambda x:x.split('_rep')[0], names)))
    names.sort()
    
    for name in names:
        df_pop = df[df.test.str.startswith(f'{name}_rep')] 
        
        if save_folder:
            #save_to = save_folder+"\\pop_"+name+'.pdf'
            save_to = os.path.join(save_folder,"pop_"+name)
        else:
            save_to = ""        

        res = population_fit_cosinor(df_pop, period = period, save_to = save_to, **kwargs)
        
        d = {'test': name, 
         'p':res['p_value'], 
         'amplitude': res['means'][-2],
         'p(amplitude)':res['p_amp'],
         'CI(amplitude)': [res['confint']['amp'][0], res['confint']['amp'][1]],
         'p(mesor)':res['p_mesor'],
         'mesor': res['means'][0],
         'CI(mesor)': [res['confint']['MESOR'][0], res['confint']['MESOR'][1]],
         'acrophase':res['means'][-1],
         'p(acrophase)':res['p_acr'],
         'CI(acrophase)': [res['confint']['acr'][0], res['confint']['acr'][1]],
         'acrophase[h]':cosinor.acrophase_to_hours(res['means'][-1],period)}
    
        #df_cosinor1_fits = df_cosinor1_fits.append(d, ignore_index=True)
        df_cosinor1_fits = df_add_row(df_cosinor1_fits, d)
    
    df_cosinor1_fits['q'] = multi.multipletests(df_cosinor1_fits['p'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(amplitude)'] = multi.multipletests(df_cosinor1_fits['p(amplitude)'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(mesor)'] = multi.multipletests(df_cosinor1_fits['p(mesor)'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(acrophase)'] = multi.multipletests(df_cosinor1_fits['p(acrophase)'], method = 'fdr_bh')[1]


    return df_cosinor1_fits    

def population_test_cosinor_pairs(df, pairs, period = 24, save_folder = "", plot_on=True, **kwargs):
    df_res = pd.DataFrame(columns = ['test', 
                                    'd_amplitude', 
                                    'p(d_amplitude)',
                                    'q(d_amplitude)',                              
                                    'd_acrophase',
                                    'p(d_acrophase)',
                                    'q(d_acrophase)'], dtype=float)

    for pair in pairs:
        df_pop1 = df[df.test.str.startswith(f'{pair[0]}_rep')] 
        df_pop2 = df[df.test.str.startswith(f'{pair[1]}_rep')] 

        res1 = population_fit_cosinor(df_pop1, period = period, plot_on = plot_on, hold_on=True, color="black", **kwargs)      
        res2 = population_fit_cosinor(df_pop2, period = period, plot_on = plot_on, hold_on=True, color="red", **kwargs)
        
        plt.legend()
        plt.title(f"{pair[0]} vs. {pair[1]}")
        
        if save_folder:
            save_to = os.path.join(save_folder,f"{pair[0]}_vs_{pair[1]}")
            plt.savefig(save_to+'.pdf')
            plt.savefig(save_to+'.png')
            plt.close()
        else:
            plt.show()

        res = population_test_cosinor(res1, res2)
        d = {'test': res['test'], 
            'd_amplitude':res['amp']['pop2'] - res['amp']['pop1'], 
            'p(d_amplitude)':res['amp']['p_value'],            
            'd_acrophase':cosinor.project_acr(res['acr']['pop2']-res['acr']['pop1']),
            'p(d_acrophase)':res['acr']['p_value']}

        #df_res = df_res.append(d, ignore_index=True)
        df_res = df_add_row(df_res, d)

    df_res['q(d_amplitude)'] = multi.multipletests(df_res['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_res['q(d_acrophase)'] = multi.multipletests(df_res['p(d_acrophase)'], method = 'fdr_bh')[1]

    return df_res

# compare two indepedent models using confidence intervals
def population_test_cosinor_pairs_independent(df, pairs, period=24, period2=None):
    period1 = period
    if not period2:
        period2 = period

    df_res = pd.DataFrame(columns = ['test', 
                                    'amplitude1', 
                                    'amplitude2',
                                    'd_amplitude',
                                    'p(d_amplitude)',
                                    'q(d_amplitude)',                              
                                    'CI(d_amplitude)',
                                    'acrophase1',
                                    'acrophase2',
                                    'd_acrophase',                                    
                                    'p(d_acrophase)',
                                    'q(d_acrophase)',
                                    'CI(d_acrophase)'], dtype=float)

    for pair in pairs:
        df_pop1 = df[df.test.str.startswith(f'{pair[0]}_rep')] 
        df_pop2 = df[df.test.str.startswith(f'{pair[1]}_rep')] 
        
        k1 = len(df_pop1.test.unique())
        k2 = len(df_pop2.test.unique())
        DoF1 = k1 - 1
        DoF2 = k2 - 1
        DoF = k1 + k2 - 2
        t = abs(stats.t.ppf(0.05/2,df=DoF))   

        res1 = population_fit_cosinor(df_pop1, period = period1, plot_on = False)      
        res2 = population_fit_cosinor(df_pop2, period = period2, plot_on = False)

        #amplitude ('amp')
        param = 'amp'
        amp1, amp2 = res1['means'][-2], res2['means'][-2]
        CI1 = res1['confint'][param][0], res1['confint'][param][1]
        CI2 = res2['confint'][param][0], res2['confint'][param][1]
        se = cosinor.get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = True, angular=False, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF, biased=True) 
        dev = se * t
        d_amp = amp2 - amp1
        CI_amp = [d_amp-dev, d_amp+dev]
        T0 = d_amp/se
        p_val_amp = 2 * (1 - stats.t.cdf(abs(T0), DoF))

         
        #acrophase ('acr')
        param = 'acr'
        acr1, acr2 = res1['means'][-1], res2['means'][-1]
        CI1 = res1['confint'][param][0], res1['confint'][param][1]
        CI2 = res2['confint'][param][0], res2['confint'][param][1]
        se = cosinor.get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = True, angular=True, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF, biased=True) 
        dev = se * t
        d_acr = cosinor.project_acr(acr2-acr1)
        CI_acr = [d_acr - dev, d_acr + dev]
        T0 = d_acr/se
        p_val_acr = 2 * (1 - stats.t.cdf(abs(T0), DoF))

        d = {'test': pair[0] + ' vs ' + pair[1], 
            'amplitude1':amp1,
            'amplitude2':amp2,
            'd_amplitude': d_amp,
            'p(d_amplitude)':p_val_amp,
            'CI(d_amplitude)':CI_amp,
            'acrophase1':acr1,
            'acrophase2':acr2,
            'd_acrophase': d_acr,
            'p(d_acrophase)':p_val_acr,
            'CI(d_acrophase)':CI_acr}

        #df_res = df_res.append(d, ignore_index=True)
        df_res = df_add_row(df_res, d)

    df_res['q(d_amplitude)'] = multi.multipletests(df_res['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_res['q(d_acrophase)'] = multi.multipletests(df_res['p(d_acrophase)'], method = 'fdr_bh')[1]

    return df_res


def population_fit_cosinor(df_pop, period, save_to='', alpha = 0.05, plot_on = True, plot_individuals = True, plot_measurements=True, plot_margins=True, color="black", hold_on=False, plot_residuals=False):
    params = -1
    tests = df_pop.test.unique()
    k = len(tests)
    param_names = ['Intercept', 'rrr (beta)', 'sss (gamma)', 'amp', 'acr']
    cosinors = []

    test_name = tests[0].split('_rep')[0]    
    
    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)

    for i_test, test in enumerate(tests):
        x,y = df_pop[df_pop.test == test].x.values, df_pop[df_pop.test == test].y.values
        fit_results, amp, acr, _ = fit_cosinor(x, y, period = period, save_to=save_to, plot_on = False)
        
        #X_fit = np.linspace(min(x), max(x), 100)
        X_fit = np.linspace(min_X, max_X, 100)
        rrr_fit= np.cos(2*np.pi*X_fit/period)
        sss_fit = np.sin(2*np.pi*X_fit/period)
        
        data = pd.DataFrame()
        data['rrr'] = rrr_fit
        data['sss'] = sss_fit
        Y_fit = fit_results.predict(data).values          
        
        if plot_on and plot_individuals:
            plt.plot(X_fit, Y_fit, color=color, alpha=0.1, label='_Hidden label')          

            if plot_individuals:
                plt.figure(2)
                if i_test == 0:
                    f_ind, ax_ind = plt.subplots(len(tests))
                ax_ind[i_test].plot(X_fit, Y_fit, color=color)
                plt.figure(1)
                
            if plot_residuals: 
                plt.figure(3)
                
                resid = fit_results.resid
                sm.qqplot(resid)
                plt.title(test)
                if save_to:
                    plt.savefig(save_to+f'_resid_{test}'+'.pdf')
                    plt.savefig(save_to+f'_resid_{test}'+'.png')
                    plt.close()
                else:
                    plt.show()
                    
                plt.figure(1)

        
            #M = fit_results.params[0]
            #y_fit = evaluate_cosinor(x, M, amp, acr, period)
            #plt.plot(x, y_fit, 'k')
        if plot_on and plot_measurements:
            plt.plot(x, y, 'o', color=color, markersize=1, label='_Hidden label')
            if plot_individuals:
                plt.figure(2)
                ax_ind[i_test].plot(x, y, 'o', color=color)
                plt.figure(1)
        
        if type(params) == int:
            params = np.append(fit_results.params, np.array([amp, acr]))
            if plot_on and plot_margins:
                Y_fit_all = Y_fit 
        else:
            params = np.vstack([params, np.append(fit_results.params, np.array([amp, acr]))])
            if plot_on and plot_margins:
                Y_fit_all = np.vstack([Y_fit_all, Y_fit])

        cosinors.append(fit_results)
 
    if k > 1:
        means = np.mean(params, axis=0)
    else:
        means=params
    MESOR = means[0]
    beta = means[1]
    gamma = means[2]
    
    amp, acr = amp_acr(gamma, beta)
    means[3] = amp
    means[4] = acr
    
    if k > 1:
        sd = np.std(params, axis=0, ddof = 1)
        sdm = sd[0]
        sdb = sd[1]
        sdy = sd[2]
    
        covby = np.cov(params[:,1], params[:,2])[0,1]          
        denom=(amp**2)*k   
        c22=(((sdb**2)*(beta**2))+(2*covby*beta*gamma)+((sdy**2)*(gamma**2)))/denom
        c23=(((-1*((sdb**2)-(sdy**2)))*(beta*gamma))+(covby*((beta**2)-(gamma**2))))/denom
        c33=(((sdb**2)*(gamma**2))-(2*covby*beta*gamma)+((sdy**2)*(beta**2)))/denom
    
        t=abs(stats.t.ppf(alpha/2, df = k-1))
    
        mesoru=MESOR+((t*sdm)/(k**0.5))
        mesorl=MESOR-((t*sdm)/(k**0.5))
        
        sem = sdm/(k**0.5)    
        T0 = MESOR/sem
        p_mesor = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        #p_mesor = 2 * stats.norm.cdf(-np.abs(MESOR/sem))

        ampu=amp+(t*(c22**0.5))
        ampl=amp-(t*(c22**0.5))
        se_amp = c22**0.5

        T0 = amp/se_amp
        p_amp = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        #p_amp = 2 * stats.norm.cdf(-np.abs(amp/se_amp))
        
        if (ampu > 0 and ampl < 0):
            fiu=np.nan
            fil=np.nan
            p_acr = 1
            print("Warning: Amplitude confidence interval contains zero. Acrophase confidence interval cannot be calculated and was set to NA.")
        else:
            fiu=acr+np.arctan(((c23*(t**2))+((t*np.sqrt(c33))*np.sqrt((amp**2)-(((c22*c33)-(c23**2))*((t**2)/c33)))))/((amp**2)-(c22*(t**2))))
            fil=acr+np.arctan(((c23*(t**2))-((t*np.sqrt(c33))*np.sqrt((amp**2)-(((c22*c33)-(c23**2))*((t**2)/c33)))))/((amp**2)-(c22*(t**2))))

            se_acr = (fiu - acr)/t
            T0 = acr/se_acr
            p_acr = 2 * (1 - stats.t.cdf(abs(T0), k-1))
    else:             
        mesoru=MESOR
        mesorl=MESOR  
        ampu=amp
        ampl=amp
        fiu = acr
        fil = acr
        
        p_acr = 1
        p_amp = 1
        p_mesor = 1
        
    if plot_on:
        #x = np.linspace(min(df_pop.x), max(df_pop.x), 100)
        Y_fit = evaluate_cosinor(X_fit, MESOR, amp, acr, period)
              
        #plt.plot(X_fit, Y_fit, 'black')
        plt.plot(X_fit, Y_fit, color=color, label="_".join(test.split("_")[:-1]))
        plt.title(test_name)


        if plot_margins:
            """
            me = np.linspace(mesoru, mesorl, 5)
            am = np.linspace(ampu, ampl, 5)
            fi = np.linspace(fiu, fil, 5)

            Y = y

            for m in me:
                for a in am:
                    for f in fi:
                        yy = evaluate_cosinor(x, m, a, f, period)
                        Y = np.vstack([Y, yy])
                        #plt.plot(x,yy)


            lower = np.min(Y, axis=0)        
            upper = np.max(Y, axis=0)        

            plt.fill_between(x, lower, upper, color='black', alpha=0.1)  
            """
            if k <= 1:                
                _, lower, upper = wls_prediction_std(fit_results, exog=sm.add_constant(data, has_constant='add'), alpha=0.05)        
            else:
                var_Y = np.var(Y_fit_all, axis=0, ddof = k-1)
                sd_Y = var_Y**0.5
                lower = Y_fit - ((t*sd_Y)/(k**0.5)) # biased se as above
                upper = Y_fit + ((t*sd_Y)/(k**0.5)) # biased se as above                
            plt.fill_between(X_fit, lower, upper, color=color, alpha=0.1)

            
        if not hold_on:
            if save_to:
                plt.savefig(save_to+'.pdf')
                plt.savefig(save_to+'.png')
                plt.close()
            else:  
                plt.show()

        if plot_individuals:
           
            plt.figure(2)
            
            f_ind.suptitle(test_name)
            if save_to:
                f_ind.savefig(save_to+'_ind.pdf')
                f_ind.savefig(save_to+'_ind.png')
                plt.close(f_ind)
            else:                  
                plt.show() 
            
            if hold_on:
                plt.figure(1)

            
  
  
    #print("acr", acr)
    #print("ACR (h): ", -period * acr/(2*np.pi))
    
    confint = {'amp':(ampl, ampu),
               'acr':(fil, fiu),
               'MESOR':(mesorl, mesoru)}
  
    # calculate the overall p-value
    if k > 1:
        betas= params[:,1]
        gammas= params[:,2]
        r = np.corrcoef(betas,gammas)[0,1]
        frac1=(k*(k-2))/(2*(k-1))
        frac2=1/(1-r**2)
        frac3=beta**2/sdb**2
        frac4=(beta*gamma)/(sdb*sdy)
        frac5=gamma**2/sdy**2
        brack=frac3-(2*r*frac4)+frac5
        Fvalue=frac1*frac2*brack
        df2 = k-2
        p_value= 1 - stats.f.cdf(Fvalue, 2, df2)
        #print(p_value)
    
    else:
        p_value = np.nan
     
    return {'test': test_name, 'names':param_names, 'values':params, 'means':means, 'confint':confint, 'p_value':p_value, 'p_mesor':p_mesor, 'p_amp':p_amp, 'p_acr':p_acr}




def population_test_cosinor(pop1, pop2):
    k1 = pop1['values'].shape[0]
    k2 = pop2['values'].shape[0]
    K = k1 + k2
    params1 = pop1['values']
    params2 = pop2['values']
    means1 = pop1['means']
    means2 = pop2['means']
    
    mesors1 = params1[:,0]
    mesor1 = means1[0]
    mesors2 = params2[:,0]
    mesor2 = means2[0]
    betas1 = params1[:,1]
    #beta1 = means1[1]
    betas2 = params2[:,1]
    #beta2 = means2[1]
    gammas1 = params1[:,2]
    #gamma1 = means1[2]
    gammas2 = params2[:,2]
    #gamma2 = means2[2]
    #amps1 = params1[:,3]
    amp1 = means1[3]
    #amps2 = params2[:,3]
    amp2 = means2[3]
    #acrs1 = params1[:,4]
    acr1 = means1[4]
    #acrs2 = params2[:,4]
    acr2 = means2[4]
    
    M = (k1*mesor1 + k2*mesor2)/K
    A = (k1*amp1 + k2*amp2)/K
    FI = (k1*acr1 + k2*acr2)/K
    #BETA = (k1*beta1 + k2*beta2)/K
    #GAMMA = (k1*gamma1 + k2*gamma2)/K
    TM = (k1 * (mesor1 - M)**2) + (k2 * (mesor2 - M)**2)

    
    tann=((k1*(amp1**2))*np.sin(2*acr1))+((k2*(amp2**2))*np.sin(2*acr2))
    tand=((k1*(amp1**2))*np.cos(2*acr1))+((k2*(amp2**2))*np.cos(2*acr2))
    if tand > 0:
        twofi = np.arctan(tann/tand)
    else:
        twofi = np.arctan(tann/tand) + np.pi
    
    FITILDE = twofi/2
    varm1 = np.var(mesors1, ddof = 1)
    varm2 = np.var(mesors2, ddof = 1)
    varb1 = np.var(betas1, ddof = 1)
    varb2 = np.var(betas2, ddof = 1)
    vary1 = np.var(gammas1, ddof = 1)
    vary2 = np.var(gammas2, ddof = 1)
    
    covby1 = np.cov(betas1, gammas1)[0,1]          
    covby2 = np.cov(betas2, gammas2)[0,1]          
    
    varm=(((k1-1)*varm1)/(K-2))+(((k2-1)*varm2)/(K-2))
    varb=(((k1-1)*varb1)/(K-2))+(((k2-1)*varb2)/(K-2))
    vary=(((k1-1)*vary1)/(K-2))+(((k2-1)*vary2)/(K-2))
    covby=(((k1-1)*covby1)/(K-2))+(((k2-1)*covby2)/(K-2))
    FM = TM/varm
    acrn=(amp1**2+((np.sin(acr1-FITILDE))**2)) + (amp2**2+((np.sin(acr2-FITILDE))**2))
    acrd1=varb*((np.sin(FITILDE))**2)
    acrd2=2*covby*np.cos(FITILDE)*np.sin(FITILDE)
    acrd3=vary*((np.cos(FITILDE))**2)
    acrd=acrd1-acrd2+acrd3
    FFI=acrn/acrd
    ampn=((amp1-A)**2) + ((amp2-A)**2)
    ampd1=varb*((np.cos(FI))**2)
    ampd2=2*covby*np.cos(FI)*np.sin(FI)
    ampd3=vary*((np.sin(FI))**2)
    ampd=ampd1-ampd2+ampd3
    FA=ampn/ampd
    df1=1
    df2=K-2
    PM = 1 - stats.f.cdf(FM, df1, df2)
    PA = 1 - stats.f.cdf(FA, df1, df2)
    PFI = 1 - stats.f.cdf(FFI, df1, df2)
    
    if PFI < 0.05:
        print("Results of population amplitude difference test are not reliable due to different acrophases.")
    
    return {'test': pop1['test'] + ' vs ' +pop2['test'], 
            'mesor':{'pop1':mesor1, 'pop2':mesor2, 'p_value':PM},
            'amp':{'pop1':amp1, 'pop2':amp2, 'p_value':PA},
            'acr':{'pop1':acr1, 'pop2':acr2, 'p_value':PFI}}
            

def fit_group(df, period = 24, save_folder='', plot_on=True):
    df_cosinor1_fits = pd.DataFrame(columns = ['test', 'period','p', 'q', 'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 'acrophase', 'p(acrophase)','q(acrophase)', 'CI(acrophase)'], dtype=float)
    
    if (type(period) == int) or (type(period)==float):
        period = [period] 

    for test in df.test.unique():
        x, y = df[df.test == test].x.values, df[df.test == test].y.values
        for per in period:
            if save_folder:
                #fit_results, amp, acr, statistics = fit_cosinor(x, y, per, test=test, save_to=save_folder+"\\"+test+"_"+str(per)+".pdf",  plot_on = plot_on)
                fit_results, amp, acr, statistics = fit_cosinor(x, y, per, test=test, save_to=os.path.join(save_folder,test+"_"+str(per)),  plot_on = plot_on)
            else:
                fit_results, amp, acr, statistics = fit_cosinor(x, y, per, test=test, plot_on = plot_on)
            #if acr <0:
            #    acr += 2 * np.pi
        
            d = {'test': test, 
            'p':fit_results.f_pvalue, 
            'amplitude': amp,
            'period': per,
            'p(amplitude)': statistics['p-values'][1], 
            'CI(amplitude)': [statistics['CI'][0][1], statistics['CI'][1][1]],
            'acrophase':acr,            
            'p(acrophase)': statistics['p-values'][2],
            'CI(acrophase)': [statistics['CI'][0][2], statistics['CI'][1][2]],
            'acrophase[h]':cosinor.acrophase_to_hours(acr,per)}
        
            #df_cosinor1_fits = df_cosinor1_fits.append(d, ignore_index=True)
            df_cosinor1_fits = df_add_row(df_cosinor1_fits, d)
        
    df_cosinor1_fits['q'] = multi.multipletests(df_cosinor1_fits['p'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(amplitude)'] = multi.multipletests(df_cosinor1_fits['p(amplitude)'], method = 'fdr_bh')[1]
    df_cosinor1_fits['q(acrophase)'] = multi.multipletests(df_cosinor1_fits['p(acrophase)'], method = 'fdr_bh')[1]

    return df_cosinor1_fits    
    

def fit_cosinor(X, Y, period, test='', save_to = '', plot_on = True):
         
    data = pd.DataFrame()
    data['x'] = X
    data['y'] = Y
     
    #fit_results, amp, acr = fit_cosinor_df(data, period)
    
    fit_results, amp, acr, statistics = test_cosinor_single(data, period)
    if plot_on:
        plot_single(data, fit_results, test=test, plot_measurements=True, save_to=save_to, plot_dense=True, period=period)       
    
    return fit_results, amp, acr, statistics

# perform a comparison using a joint model
def test_cosinor_pairs(df, pairs, period = 24, folder = '', prefix='', plot_measurements=True, legend=True, df_best_models = -1):
    
    df_results = pd.DataFrame(columns = ['test',
                                         'period',
                                         'p', 
                                         'q', 
                                         'amplitude1', 
                                         'p(amplitude1)', 
                                         'q(amplitude1)', 
                                         'amplitude2', 
                                         'p(amplitude2)', 
                                         'q(amplitude2)', 
                                         'd_amplitude', 
                                         'p(d_amplitude)', 
                                         'q(d_amplitude)',
                                         'CI(d_amplitde)',
                                         'acrophase1', 
                                         'p(acrophase1)',
                                         'q(acrophase1)',
                                         'acrophase2',
                                         'p(acrophase2)',
                                         'q(acrophase2)',
                                         'd_acrophase',
                                         'p(d_acrophase)',
                                         'q(d_acrophase)',
                                         'CI(d_acrophase)'], dtype=float)
    
    for test1, test2 in pairs:    
        if folder:
            #save_to = folder +'\\'+prefix+ test1 + '-' + test2
            save_to = os.path.join(folder,prefix+ test1 + '-' + test2)
        else:
            save_to = ''
        
        X1, Y1 = df[df.test == test1].x.values, df[df.test == test1].y.values
        X2, Y2 = df[df.test == test2].x.values, df[df.test == test2].y.values
        df_pair = prepare_df(X1, Y1, X2, Y2)

        idx_amp = 2
        idx_group_amp = 3
        idx_acr = 4
        idx_group_acr = 5

        if type(df_best_models) == int:
            fit_results, _, statistics_trans, _, ind_test_amp, _, ind_test_acr = test_cosinor_pair(df_pair, period)
        else:            
            period = df_best_models[df_best_models.test == test1].period.iloc[0]
            fit_results, _, statistics_trans, _, ind_test_amp, _, ind_test_acr = test_cosinor_pair(df_pair, period)
            #print(period)
        
        acr1, acr2 = statistics_trans['values'][idx_acr], statistics_trans['values'][idx_group_acr]
        amp1, amp2 = statistics_trans['values'][idx_amp], statistics_trans['values'][idx_group_amp]
        
        d = {'test':test1+' vs. '+test2,
            'p': fit_results.f_pvalue,
            'period': period,
            'amplitude1': statistics_trans['values'][idx_amp],
            'p(amplitude1)':statistics_trans['p-values'][idx_amp],            
            'amplitude2': statistics_trans['values'][idx_group_amp],
            'p(amplitude2)':statistics_trans['p-values'][idx_group_amp],
            'd_amplitude': ind_test_amp['value'],
            'p(d_amplitude)': float(ind_test_amp['p_value']),
            'CI(d_amplitude)': ind_test_amp['conf_int'],
            'acrophase1': statistics_trans['values'][idx_acr],
            'p(acrophase1)':statistics_trans['p-values'][idx_acr],
            'acrophase2': statistics_trans['values'][idx_group_acr],
            'p(acrophase2)':statistics_trans['p-values'][idx_group_acr],
            'd_acrophase': ind_test_acr['value'],
            'p(d_acrophase)': float(ind_test_acr['p_value']),
            'CI(d_acrophase)': ind_test_acr['conf_int']}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
        plot_pair(df_pair, fit_results, test1=test1, test2=test2, plot_measurements=plot_measurements, save_to=save_to, period=period, legend=legend)
        
        CI_l, CI_u = statistics_trans['CI']   
        CI_acrs = [(CI_l[-2],CI_u[-2]), (CI_l[-1], CI_u[-1])]
        CI_amps = [(CI_l[-4],CI_u[-4]), (CI_l[-3], CI_u[-3])]
        
        cosinor.plot_phases([acr1, acr2], [amp1, amp2], [test1, test2], period=24, folder = folder, prefix="phase_", legend=legend, CI_acrs = CI_acrs, CI_amps = CI_amps)
        
    df_results['q'] = multi.multipletests(df_results['p'], method = 'fdr_bh')[1]
    df_results['q(amplitude1)'] = multi.multipletests(df_results['p(amplitude1)'], method = 'fdr_bh')[1]
    df_results['q(amplitude2)'] = multi.multipletests(df_results['p(amplitude2)'], method = 'fdr_bh')[1]
    df_results['q(acrophase1)'] = multi.multipletests(df_results['p(acrophase1)'], method = 'fdr_bh')[1]
    df_results['q(acrophase2)'] = multi.multipletests(df_results['p(acrophase2)'], method = 'fdr_bh')[1]
    
    df_results['q(d_amplitude)'] = multi.multipletests(df_results['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_results['q(d_acrophase)'] = multi.multipletests(df_results['p(d_acrophase)'], method = 'fdr_bh')[1]

    return df_results

# compare two indepedent models using confidence intervals
def test_cosinor_pairs_independent(df, pairs, period = 24, period2 = None, df_best_models = -1):
    
    period1 = period
    if not period2:
        period2 = period

    df_results = pd.DataFrame(columns = ['test',
                                         'p1', 
                                         'q1',
                                         'p2',
                                         'q2', 
                                         'period1',
                                         'period2',
                                         'amplitude1', 
                                         'amplitude2', 
                                         'd_amplitude', 
                                         'p(d_amplitude)', 
                                         'q(d_amplitude)',
                                         'CI(d_amplitude)',
                                         'acrophase1', 
                                         'acrophase2',
                                         'd_acrophase',
                                         'p(d_acrophase)',
                                         'q(d_acrophase)',
                                         'CI(d_acrophase)'], dtype=float)
    
    for test1, test2 in pairs:           
        
        X1, Y1 = df[df.test == test1].x.values, df[df.test == test1].y.values
        X2, Y2 = df[df.test == test2].x.values, df[df.test == test2].y.values
        
        if type(df_best_models) == int:
            pass
        else:            
            period1 = df_best_models[df_best_models.test == test1].period.iloc[0]
            period2 = df_best_models[df_best_models.test == test2].period.iloc[0]            
        
        fit_results1, amp1, acr1, stats1 = fit_cosinor(X1, Y1, period = period1, plot_on=False)
        fit_results2, amp2, acr2, stats2 = fit_cosinor(X2, Y2, period = period2, plot_on=False)

        k = len(X1) + len(X2)
        k1 = len(X1)
        k2 = len(X2)
        DoF = k - (len(fit_results1.params) + len(fit_results2.params))
        DoF1 = k1 - len(fit_results1.params)
        DoF2 = k2 - len(fit_results2.params)
        t = abs(stats.t.ppf(0.05/2,df=DoF))   

        d_amp = amp2-amp1
        d_acr = cosinor.project_acr(acr2-acr1)

        CI_amp1 = stats1['CI'][0][-2], stats1['CI'][1][-2]
        CI_amp2 = stats2['CI'][0][-2], stats2['CI'][1][-2]
        se_amp = cosinor.get_se_diff_from_CIs(CI_amp1, CI_amp2, DoF1, DoF2, t_test = True, angular=False, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF, biased=True) 
        dev_amp = t*se_amp
        T0 = d_amp/se_amp
        p_val_amp = 2 * (1 - stats.t.cdf(abs(T0), DoF))
        CI_amp = [d_amp-dev_amp, d_amp+dev_amp]

        CI_acr1 = stats1['CI'][0][-1], stats1['CI'][1][-1]
        CI_acr2 = stats2['CI'][0][-1], stats2['CI'][1][-1]
        se_acr = cosinor.get_se_diff_from_CIs(CI_acr1, CI_acr2, DoF1, DoF2, t_test = True, angular=True, CI_type = "se", n1 = k1, n2 = k2, DoF = DoF, biased=True) 
        dev_acr = se_acr*t
        T0 = d_acr/se_acr
        p_val_acr = 2 * (1 - stats.t.cdf(abs(T0), DoF))
        CI_acr = [d_acr - dev_acr, d_acr + dev_acr]
             
        d = {'test':test1+' vs. '+test2,
            'p1': fit_results1.f_pvalue,
            'p2': fit_results2.f_pvalue,
            'period1': period1,
            'period2': period2,
            'amplitude1': amp1,
            'amplitude2': amp2,
            'd_amplitude': d_amp,
            'p(d_amplitude)': p_val_amp,
            'CI(d_amplitude)': CI_amp,
            'acrophase1': acr1,
            'acrophase2': acr2,
            'd_acrophase': d_acr,
            'p(d_acrophase)': p_val_acr,
            'CI(d_acrophase)': CI_acr}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
        
    df_results['q1'] = multi.multipletests(df_results['p1'], method = 'fdr_bh')[1]
    df_results['q2'] = multi.multipletests(df_results['p2'], method = 'fdr_bh')[1]
    df_results['q(d_amplitude)'] = multi.multipletests(df_results['p(d_amplitude)'], method = 'fdr_bh')[1]
    df_results['q(d_acrophase)'] = multi.multipletests(df_results['p(d_acrophase)'], method = 'fdr_bh')[1]



    return df_results


def test_cosinor_single(data, period = 24, corrected = True):
    
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
            
        
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ rrr + sss', data).fit()

    beta_s = results.params['sss']
    beta_r = results.params['rrr']
    amp, acr = amp_acr(beta_s, beta_r)
    # project acropahse to interval -pi,pi
    acr = cosinor.project_acr(acr)
    
    vmat = results.cov_params().loc[['rrr', 'sss'], ['rrr', 'sss']]
    indVmat = vmat
        
    a_r = (beta_r**2 + beta_s**2)**(-0.5) * beta_r
    a_s = (beta_r**2 + beta_s**2)**(-0.5) * beta_s
    b_r = (1 / (1 + (beta_s**2 / beta_r**2))) * (-beta_s / beta_r**2)
    b_s = (1 / (1 + (beta_s**2 / beta_r**2))) * (1 / beta_r)
    
    if corrected:
        b_r = -b_r
        b_s = -b_s


    jac = np.array([[a_r, a_s], [b_r, b_s]]) 
    
    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only =np.sqrt(np.diag(cov_trans))
    zt = abs(stats.norm.ppf((1-0.95)/2))

    trans_names = [results.params.index.values[0]] + ['amp', 'acr']
    
    coef_trans = np.array([results.params.iloc[0], amp, acr])
    se_trans = np.concatenate((np.sqrt(np.diag(results.cov_params().loc[['Intercept'], ['Intercept']])),se_trans_only))  

    
    
    
 
    lower_CI_trans = coef_trans - np.abs(zt * se_trans)
    upper_CI_trans = coef_trans + np.abs(zt * se_trans)
    p_value_trans = 2 * stats.norm.cdf(-np.abs(coef_trans/se_trans)) 

    statistics= {'parameters': trans_names,
                    'values': coef_trans,
                    'SE': se_trans,
                    'CI': (lower_CI_trans, upper_CI_trans),
                    'p-values': p_value_trans,
                    'F-test': results.f_pvalue}    
       
    return results, amp, acr, statistics
                    
    


def test_cosinor_pair(data, period, corrected = True):
    
    rrr = np.cos(2*np.pi*data.x/period)
    sss = np.sin(2*np.pi*data.x/period)
            
        
    data['rrr'] = rrr
    data['sss'] = sss

    results = smf.ols('y ~ group + rrr + sss + group:rrr + group:sss', data).fit()

    beta_s = np.array([results.params['sss'], results.params['group:sss']])
    beta_r = np.array([results.params['rrr'], results.params['group:rrr']])
        
    groups_s = beta_s.copy()
    groups_s[1] += groups_s[0]
    groups_r = beta_r.copy()
    groups_r[1] += groups_r[0]
        
    amp, acr = amp_acr(groups_s, groups_r)
          
    vmat = results.cov_params().loc[['rrr', 'group:rrr', 'sss', 'group:sss'],['rrr', 'group:rrr', 'sss', 'group:sss'],]   
    indexmat = np.eye(4)
    indexmat[1,0]=1
    indexmat[3,2]=1
    indVmat = np.dot(np.dot(indexmat, vmat), np.transpose(indexmat))

    a_r = (groups_r**2 + groups_s**2)**(-0.5) * groups_r
    a_s = (groups_r**2 + groups_s**2)**(-0.5) * groups_s

    b_r = (1 / (1 + (groups_s**2 / groups_r**2))) * (-groups_s / groups_r**2)
    b_s = (1 / (1 + (groups_s**2 / groups_r**2))) * (1 / groups_r)

    if corrected:
        b_r = - b_r
        b_s = - b_s

    jac = np.array((np.concatenate((np.diag(a_r)[0,:], np.diag(a_s)[0,:])),
    np.concatenate((np.diag(a_r)[1,:], np.diag(a_s)[1,:])),
    np.concatenate((np.diag(b_r)[0,:], np.diag(b_s)[0,:])),
    np.concatenate((np.diag(b_r)[1,:], np.diag(b_s)[1,:]))))

    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only =np.sqrt(np.diag(cov_trans))
    
    
    zt = abs(stats.norm.ppf((1-0.95)/2))
    
    
    coef_raw = results.params
    raw_names = list(coef_raw.index.values)
    coef_raw = coef_raw.values
    se_raw = np.sqrt(np.diag(results.cov_params()))
    lower_CI_raw = coef_raw - zt * se_raw
    upper_CI_raw = coef_raw + zt * se_raw
    p_value_raw = 2 * stats.norm.cdf(-np.abs(coef_raw/se_raw))
    statistics_raw={'parameters': raw_names,
                    'values': coef_raw,
                    'SE': se_raw,
                    'CI': (lower_CI_raw, upper_CI_raw),
                    'p-values': p_value_raw}    
    
    trans_names = list(results.params.index.values[:2]) + ['amp', 'group:amp', 'acr', 'group:acr']
    coef_trans = np.concatenate((np.array(results.params.iloc[0:2]), amp, acr))
    # correct acropahse
    coef_trans[-1] = cosinor.project_acr(coef_trans[-1])
    coef_trans[-2] = cosinor.project_acr(coef_trans[-2])

    se_trans = np.concatenate((np.sqrt(np.diag(results.cov_params().loc[['Intercept', 'group'], ['Intercept', 'group']])),se_trans_only))             
    lower_CI_trans = coef_trans - zt * se_trans
    upper_CI_trans = coef_trans + zt * se_trans
    p_value_trans = 2 * stats.norm.cdf(-np.abs(coef_trans/se_trans))
    statistics_trans={'parameters': trans_names,
                    'values': coef_trans,
                    'SE': se_trans,
                    'CI': (lower_CI_trans, upper_CI_trans),
                    'p-values': p_value_trans}    
    
    
    
    diff_est_amp = coef_trans[3] - coef_trans[2]
    idx = np.array([-1,1,0,0])
    diff_var_amp = np.dot(np.dot(idx, cov_trans), np.transpose(idx[np.newaxis]))
    glob_chi_amp = diff_est_amp * (1/diff_var_amp) * diff_est_amp
    ind_Z_amp = diff_est_amp/np.sqrt(np.diag(diff_var_amp))
    interval_amp = np.array((diff_est_amp - 1.96 * np.sqrt(np.diag(diff_var_amp)), diff_est_amp + 1.96 * np.sqrt(np.diag(diff_var_amp))))
    df_amp = 1
    global_p_value_amp = 1-stats.chi2.cdf(glob_chi_amp, df_amp)
    ind_p_value_amp = 2*stats.norm.cdf(-abs(ind_Z_amp))

    diff_est_acr = coef_trans[5] - coef_trans[4]
    diff_est_acr = cosinor.project_acr(diff_est_acr)
    idx = np.array([0,0,-1,1])
    diff_var_acr = np.dot(np.dot(idx, cov_trans), np.transpose(idx[np.newaxis]))
    glob_chi_acr = diff_est_acr * (1/diff_var_acr) * diff_est_acr
    ind_Z_acr = diff_est_acr/np.sqrt(np.diag(diff_var_acr))
    interval_acr = np.array((diff_est_acr - 1.96 * np.sqrt(np.diag(diff_var_acr)), diff_est_acr + 1.96 * np.sqrt(np.diag(diff_var_acr))))
    df_acr = 1
    global_p_value_acr = 1-stats.chi2.cdf(glob_chi_acr, df_acr)
    ind_p_value_acr = 2*stats.norm.cdf(-abs(ind_Z_acr))
    
    
    global_test_amp = {'name': 'global test of amplitude change',
                       'statistics': glob_chi_amp.flatten(),
                       'df': df_amp,
                       'p_value': global_p_value_amp.flatten()}
    ind_test_amp = {'name': 'individual test of amplitude change',
                    'statistics': ind_Z_amp.flatten(),
                    'df': np.nan,
                    'value': diff_est_amp,
                    'conf_int': interval_amp.flatten(),
                    'p_value': ind_p_value_amp.flatten()}

    global_test_acr = {'name': 'global test of acrophase shift',
                       'statistics': glob_chi_acr.flatten(), 
                       'df': df_acr,
                       'p_value': global_p_value_acr.flatten()}
    ind_test_acr = {'name': 'individual test of acrophase shift',
                    'statistics': ind_Z_acr.flatten(),
                    'df': np.nan,
                    'value': diff_est_acr,
                    'conf_int': interval_acr.flatten(),
                    'p_value': ind_p_value_acr.flatten()}
    
    return results, statistics_raw, statistics_trans, global_test_amp, ind_test_amp, global_test_acr, ind_test_acr
                    

# amplitude_detection: determines the minimal number of samples to obtain a statistically significant result for the zero amplitude test
# A ... presumed amplitude
# var ... residual variance
# p ... threshold for statistical significance of the zero amplitude test
# alpha ... type I error probability
def amplitude_detection(A, var, p = 0.95, alpha = 0.05):

    sigma2 = var
    C_2 = A**2/sigma2 

    N = 4
    
    while True:
        lmbda = (N * C_2)/4       
        f2 = stats.f(2, N-3, 0).ppf(1-alpha)
        F = 1-stats.ncf(2,N-3,lmbda).cdf(f2)
        if F >= p:
            break
        
        N += 1
        
    return N

# amplitude_confidence: determines the minimal number of samples to obtain a given length of the confidence interval for the estimated amplitude
# L ... maximal acceptable length of the confidence interval
# var ... residual variance
# alpha ... 1-alpha = confidence level of the cofidence interval
def amplitude_confidence(L, var, alpha = 0.05):
    sigma2 = var
    N = 8 * sigma2 * (stats.norm.ppf(1-alpha/2)**2)/(L**2)
    return int(np.ceil(N))

# acrophase_confidence: determines the minimal number of samples to obtain a given length of the confidence interval for the estimated acrophase
# L ... maximal acceptable length of the confidence interval
# A_0 ... presumed minimal amplitude
# var ... residual variance
# alpha ... 1-alpha = confidence level of the cofidence interval
def acrophase_confidence(L, A_0, var, alpha = 0.05):
    sigma2 = var
    #N = 8 * sigma2 * (stats.norm.ppf(1-alpha/2)**2)/(L**2 * A_0**2) # approximation
    N = (2 * stats.norm.ppf(1-alpha/2)**2 * sigma2)/(A_0**2 * np.sin(L/2)**2) # accurate

    return int(np.ceil(N))

# acrophase_shift_detection: determines the minimal number of samples to detect a specific shift in the acrophase
def acrophase_shift_detection(shift, A_0, var, alpha = 0.05):
    L = 0.5*shift
    return acrophase_confidence(L, A_0, var, alpha)    

    

