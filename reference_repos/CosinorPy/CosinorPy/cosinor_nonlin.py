from scipy.linalg.decomp import eigvals_banded
from CosinorPy import cosinor

import numpy as np
np.seterr(divide='ignore')
import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import percentileofscore
from scipy.stats import circstd, circmean

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multi
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import os

eps = 10**-10

from CosinorPy.helpers import df_add_row


#####################
# FITTING FUNCTIONS #
#####################

# basic fit
def cosinor_basic(predictor, A, B, acrophase, period):
    X = predictor
    return A + B * np.cos(2*np.pi*X/period + acrophase)

def cosinor_basic2(predictor, A, B, acrophase, B2, acrophase2, period):
    X = predictor
    return A + B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2)

def cosinor_basic3(predictor, A, B, acrophase, B2, acrophase2, B3, acrophase3, period):
    X = predictor
    return A + B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3)

def cosinor_basic4(predictor, A, B, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4, period):
    X = predictor
    return A + B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3) + B4 * np.cos(2*np.pi*X/(period/4) + acrophase4)

# lin comp
def cosinor_lin_comp(predictor, A, B, C, acrophase, period):
    X = predictor
    return A + B * np.cos(2*np.pi*X/period + acrophase) + C * X 


# generalised models

# A ... MESOR
# B ... amplitude
# C ... amplification coefficient: 
#  C > 0 ... oscillations are amplified with time
#  C < 0 ... oscillations are damped with time
#  C = 0 ... sustained oscillations
# D ... linear component
# Phi ... acrophase
# Per ... period
def generalized_cosinor(predictor, A, B, C, D, acrophase, period):
    X = predictor
    #return A + B * (1 + C*X) * np.cos(2*np.pi*X/period + acrophase) + D * X 
    return A + B * np.exp(C*X) * np.cos(2*np.pi*X/period + acrophase) + D * X 

def generalized_cosinor2(predictor, A, B, C, D, acrophase, B2, acrophase2, period):
    X = predictor
    #return A + (1 + C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2)) + D * X 
    return A + np.exp(C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2)) + D * X 

def generalized_cosinor3(predictor, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, period):
    X = predictor
    #return A + (1 + C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3)) + D * X 
    return A + np.exp(C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3)) + D * X 

def generalized_cosinor4(predictor, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4, period):
    X = predictor
    #return A + (1 + C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3) + B4 * np.cos(2*np.pi*X/(period/4) + acrophase4)) + D * X 
    return A + np.exp(C*X) * (B * np.cos(2*np.pi*X/period + acrophase) + B2 * np.cos(2*np.pi*X/(period/2) + acrophase2) + B3 * np.cos(2*np.pi*X/(period/3) + acrophase3) + B4 * np.cos(2*np.pi*X/(period/4) + acrophase4)) + D * X 

"""
# the model below is not ok because we cannot assess the significance of amplitude being different than zero directly
def generalized_cosinor_exp(predictor, A, B, C, D, acrophase, period):
    X = predictor
    return A + B * (1 + np.exp(C*X)) * np.cos(2*np.pi*X/period + acrophase) + D * X 
    # opcije: 
    #  ... + np.exp(D*X)
    #  ... + np.log(D*X)
"""
  
def generalized_cosinor_compare(predictor, A, B, C, D, acrophase, A0, B0, C0, D0, acrophase0, period):
    X = predictor[0]
    H = predictor[1]
    #return (A + H*A0) + (B + H*B0) * (1 + (C + H*C0)*X) * np.cos(2*np.pi*X/period + (acrophase + H*acrophase0)) + (D + H*D0) * X 
    return (A + H*A0) + (B + H*B0) * np.exp((C + H*C0)*X) * np.cos(2*np.pi*X/period + (acrophase + H*acrophase0)) + (D + H*D0) * X 
"""
def generalized_cosinor_compare_exp(predictor, A, B, C, D, acrophase, A0, B0, C0, D0, acrophase0, period):
    X = predictor[0]
    H = predictor[1]
    return (A + H*A0) + (B + H*B0) * (1 + np.exp((C + H*C0)*X)) * np.cos(2*np.pi*X/period + (acrophase + H*acrophase0)) + (D + H*D0) * X 
"""


#########################################
# STATISTICS AND RHYTHMICITY PARAMETERS #
#########################################

def calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters):
    # Compute standard errors of parameter estimates
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    # how to evaluate standard errors of coefficients
    # https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i

    perr = np.sqrt(np.diag(pcov))
    #DoF = len(Y) - len(popt)
    p_values=np.zeros(len(popt))
    CIs=np.zeros((len(popt),2))

    # statistics
    #if t_test:
    n_devs = abs(stats.t.ppf(0.05/2,df=DoF)) # number of standard deviations for CIs
    #else:
    #    n_devs = 1.96

    for i in np.arange(len(perr)):             
        T0 = popt[i]/perr[i]
        p_values[i] = 2 * (1 - stats.t.cdf(abs(T0), DoF))
        lower = popt[i] - n_devs * np.abs(perr[i])
        upper = popt[i] + n_devs * np.abs(perr[i])
        CIs[i] = np.array([lower, upper])
    
    p_dict = {}
    p_dict['params'] = {}
    p_dict['p_values'] = {}
    p_dict['CIs'] = {}
    for param, val, p_val, CI in zip(parameters, popt, p_values, CIs):
        p_dict['params'][param] = val
        p_dict['p_values'][param] = p_val
        p_dict['CIs'][param] = list(CI)

    return p_dict

def calculate_statistics_nonlinear(X, Y, Y_fit, n_params, period):
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean())**2)
    RSS = sum((Y - Y_fit)**2)
    N = Y.size

    try:
        F = (MSS/(n_params - 1)) / (RSS/(N - n_params)) 
        p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)
    except:
        p = 1
    
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
    try:
        F = (SSLOF/(n_T-n_params))/(SSPE/(N-n_T))
        p_reject = 1 - stats.f.cdf(F, n_T-n_params, N-n_T)
    except:
        p_reject = 1
    
    
    
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
    try:
        resid_SE = np.sqrt(RSS/DoF)
        # https://scientificallysound.org/2017/05/16/independent-t-test-python/
        # https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/margin-of-error/
        critical_value = stats.t.ppf(1-0.025, DoF)
        ME = critical_value * resid_SE
    except:
        resid_SE = np.nan
        ME = np.nan


    
    return {'p':p, 'p_reject':p_reject, 'SNR':SNR, 'RSS': RSS, 'resid_SE': resid_SE, 'ME': ME}


##############
# REGRESSION #
##############

def fit_cosinor_basic(X,Y, period=24, min_per = 12, max_per=36, plot=False, plot_margins=True, save_to = "", **kwargs):
    
    min_bounds = {'A':min(0, min(Y)), 
                      'B':0,
                      'acrophase':-np.pi}                      
                      
    max_bounds = {'A':max(Y), 
                      'B':abs(max(Y)), 
                      'acrophase':np.pi}
    if not period:
        min_bounds['period'] =  min_per
        max_bounds['period'] =  max_per


    if period:
        parameters = ['A', 'B', 'acrophase']
    else:
        parameters = ['A', 'B', 'acrophase', 'period']

    predictor = X
    min_bounds = [min_bounds[name] for name in parameters]
    max_bounds = [max_bounds[name] for name in parameters]  

    if period:
        popt, pcov = curve_fit(lambda x, A, B, acrophase: cosinor_basic(x, A, B, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
    else:
        popt, pcov = curve_fit(cosinor_basic, predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
   
    DoF = Y.size - len(popt)

    if not period:
        period = popt[-1]
        popt_ext = list(popt)
    else:
        popt_ext = list(popt) + [period]

    Y_fit = cosinor_basic(predictor, *popt_ext)
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    statistics_params = calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters)

    if plot:
        X_plot = np.linspace(min(X), max(X), 1000)
        Y_plot = cosinor_basic(X_plot, *popt_ext)
        plt.plot(X_plot, Y_plot, label='fit', color="black")
        plt.plot(X, Y, 'o', markersize=1, label='measurements', color="black")
        if plot_margins:
            lower = Y_plot - statistics['ME']
            upper = Y_plot + statistics['ME']
            plt.fill_between(X_plot, lower, upper, color="black", alpha=0.1)
        
        plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])

        plt.legend()
        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()


    return popt_ext, statistics, statistics_params   



def fit_cosinor_basic_n_comp(X,Y, period=24, n_components = 1, min_per = 12, max_per=36, plot=False, plot_margins=True, save_to = "", **kwargs):
    min_bounds = {'A':min(0, min(Y)), 
                      'B':0,
                      'B2':0,
                      'B3':0,
                      'B4':0,
                      'acrophase':-np.pi,
                      'acrophase2':-np.pi,
                      'acrophase3':-np.pi,
                      'acrophase4':-np.pi}
                      
    max_bounds = {'A':abs(max(Y)), 
                      'B':abs(max(Y)), 
                      'B2':abs(max(Y)),
                      'B3':abs(max(Y)),
                      'B4':abs(max(Y)),
                      'acrophase':np.pi,
                      'acrophase2':np.pi,
                      'acrophase3':np.pi,
                      'acrophase4':np.pi}
    if not period:
        min_bounds['period'] =  min_per
        max_bounds['period'] =  max_per

    parameters = ['A', 'B', 'acrophase']
    for i in range(2, n_components + 1):
        parameters += [f"B{i}", f"acrophase{i}"]

    if not period:
        parameters += ['period']

    predictor = X
    min_bounds = [min_bounds[name] for name in parameters]
    max_bounds = [max_bounds[name] for name in parameters]  

    if n_components == 1:
        fitting_func = cosinor_basic       
    elif n_components == 2:
        fitting_func = cosinor_basic2
    elif n_components == 3:
        fitting_func = cosinor_basic3
    elif n_components == 4:
        fitting_func = cosinor_basic4
    else:
        print("Invalid option!")
        return



    if period:
        if n_components == 1:
            popt, pcov = curve_fit(lambda x, A, B, acrophase: fitting_func(x, A, B, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
        elif n_components == 2:
            popt, pcov = curve_fit(lambda x, A, B, acrophase, B2, acrophase2: fitting_func(x, A, B, acrophase, B2, acrophase2, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
        elif n_components == 3:
            popt, pcov = curve_fit(lambda x, A, B, acrophase, B2, acrophase2, B3, acrophase3: fitting_func(x, A, B, acrophase, B2, acrophase2, B3, acrophase3, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
        elif n_components == 4:
            popt, pcov = curve_fit(lambda x, A, B, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4: fitting_func(x, A, B, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
    else:
        popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)

  
    
    DoF = Y.size - len(popt)

    if not period:
        period = popt[-1]
        popt_ext = list(popt)
    else:
        popt_ext = list(popt) + [period]

    Y_fit = fitting_func(predictor, *popt_ext)
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    statistics_params = calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters)

    if plot:
        X_plot = np.linspace(min(X), max(X), 1000)
        Y_plot = fitting_func(X_plot, *popt_ext)
        plt.plot(X_plot, Y_plot, label='fit', color="black")
        plt.plot(X, Y, 'o', markersize=1, label='measurements', color="black")
        if plot_margins:
            lower = Y_plot - statistics['ME']
            upper = Y_plot + statistics['ME']
            plt.fill_between(X_plot, lower, upper, color="black", alpha=0.1)

        plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])

        plt.legend()
        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()


    # evaluate rhythm params
    X_eval = np.linspace(0, 2*period, 1000)
    Y_eval = fitting_func(X_eval, *popt_ext)
    rhythm_params = cosinor.evaluate_rhythm_params(X_eval, Y_eval, period=period)

    # returns
    return popt_ext, statistics, statistics_params, rhythm_params   


# if period:
#   popt,pcov = curve_fit(lambda x, A, B, C, D, phi: generalized_cosinor(x, A, B, C, D, phi, PER), x, y, bounds)
# else:
#   popt,pcov = curve_fit(generalized_cosinor, x, y, bounds)
def fit_generalized_cosinor(X,Y, period=24, min_per = 12, max_per=36, plot=False, plot_margins=True, save_to = "", x_label="time [h]", y_label="measurements", test="", hold_on=False, color="black", lin_comp = True, amp_comp = True, **kwargs):

    #if not exp:
    fitting_func = generalized_cosinor
    #else:
    #    fitting_func = generalized_cosinor_exp

    max_C = 1 if amp_comp else eps
    max_D = 10 if lin_comp else eps
     

    min_bounds = {'A':min(0, min(Y)), 
                      'B':0,
                      'C':-max_C,
                      'D':-max_D,
                      'acrophase':-np.pi}                      
                      
    max_bounds = {'A':abs(max(Y)), 
                      'B':abs(max(Y)), 
                      'C':max_C,
                      'D':max_D,
                      'acrophase':np.pi}
    if not period:
        min_bounds['period'] =  min_per
        max_bounds['period'] =  max_per
    
    if period: # if period is not specified
        parameters = ['A', 'B', 'C', 'D', 'acrophase']
    else:
        parameters = ['A', 'B', 'C', 'D', 'acrophase', 'period']

    predictor = X
    min_bounds = [min_bounds[name] for name in parameters]
    max_bounds = [max_bounds[name] for name in parameters]  

   
    # parameters = ['A', 'B', 'acrophase', 'period']
    p0, _, _ = fit_cosinor_basic(X,Y, period = period) # p0, statistics0, statistics_param0
    # if not generalized:
    # add C and D to these parameters, set to 0, p-value to 1, confidence intervals to 0,0
    #   return these parameters, plot...
    

    p0 = p0[:2] + [0,0] + p0[2:]


    try:
        if period:
            try:
                popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase: fitting_func(x, A, B, C, D, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0[:-1], **kwargs)
            except:
                popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase: fitting_func(x, A, B, C, D, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)            
        else:
            try:
                popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0, **kwargs)
            except:
                popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
    except:         
        print(f"Divergence error at {test} with 1 components!")         
        return 
    
    DoF = Y.size - len(popt)

    if not period:
        period = popt[-1]
        popt_ext = popt
    else:
        popt_ext = list(popt) + [period]

    Y_fit = fitting_func(predictor, *popt_ext)
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    statistics_params = calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters)

    if plot:
        X_plot = np.linspace(min(X), max(X), 1000)
        Y_plot = fitting_func(X_plot, *popt_ext)
        plt.plot(X_plot, Y_plot, label='fit', color=color)
        if not hold_on:
            plt.plot(X, Y, 'o', markersize=1, label='measurements', color=color)
        else:
            plt.plot(X, Y, 'o', markersize=1, label='_nolegend_', color=color)
        if plot_margins:
            lower = Y_plot - statistics['ME']
            upper = Y_plot + statistics['ME']
            plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)
        

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if not hold_on:
            plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])
            if test:
                plt.title(test)
            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()

    return popt_ext, statistics, statistics_params   

def fit_generalized_cosinor_n_comp(X,Y, period=24, n_components = 1, min_per = 12, max_per=36, plot=False, plot_margins=True, color = 'black', hold_on = False, save_to = "", x_label="time [h]", y_label="measurements", test="", lin_comp = True, amp_comp = True, **kwargs):

   

    #if not exp:
    #    fitting_func = generalized_cosinor
    #else:
    #    fitting_func = generalized_cosinor_exp

    max_C = 1 if amp_comp else eps
    max_D = 10 if lin_comp else eps


    min_bounds = {'A':min(0, min(Y)), 
                      'B':0,
                      'C':-max_C,
                      'D':-max_D,
                      'B2':0,
                      'B3':0,
                      'B4':0,
                      'acrophase':-np.pi,
                      'acrophase2':-np.pi,
                      'acrophase3':-np.pi,
                      'acrophase4':-np.pi}                 
                      
    max_bounds = {'A':abs(max(Y)), 
                      'B':abs(max(Y)), 
                      'C':max_C,
                      'D':max_D,
                      'B2':abs(max(Y)),
                      'B3':abs(max(Y)),
                      'B4':abs(max(Y)),
                      'acrophase':np.pi,
                      'acrophase2':np.pi,
                      'acrophase3':np.pi,
                      'acrophase4':np.pi}
    if not period:
        min_bounds['period'] =  min_per
        max_bounds['period'] =  max_per


    parameters = ['A', 'B', 'C', 'D', 'acrophase']
    
    for i in range(2, n_components + 1):
        parameters += [f"B{i}", f"acrophase{i}"]

    if not period:
        parameters += ['period']


    predictor = X
    min_bounds = [min_bounds[name] for name in parameters]
    max_bounds = [max_bounds[name] for name in parameters]  
   
    if n_components == 1:
        fitting_func = generalized_cosinor       
    elif n_components == 2:
        fitting_func = generalized_cosinor2
    elif n_components == 3:
        fitting_func = generalized_cosinor3
    elif n_components == 4:
        fitting_func = generalized_cosinor4
    else:
        print("Invalid option!")
        return

    # parameters = ['A', 'B', 'acrophase', 'period']
    p0, _, _, _ = fit_cosinor_basic_n_comp(X,Y, period = period, n_components=n_components, plot=False) # p0, statistics0, statistics_param0
    # if not generalized:
    # add C and D to these parameters, set to 0, p-value to 1, confidence intervals to 0,0
    #   return these parameters, plot...
    p0 = p0[:2] + [0,0] + p0[2:]
    
    try:
        if period:
            if n_components == 1:
                try:                    
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase: fitting_func(x, A, B, C, D, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0[:-1], **kwargs)
                except:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase: fitting_func(x, A, B, C, D, acrophase, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)                
            elif n_components == 2:
                try:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, period), predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0[:-1], **kwargs)
                except:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
                
            elif n_components == 3:
                try:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, period), predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0[:-1], **kwargs)
                except:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
            elif n_components == 4:
                try:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4, period), predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0[:-1], **kwargs)        
                except:
                    popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4: fitting_func(x, A, B, C, D, acrophase, B2, acrophase2, B3, acrophase3, B4, acrophase4, period), predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)        
        else:
            try:
                popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), p0 = p0, **kwargs)
            except:
                popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), **kwargs)
    except: 
        print(f"Divergence error at {test} with {n_components} components!")         
        return 

    DoF = Y.size - len(popt)

    if not period:
        period = popt[-1]
        popt_ext = popt
    else:
        popt_ext = list(popt) + [period]

    Y_fit = fitting_func(predictor, *popt_ext)
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    statistics_params = calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters)

    if plot:
        X_plot = np.linspace(min(X), max(X), 1000)
        Y_plot = fitting_func(X_plot, *popt_ext)
        plt.plot(X_plot, Y_plot, label='fit', color=color)        
        if hold_on==False:
            plt.plot(X, Y, 'o', markersize=1, label='measurements', color=color)
        else:
            plt.plot(X, Y, 'o', markersize=1, label='_nolegend_', color=color)
        if plot_margins:
            lower = Y_plot - statistics['ME']
            upper = Y_plot + statistics['ME']
            plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)        
        
        if hold_on == False:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(test)
            plt.legend()

            plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])

            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()

    # evaluate rhythm params
    popt_eval = popt_ext.copy()
    popt_eval[2] = 0 #set C to 0
    popt_eval[3] = 0 #set D to 0
    X_eval = np.linspace(0, 2*period, 1000)
    Y_eval = fitting_func(X_eval, *popt_eval)    
    rhythm_params = cosinor.evaluate_rhythm_params(X_eval, Y_eval, period=period)
        
    rhythm_params['amplification'] = statistics_params['params']['C']
    rhythm_params['lin_comp'] = statistics_params['params']['D']

    # returns
    return popt_ext, statistics, statistics_params, rhythm_params   

def population_fit_generalized_cosinor_n_comp(df_pop, period=24, n_components = 1, plot=False, plot_margins = True, plot_individuals=True, save_to = "", hold_on = False, color="black", x_label="time [h]", y_label="measurements", **kwargs):
 
    parameters = ['A', 'B', 'C', 'D', 'acrophase']
    
    for i in range(2, n_components + 1):
        parameters += [f"B{i}", f"acrophase{i}"]

    if not period:
        parameters += ['period']

    if n_components == 1:
        fitting_func = generalized_cosinor       
    elif n_components == 2:
        fitting_func = generalized_cosinor2
    elif n_components == 3:
        fitting_func = generalized_cosinor3
    elif n_components == 4:
        fitting_func = generalized_cosinor4
    else:
        print("Invalid option!")
        return

    tests = df_pop.test.unique()
    k = len(tests)

    popts = []
    if plot_margins:
        Y_plot_all = []

    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)
    if plot:
        X_plot = np.linspace(min_X, max_X, 1000)

    for test in tests:
        X,Y = df_pop[df_pop.test == test].x.values, df_pop[df_pop.test == test].y.values
        popt, _, _, _ = fit_generalized_cosinor_n_comp(X,Y, period=period, n_components=n_components, plot=False, **kwargs)
        popts.append(popt)
        if plot:
            plt.plot(X,Y,'o', color=color, markersize=1, label="_nolegend_")    
            if plot_individuals:
                Y_plot = fitting_func(X_plot, *popt)         
                plt.plot(X_plot,Y_plot,color=color, alpha=0.25, label="_nolegend_")
            if plot_margins:
                Y_plot_all.append(Y_plot)

            plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])
            

    params = np.array(popts)
    # parameter statistics: means, variances, stadndard deviations, confidence intervals, p-values
    #http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    if k > 1:
        means = np.mean(params, axis=0)
        variances = np.sum((params-np.mean(params, axis=0))**2, axis = 0)/(k-1) # np.var(params, axis=0) # isto kot var z ddof=k-1
        sd = variances**0.5

        # different functions need to be used for the estimation of angular means and standard deviations
        for i in range(n_components):
            idx = 2*i+4            
            phase = params[:, idx]               
            means[idx] = cosinor.project_acr(circmean(phase))
            sd[idx] = circstd(phase)


        se = sd/((k-1)**0.5)
        T0 = means/se
        p_values = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        t = abs(stats.t.ppf(0.05/2,df=k-1))
        lower_CI = means - ((t*sd)/((k-1)**0.5))
        upper_CI = means + ((t*sd)/((k-1)**0.5))                
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

    if plot: 
        Y_plot = fitting_func(X_plot, *means)
        plt.plot(X_plot, Y_plot, color=color)#, label=pop_name)

        if plot_margins:
            Y_plot_all = np.array(Y_plot_all)
            var_Y = np.var(Y_plot_all, axis=0, ddof = k-1)
            sd_Y = var_Y**0.5
            lower = Y_plot - ((t*sd_Y)/((k-1)**0.5))
            upper = Y_plot + ((t*sd_Y)/((k-1)**0.5))
            plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)  
        
        if not hold_on:
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            test = tests[0].split("_")[0]
            plt.title(test)

            plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])

            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()

    # evaluate rhythm params
    popt_eval = means.copy()
    popt_eval[2] = 0 #set C to 0
    popt_eval[3] = 0 #set D to 0
    X_eval = np.linspace(0, 2*period, 1000)
    Y_eval = fitting_func(X_eval, *popt_eval)
    rhythm_params = cosinor.evaluate_rhythm_params(X_eval, Y_eval, period=period)

    CIs = list(zip(lower_CI, upper_CI))
    #return {"params":means, "p_values": p_values, "CIs": CIs}

    p_dict = {}
    p_dict['params'] = {}
    p_dict['p_values'] = {}
    p_dict['CIs'] = {}
    
    for param, val, p_val, CI in zip(parameters, means, p_values, CIs):
        p_dict['params'][param] = val
        p_dict['p_values'][param] = p_val
        p_dict['CIs'][param] = list(CI)

    
    X = np.array(df_pop.x)
    Y = np.array(df_pop.y)
   
    Y_fit =  fitting_func(X, *means)
    p_dict['RSS'] = sum((Y - Y_fit)**2)

    if not period:
        period = p_dict['params']['period']

    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    
    return statistics, p_dict, rhythm_params


def population_fit_generalized_cosinor(df_pop, period=24, plot=False, plot_margins = True, plot_individuals=True, hold_on=False, save_to = "", x_label="time [h]", y_label="measurements", color='black', **kwargs):
    #if not exp:
    fitting_func = generalized_cosinor
    #else:
    #    fitting_func = generalized_cosinor_exp

    if period: # if period is specified
        parameters = ['A', 'B', 'C', 'D', 'acrophase']
    else:
        parameters = ['A', 'B', 'C', 'D', 'acrophase', 'period']
    
    tests = df_pop.test.unique()
    k = len(tests)

    popts = []
    if plot_margins:
        Y_plot_all = []

    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)
    if plot:
        X_plot = np.linspace(min_X, max_X, 1000)

    for test in tests:
        X,Y = df_pop[df_pop.test == test].x.values, df_pop[df_pop.test == test].y.values
        popt, _, _ = fit_generalized_cosinor(X,Y, period=period, plot=False, plot_margins=plot_margins, **kwargs)
        popts.append(popt)
        if plot:
            plt.plot(X,Y,'o', color=color, markersize=1, label='_nolegend_')    
            if plot_individuals:
                Y_plot = fitting_func(X_plot, *popt)         
                plt.plot(X_plot,Y_plot,color=color, alpha=0.25, label='_nolegend_')
            if plot_margins:
                Y_plot_all.append(Y_plot)
            
            plt.axis([min(X), max(X), 0.9*min(min(Y), min(Y_plot)), 1.1*max(max(Y), max(Y_plot))])
            

    params = np.array(popts)
    # parameter statistics: means, variances, stadndard deviations, confidence intervals, p-values
    #http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    if k > 1:      
        means = np.mean(params, axis=0)
        variances = np.sum((params-np.mean(params, axis=0))**2, axis = 0)/(k-1) # np.var(params, axis=0) # isto kot var z ddof=k-1
        sd = variances**0.5
       
        # different functions need to be used for the estimation of angular means and standard deviations
        phase = params[:, -2]        
        means[-2] = cosinor.project_acr(circmean(phase))
        sd[-2] = circstd(phase)
     
        se = sd/((k-1)**0.5)
        T0 = means/se
        p_values = 2 * (1 - stats.t.cdf(abs(T0), k-1))
        t = abs(stats.t.ppf(0.05/2,df=k-1))
        lower_CI = means - ((t*sd)/((k-1)**0.5))
        upper_CI = means + ((t*sd)/((k-1)**0.5))                
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

    if plot: 
        Y_plot = fitting_func(X_plot, *means)
        plt.plot(X_plot, Y_plot, color=color)#, label=pop_name)

        if plot_margins:
            Y_plot_all = np.array(Y_plot_all)
            var_Y = np.var(Y_plot_all, axis=0, ddof = k-1)
            sd_Y = var_Y**0.5
            lower = Y_plot - ((t*sd_Y)/((k-1)**0.5))
            upper = Y_plot + ((t*sd_Y)/((k-1)**0.5))
            plt.fill_between(X_plot, lower, upper, color=color, alpha=0.1)  

        plt.axis([min(df_pop.x.values), max(df_pop.x.values), 0.9*min(min(df_pop.y.values), min(Y_plot)), 1.1*max(max(df_pop.y.values), max(Y_plot))])

        if hold_on == False:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(tests[0].split("_")[0])
            
            if save_to:
                plt.savefig(save_to+'.png')
                plt.savefig(save_to+'.pdf')
                plt.close()
            else:
                plt.show()

    CIs = list(zip(lower_CI, upper_CI))
    #return {"params":means, "p_values": p_values, "CIs": CIs}

    p_dict = {}
    p_dict['params'] = {}
    p_dict['p_values'] = {}
    p_dict['CIs'] = {}
    for param, val, p_val, CI in zip(parameters, means, p_values, CIs):
        p_dict['params'][param] = val
        p_dict['p_values'][param] = p_val
        p_dict['CIs'][param] = list(CI)


    X = np.array(df_pop.x)
    Y = np.array(df_pop.y)

    Y_fit =  fitting_func(X, *means)
    p_dict['RSS'] = sum((Y - Y_fit)**2)

    if not period:
        period = p_dict['params']['period']

    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    p_dict['statistics'] = statistics

    return p_dict

 
def fit_generalized_cosinor_compare(X1, Y1, X2, Y2, period=24, min_per = 12, max_per=36, plot=False, plot_margins=True, save_to = "", test1 = "test1", test2 = "test2", x_label="time [h]", y_label="measurements", lin_comp=True, amp_comp = True, **kwargs):
    #if not exp:
    fitting_func = generalized_cosinor_compare
    #else:
    #    fitting_func = generalized_cosinor_compare_exp


    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    Y = np.concatenate((Y1, Y2))    
    X = np.concatenate((X1, X2))    
    H = np.concatenate((H1, H2))
    predictor = np.array([X,H])

    max_C = 1 if amp_comp else eps
    max_D = 10 if lin_comp else eps

    min_bounds =   {'A':min(0, min(Y)), 
                    'B':0,
                    'C':-max_C,
                    'D':-max_D,
                    'acrophase':-np.pi,
                    'A0':min(-2*max(Y), 2*min(Y)), 
                    'B0':-20,
                    'C0':-2*max_C,
                    'D0':-2*max_D,
                    'acrophase0':-np.pi}                      
                      
    max_bounds =   {'A':abs(max(Y)), 
                    'B':abs(max(Y)), 
                    'C':max_C,
                    'D':max_D,
                    'acrophase':np.pi,
                    'A0':2*abs(max(Y)), 
                    'B0':2*abs(max(Y)), 
                    'C0':2*max_C,
                    'D0':2*max_D,
                    'acrophase0':np.pi}
    
    if not period: # if period is not specified
        min_bounds['period'] =  min_per
        max_bounds['period'] =  max_per

    if period:
        parameters = ['A', 'B', 'C', 'D', 'acrophase', 'A0', 'B0', 'C0', 'D0', 'acrophase0']
    else:
        parameters = ['A', 'B', 'C', 'D', 'acrophase', 'A0', 'B0', 'C0', 'D0', 'acrophase0', 'period']

    
    min_bounds = [min_bounds[name] for name in parameters]
    max_bounds = [max_bounds[name] for name in parameters]  

    popt1, _, _ = fit_generalized_cosinor(X1, Y1, period=period)
    popt2, _, _ = fit_generalized_cosinor(X2, Y2, period=period)
    
    per2 = popt2[-1] 

    popt1 = popt1[:-1]
    popt2 = popt2[:-1]
         
    popt2 = np.array(popt2) - np.array(popt1)
    popt2[-1] = cosinor.project_acr(popt2[-1]) # correct acr
    if period:
        p0 = list(popt1) + list(popt2)
    else:
        p0 = list(popt1) + list(popt2) + [per2]


    if period:
        popt, pcov = curve_fit(lambda x, A, B, C, D, acrophase, A0, B0, C0, D0, acrophase0: fitting_func(x, A, B, C, D, acrophase, A0, B0, C0, D0, acrophase0, period), predictor, Y, bounds=(min_bounds, max_bounds), p0=p0, **kwargs)
    else:
        popt, pcov = curve_fit(fitting_func, predictor, Y, bounds=(min_bounds, max_bounds), p0=p0, **kwargs)
        
    DoF = Y.size - len(popt)

    if not period:
        period = popt[-1]
        popt_ext = popt
    else:
        popt_ext = list(popt) + [period]

    Y_fit = fitting_func(predictor, *popt_ext)
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), period)
    statistics_params = calculate_statistics_curve_fit_parameters(popt, pcov, DoF, parameters)

    if plot:
        X_plot = np.linspace(min(X), max(X), 1000)
        Y_plot1 = fitting_func([X_plot, np.zeros(len(X_plot))], *popt_ext)
        Y_plot2 = fitting_func([X_plot, np.ones(len(X_plot))], *popt_ext)
        plt.plot(X_plot, Y_plot1, label=test1, color="black")
        plt.plot(X_plot, Y_plot2, label=test2, color="red")
        plt.plot(X1, Y1, 'o', markersize=1, color="black")
        plt.plot(X2, Y2, 'o', markersize=1, color="red")
        if plot_margins:
            lower1 = Y_plot1 - statistics['ME']
            upper1 = Y_plot1 + statistics['ME']
            plt.fill_between(X_plot, lower1, upper1, color="black", alpha=0.1)

            lower2 = Y_plot2 - statistics['ME']
            upper2 = Y_plot2 + statistics['ME']
            plt.fill_between(X_plot, lower2, upper2, color="red", alpha=0.1)

        plt.legend()

        plt.axis([min(X), max(X), 0.9*min(min(Y), min(min(Y_plot1), min(Y_plot2))), 1.1*max(max(Y), max(max(Y_plot1), max(Y_plot2)))])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{test1} vs. {test2}")

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()


    return popt_ext, statistics, statistics_params 


###################################
# EVALUATION OF PARAMS STATISTICS #
###################################

def eval_params_n_comp_bootstrap(X,Y, n_components = 1, period = 24, rhythm_params = {}, bootstrap_size=1000, bootstrap_type="std", t_test=True, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], **kwargs):       #parameters_to_analyse = ['amplitude', 'acrophase', 'mesor', 'amplification', 'lin_comp']
    # generate and evaluate bootstrap samples
    params_bs = {}
    for param in parameters_to_analyse:
        params_bs[param] = np.zeros(bootstrap_size)
    
    idxs = np.arange(len(X))

    if not rhythm_params:
        _, _, _, rhythm_params = fit_generalized_cosinor_n_comp(X, Y, period=period, n_components = n_components, plot=False, **kwargs)

    Y = np.array(Y)
    X = np.array(X)

    for i in range(bootstrap_size):        
        idxs_bs = np.random.choice(idxs, len(idxs), replace=True)
        Y_bs, X_bs = Y[idxs_bs], X[idxs_bs]

        try:
            popt_ext_bs, _, _, rhythm_params_bs = fit_generalized_cosinor_n_comp(X_bs, Y_bs, period=period, n_components = n_components, plot=False, **kwargs)            
            for param in parameters_to_analyse:
                params_bs[param][i] = rhythm_params_bs[param]    
        except:            
            for param in parameters_to_analyse:
                params_bs[param][i] = np.nan
       
    n_params = len(popt_ext_bs)
    if period:
        n_params -= 1

    # analyse bootstrap samples
    DoF = bootstrap_size - n_params    
    rhythm_params['DoF'] = DoF

    for param in parameters_to_analyse:
        if param in parameters_angular:
            angular = True
        else:
            angular = False
    
        sample_bs = params_bs[param]
        mean, p_val, CI = cosinor.bootstrap_statistics(sample_bs, angular=angular, bootstrap_type = bootstrap_type, t_test= t_test, n_params=n_params)

        rhythm_params[f'{param}_bootstrap'] = mean
        rhythm_params[f'CI({param})'] = CI
        rhythm_params[f'p({param})'] = p_val
    
    return rhythm_params


########################
# COMPARATIVE ANALYSIS #
########################

def compare_pairs_n_comp_basic(X1, Y1, X2, Y2, n_components= 1, n_components2 = None, period1 = 24, period2 = 24, t_test=True, plot=False, save_to = "", test1="test1", test2="test2", x_label="time [h]", y_label="measurements", **kwargs):       
    n_components1 = n_components
    if not n_components2:
        n_components2 = n_components1

       
    popt_ext1, _, statistics_params1, _ = fit_generalized_cosinor_n_comp(X1,Y1, period=period1, n_components=n_components1, plot=plot, hold_on=True, color="black", **kwargs)
    popt_ext2, _, statistics_params2, _ = fit_generalized_cosinor_n_comp(X2,Y2, period=period2, n_components=n_components2, plot=plot, hold_on=True, color="red", **kwargs)
    if plot:
        plt.legend([test1, test2])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{test1} vs. {test2}")

        _, _, min_Y, max_Y = plt.axis()
        plt.axis([min(min(X1), min(X2)), max(max(X1), max(X2)), min_Y, max_Y])

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
    

    params1 = statistics_params1['params']
    params2 = statistics_params2['params']
    params1_CIs = statistics_params1['CIs']
    params2_CIs = statistics_params2['CIs']
    
    n_params1 = len(popt_ext1)
    n_params2 = len(popt_ext2)
    n_params = n_params1 + n_params2

    n1 = len(X1)
    n2 = len(X2)
    n = n1 + n2

    DoF1 = n1 - n_params1
    DoF2 = n2 - n_params2
    DoF = n - n_params

    if t_test:
        n_devs = abs(stats.t.ppf(0.05/2,df=DoF))
    else:
        n_devs = 1.96

    parameters_to_analyse = ['A', 'C', 'D'] # A: baseline, C: amplification, D: lin_comp

    d_params = {}
    p_values = {}
    CIs = {}
    for param in parameters_to_analyse:
        d_param = params2[param] - params1[param]
        
        CI1 = params1_CIs[param]
        CI2 = params2_CIs[param]

        se_param = cosinor.get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = t_test, angular=False, CI_type = 'se', n1 = n1, n2 = n2, DoF = DoF) 
        CI_param =  [d_param - n_devs*se_param, d_param + n_devs*se_param] 
        p_value = cosinor.get_p_t_test(d_param, se_param, DoF)

        d_params[f'd_{param}'] = d_param
        CIs[f'd_{param}'] = CI_param
        p_values[f'd_{param}'] = p_value

    if not period1:
        d_params['period1'] = params1['period']
    else:
        d_params['period1'] = period1

    if not period2:
        d_params['period2'] = params2['period']
    else:
        d_params['period2'] = period2


    return {'params': d_params, 'CIs': CIs, 'p_values': p_values}

def population_compare_pairs_n_comp_basic(df_pop1, df_pop2, n_components= 1, n_components2 = None, period1 = 24, period2 = 24, t_test=True, plot=False, save_to = "", test1="test1", test2="test2", x_label="time [h]", y_label="measurements", **kwargs):       
    n_components1 = n_components
    if not n_components2:
        n_components2 = n_components1


    _, statistics_params1, rhythm_params1 = population_fit_generalized_cosinor_n_comp(df_pop1, period = period1, n_components=n_components1, plot=plot, hold_on=True, color="black", **kwargs)
    _, statistics_params2, rhythm_params2 = population_fit_generalized_cosinor_n_comp(df_pop2, period = period2, n_components=n_components2, plot=plot, hold_on=True, color="red", **kwargs)

    if plot:
        plt.legend([test1, test2])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{test1} vs. {test2}")

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()
    

    params1 = statistics_params1['params']
    params2 = statistics_params2['params']
    params1_CIs = statistics_params1['CIs']
    params2_CIs = statistics_params2['CIs']
    
    """
    n_params1 = len(statistics_params1['params'])
    n_params2 = len(statistics_params2['params'])
    n_params = n_params1 + n_params2
    """

    n1 = len(df_pop1.test.unique())
    n2 = len(df_pop2.test.unique())
    n = n1 + n2

    DoF1 = n1 - 1
    DoF2 = n2 - 1
    DoF = n - 2

    if t_test:
        n_devs = abs(stats.t.ppf(0.05/2,df=DoF))
    else:
        n_devs = 1.96

    parameters_to_analyse = ['A', 'C', 'D'] # A: baseline, C: amplification, D: lin_comp

    d_params = {}
    p_values = {}
    CIs = {}
    for param in parameters_to_analyse:
        d_param = params2[param] - params1[param]
        
        CI1 = params1_CIs[param]
        CI2 = params2_CIs[param]

        se_param = cosinor.get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = t_test, angular=False, CI_type = 'se', n1 = n1, n2 = n2, DoF = DoF) 
        CI_param =  [d_param - n_devs*se_param, d_param + n_devs*se_param] 
        p_value = cosinor.get_p_t_test(d_param, se_param, DoF)

        d_params[f'd_{param}'] = d_param
        CIs[f'd_{param}'] = CI_param
        p_values[f'd_{param}'] = p_value

    if not period1:
        d_params['period1'] = params1['period']
    else:
        d_params['period1'] = period1

    if not period2:
        d_params['period2'] = params2['period']
    else:
        d_params['period2'] = period2

    rhythm_params = {}
    rhythm_params['d_amplitude'] = rhythm_params2['amplitude'] - rhythm_params1['amplitude']
    rhythm_params['d_acrophase'] = cosinor.project_acr(rhythm_params2['acrophase'] - rhythm_params1['acrophase'])
    rhythm_params['d_mesor'] = cosinor.project_acr(rhythm_params2['mesor'] - rhythm_params1['mesor'])
    

    return {'params': d_params, 'CIs': CIs, 'p_values': p_values, 'rhythm_params': rhythm_params}

def compare_pairs_n_comp_bootstrap(X1, Y1, X2, Y2, n_components= 1, n_components2 = None, period = 24, period2 = 24, rhythm_params1 = {}, rhythm_params2 = {}, parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], bootstrap_size = 1000, t_test=True, **kwargs): #parameters_to_analyse = ['amplitude', 'acrophase', 'mesor', 'amplification', 'lin_comp']       
    n_components1 = n_components
    if not n_components2:
        n_components2 = n_components1

    period1 = period

    if not rhythm_params1:
        rhythm_params1 = eval_params_n_comp_bootstrap(X1, Y1, n_components1, period1, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, bootstrap_size=bootstrap_size, t_test = t_test, **kwargs)

    if not rhythm_params2:
        rhythm_params2 = eval_params_n_comp_bootstrap(X2, Y2, n_components2, period2, parameters_to_analyse=parameters_to_analyse, parameters_angular=parameters_angular, bootstrap_size=bootstrap_size, t_test = t_test, **kwargs)
    

    n1 = bootstrap_size
    n2 = bootstrap_size
    n = n1 + n2
    n_params1 = 3 + n_components1*2
    if not period1:
        n_params1 += 1
    n_params2 = 3 + n_components2*2
    if not period2:
        n_params2 += 1

    n_params = n_params1 + n_params2
    DoF1 = n1 - n_params1
    DoF2 = n2 - n_params2
    DoF = n - n_params
    if t_test:
        n_devs = abs(stats.t.ppf(0.05/2,df=DoF))
    else:
        n_devs = 1.96

    d_params = {}
    p_values = {}
    CIs = {}
    for param in parameters_to_analyse:
        d_param = rhythm_params2[param] - rhythm_params1[param]   
        if param in parameters_angular:
            d_param = cosinor.project_acr(d_param)
            angular = True
        else:
            angular = False
        
        CI1 = rhythm_params1[f'CI({param})']
        CI2 = rhythm_params2[f'CI({param})']

        se_param = cosinor.get_se_diff_from_CIs(CI1, CI2, DoF1, DoF2, t_test = t_test, angular=angular, CI_type = 'se', n1 = n1, n2 = n2, DoF = DoF) 
        CI_param =  [d_param - n_devs*se_param, d_param + n_devs*se_param] 
        p_value = cosinor.get_p_t_test(d_param, se_param, DoF)

        d_params[f'd_{param}'] = d_param
        CIs[f'd_{param}'] = CI_param
        p_values[f'd_{param}'] = p_value

    return {'params': d_params, 'CIs': CIs, 'p_values': p_values}

def population_fit_generalized_cosinor_compare_independent(df, pop1, pop2, period1=24, period2 = 24, plot=False, save_to = "", x_label="time [h]", y_label="measurements",**kwargs):#, min_per = 12, max_per=36, plot=False, plot_margins=True, test1 = "test1", test2 = "test2", exp=False):
   
  

    df_pop1 = df[df.test.str.startswith(pop1)]
    df_pop2 = df[df.test.str.startswith(pop2)]
    
    statistics_params1 = population_fit_generalized_cosinor(df_pop1, period=period1, hold_on=True, plot=plot, color='black', **kwargs)
    statistics_params2 = population_fit_generalized_cosinor(df_pop2, period=period2, hold_on=True, plot=plot, color='red', **kwargs)

    n1 = len(df_pop1.test.unique())
    n2 = len(df_pop2.test.unique())
    DoF1 = n1 - 1
    DoF2 = n2 - 1
    DoF = n1 + n2 - 2
    n_devs = abs(stats.t.ppf(0.05/2,df=DoF))  

    CI1 = statistics_params1['CIs']
    CI2 = statistics_params2['CIs']
    params1 = statistics_params1['params']
    params2 = statistics_params2['params']
    if not period1 or not period2: # if one of the periods is not defined
        if 'period' not in CI1:
            CI1['period'] = [period1, period1]
            params1['period'] = period1          
        if 'period' not in CI2:
            CI2['period'] = [period2, period2]
            params2['period'] = period2
    
    if not period1:
        period1 = params1['period']

    if not period2:
        period2 = params2['period']


    d_params = {}
    p_values = {}
    CIs = {}

    d_params['period1'] = period1
    d_params['period2'] = period2
    
    for param in params1:
        d_param = params2[param] - params1[param]   
        if param == 'acrophase':
            d_param = cosinor.project_acr(d_param)
            angular = True
        else:
            angular = False
        
        se_param = cosinor.get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = True, angular=angular, CI_type = 'se', n1 = n1, n2 = n2, DoF = DoF) 
        CI_param =  [d_param - n_devs*se_param, d_param + n_devs*se_param] 
        p_value = cosinor.get_p_t_test(d_param, se_param, DoF)

        d_params[f'd_{param}'] = d_param
        CIs[f'd_{param}'] = CI_param
        p_values[f'd_{param}'] = p_value

    if plot:
        plt.legend([pop1, pop2])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{pop1} vs. {pop2}")

        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()



    return {'params': d_params, 'CIs': CIs, 'p_values': p_values}

def fit_generalized_cosinor_compare_independent(X1, Y1, X2, Y2, period1=24, period2 = 24, test1="test1", test2="test2", plot=False, save_to = "", x_label="time [h]", y_label="measurements", **kwargs):#, min_per = 12, max_per=36, plot=False, plot_margins=True, test1 = "test1", test2 = "test2", exp=False):
   
    popt1, _, statistics_params1 = fit_generalized_cosinor(X1, Y1, period=period1, plot=plot, color='black', hold_on = True, **kwargs)
    popt2, _, statistics_params2 = fit_generalized_cosinor(X2, Y2, period=period2, plot=plot, color='red', hold_on = True, **kwargs)

    n1 = len(X1)
    n2 = len(X2)
    n = n1 + n2
    n_params1 = len(popt1)
    n_params2 = len(popt2)
    n_params = n_params1 + n_params2
    DoF1 = n1 - n_params1
    DoF2 = n2 - n_params2
    DoF = n - n_params
    n_devs = abs(stats.t.ppf(0.05/2,df=DoF))

    CI1 = statistics_params1['CIs']
    CI2 = statistics_params2['CIs']
    params1 = statistics_params1['params']
    params2 = statistics_params2['params']
    if not period1 or not period2: # if one of the periods is not defined
        if 'period' not in CI1:
            CI1['period'] = [period1, period1]
            params1['period'] = period1
        if 'period' not in CI2:
            CI2['period'] = [period2, period2]
            params2['period'] = period2

    if not period1:
        period1 = params1['period']

    if not period2:
        period2 = params2['period']


    d_params = {}
    p_values = {}
    CIs = {}

    d_params['period1'] = period1
    d_params['period2'] = period2

    for param in params1:
        d_param = params2[param] - params1[param]   
        if param == 'acrophase':
            d_param = cosinor.project_acr(d_param)
            angular = True
        else:
            angular = False
        
        se_param = cosinor.get_se_diff_from_CIs(CI1[param], CI2[param], DoF1, DoF2, t_test = True, angular=angular, CI_type = 'se', n1 = n1, n2 = n2, DoF = DoF) 
        CI_param =  [d_param - n_devs*se_param, d_param + n_devs*se_param] 
        p_value = cosinor.get_p_t_test(d_param, se_param, DoF)

        d_params[f'd_{param}'] = d_param
        CIs[f'd_{param}'] = CI_param
        p_values[f'd_{param}'] = p_value

    if plot:
        plt.legend([test1, test2])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{test1} vs. {test2}")

        _, _, min_Y, max_Y = plt.axis()
        plt.axis([min(min(X1), min(X2)), max(max(X1), max(X2)), min_Y, max_Y])


        if save_to:
            plt.savefig(save_to+'.png')
            plt.savefig(save_to+'.pdf')
            plt.close()
        else:
            plt.show()


    return {'params': d_params, 'CIs': CIs, 'p_values': p_values}

###########
# HELPERS #
###########

def get_best_model(X, Y, period=24, n_components = [1,2,3], plot=False, plot_margins=True, save_to = "", **kwargs):
    
    popt_ext1, statistics1, statistics_params1, rhythm_params1 = fit_generalized_cosinor_n_comp(X, Y, period = period, n_components=n_components[0], plot=False, **kwargs)
    RSS1 = statistics1['RSS']
    n_params1 = len(popt_ext1)
    if period:
        n_params1 -= 1
    DoF1 = len(X) - n_params1
    best_comps = n_components[0]

    #print(RSS1)

    for n_comps in n_components[1:]:
        try:
            popt_ext2, statistics2, statistics_params2, rhythm_params2 = fit_generalized_cosinor_n_comp(X, Y, period = period, n_components=n_comps, plot=False, **kwargs)
        except:
            continue

        RSS2 = statistics2['RSS']
        #print(RSS2)
        n_params2 = len(popt_ext2)
        if period:  
            n_params2 -= 1
        DoF2 = len(X) - n_params2

        if cosinor.compare_models(RSS1, RSS2, DoF1, DoF2) < 0.05:
            RSS1 = RSS2
            DoF1 = DoF2
            n_params1 = n_params2
            popt_ext1, statistics1, statistics_params1, rhythm_params1 = popt_ext2, statistics2, statistics_params2, rhythm_params2
            best_comps = n_comps
        
    if plot:
        fit_generalized_cosinor_n_comp(X, Y, period = period, n_components=best_comps, plot=True, plot_margins=plot_margins, save_to = save_to, **kwargs)



    return best_comps, popt_ext1, statistics1, statistics_params1, rhythm_params1

def get_best_model_population(df_pop, period=24, n_components = [1,2,3], plot=False, plot_margins=True, save_to = "", **kwargs):
    
    n_points = len(df_pop.x.values)

    statistics1, p_dict1, rhythm_params1 = population_fit_generalized_cosinor_n_comp(df_pop, period = period, n_components=n_components[0], plot=False, **kwargs)
    RSS1 = p_dict1['RSS']     
    n_params1 = 3 + n_components[0]*2
    if not period:      
        n_params1 += 1
    DoF1 = n_points - n_params1
    best_comps = n_components[0]

    for n_comps in n_components[1:]:
        try:
            statistics2, p_dict2, rhythm_params2 = population_fit_generalized_cosinor_n_comp(df_pop, period = period, n_components=n_comps, plot=False, **kwargs)
        except:            
            continue
        RSS2 = p_dict2['RSS']
        n_params2 = 3 + n_comps*2
        if not period:
            n_params2 += 1        
        DoF2 = n_points - n_params2

        #print(best_comps, "vs", n_comps)
        #print("RSS1",RSS1)
        #print("RSS2",RSS2)
        #print("DoF1",DoF1)
        #print("DoF2",DoF2)
        #print("p-val", cosinor.compare_models(RSS1, RSS2, DoF1, DoF2))

        if cosinor.compare_models(RSS1, RSS2, DoF1, DoF2) < 0.05:
            RSS1 = RSS2
            DoF1 = DoF2
            n_params1 = n_params2
            statistics1, p_dict1, rhythm_params1 = statistics2, p_dict2, rhythm_params2
            best_comps = n_comps
        
        
    if plot:
        population_fit_generalized_cosinor_n_comp(df_pop, period = period, n_components=best_comps, plot=True, plot_margins = plot_margins, save_to = save_to, **kwargs)


    return best_comps, statistics1, p_dict1, rhythm_params1

############
# WRAPPERS #
############

# if the period is set to 0 it is evaluated in the regression process
def fit_generalized_cosinor_group(df, period=24, folder="", **kwargs):

    columns = ['test', 'period', 'p', 'q', 'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 'acrophase', 'p(acrophase)','q(acrophase)', 'CI(acrophase)', 'amplification', 'p(amplification)', 'q(amplification)', 'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    tests = df.test.unique()
    for test in tests:   
        X = df[df.test == test].x.values
        Y = df[df.test == test].y.values
        
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''
        
        res = fit_generalized_cosinor(X, Y, period = period, save_to = save_to, test=test, **kwargs)         
        try:
            _, stats, stats_params = res
        except:
            continue

        p = stats['p']
        #p_reject = stats['p_reject']
        if period:
            per = period
        else:
            per = stats_params['params']['period']

     


        amplitude, p_amplitude, CI_amplitude = stats_params['params']['B'], stats_params['p_values']['B'], stats_params['CIs']['B']
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']
        acrophase, p_acrophase, CI_acrophase = stats_params['params']['acrophase'], stats_params['p_values']['acrophase'], stats_params['CIs']['acrophase']

        d = {'test':test, 'period':per, 'p':p, #'p_reject': p_reject,
             'amplitude': amplitude, 'p(amplitude)':p_amplitude, 'CI(amplitude)':CI_amplitude,
             'amplification': amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp': lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp,
             'acrophase': acrophase, 'p(acrophase)':p_acrophase, 'CI(acrophase)':CI_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    #df_results["q_reject"] = multi.multipletests(df_results["p_reject"], method = 'fdr_bh')[1]
    df_results["q(amplitude)"] = multi.multipletests(df_results["p(amplitude)"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(acrophase)"] = multi.multipletests(df_results["p(acrophase)"], method = 'fdr_bh')[1]
    
    return df_results

def population_fit_generalized_cosinor_group(df, period=24, folder="", **kwargs):
    columns = ['test', 'period', 'p', 'q', 'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 'acrophase', 'p(acrophase)','q(acrophase)', 'CI(acrophase)', 'amplification', 'p(amplification)', 'q(amplification)', 'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    tests = df.test.unique()
    tests = {t.split("_")[0] for t in tests}
    
    for test in tests:
        df_pop = df[df.test.str.startswith(test)]    
        
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''
        
        stats_params = population_fit_generalized_cosinor(df_pop, period = period, save_to = save_to, test=test, **kwargs)
        
        p = stats_params['statistics']['p']
        #p_reject = stats_params['statistics']['p_reject']
        if period:
            per = period
        else:
            per = stats_params['params']['period']

     
        amplitude, p_amplitude, CI_amplitude = stats_params['params']['B'], stats_params['p_values']['B'], stats_params['CIs']['B']
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']
        acrophase, p_acrophase, CI_acrophase = stats_params['params']['acrophase'], stats_params['p_values']['acrophase'], stats_params['CIs']['acrophase']

        d = {'test':test, 'period':per, 'p':p, #'p_reject': p_reject,
             'amplitude': amplitude, 'p(amplitude)':p_amplitude, 'CI(amplitude)':CI_amplitude,
             'amplification': amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp': lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp,
             'acrophase': acrophase, 'p(acrophase)':p_acrophase, 'CI(acrophase)':CI_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    #df_results["q_reject"] = multi.multipletests(df_results["p_reject"], method = 'fdr_bh')[1]
    df_results["q(amplitude)"] = multi.multipletests(df_results["p(amplitude)"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(acrophase)"] = multi.multipletests(df_results["p(acrophase)"], method = 'fdr_bh')[1]
    
    return df_results

# if the period is set to 0 it is evaluated in the regression process
def fit_generalized_cosinor_compare_pairs_dependent(df, pairs, period=24, folder="", **kwargs):

    columns = ['test', 'period', 'p', 'q', 'd_amplitude', 'p(d_amplitude)', 'q(d_amplitude)', 'CI(d_amplitude)', 'd_acrophase', 'p(d_acrophase)','q(d_acrophase)', 'CI(d_acrophase)', 'd_amplification', 'p(d_amplification)', 'q(d_amplification)', 'CI(d_amplification)','d_lin_comp', 'p(d_lin_comp)', 'q(d_lin_comp)', 'CI(d_lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    for test1, test2 in pairs:    

        if folder:        
            save_to = os.path.join(folder,test1+'_'+test2)
        else:
            save_to = ""

        X1 = df[df.test == test1].x.values
        Y1 = df[df.test == test1].y.values

        X2 = df[df.test == test2].x.values
        Y2 = df[df.test == test2].y.values

        _, stats, stats_params = fit_generalized_cosinor_compare(X1, Y1, X2, Y2, period=period, save_to = save_to, test1=test1, test2=test2, **kwargs)

        p = stats['p']
        #p_reject = stats['p_reject']
        if period:
            per = period
        else:
            per = stats_params['params']['period']

        d_amplitude, p_d_amplitude, CI_d_amplitude = stats_params['params']['B0'], stats_params['p_values']['B0'], stats_params['CIs']['B0']
        d_amplification, p_d_amplification, CI_d_amplification = stats_params['params']['C0'], stats_params['p_values']['C0'], stats_params['CIs']['C0']    
        d_lin_comp, p_d_lin_comp, CI_d_lin_comp = stats_params['params']['D0'], stats_params['p_values']['D0'], stats_params['CIs']['D0']
        d_acrophase, p_d_acrophase, CI_d_acrophase = stats_params['params']['acrophase0'], stats_params['p_values']['acrophase0'], stats_params['CIs']['acrophase0']

        d = {'test':test1 + ' vs. ' + test2, 'period':per, 'p':p, #'p_reject': p_reject,
             'd_amplitude': d_amplitude, 'p(d_amplitude)':p_d_amplitude, 'CI(d_amplitude)':CI_d_amplitude,
             'd_amplification': d_amplification, 'p(d_amplification)':p_d_amplification, 'CI(d_amplification)':CI_d_amplification,
             'd_lin_comp': d_lin_comp, 'p(d_lin_comp)':p_d_lin_comp, 'CI(d_lin_comp)':CI_d_lin_comp,
             'd_acrophase': d_acrophase, 'p(d_acrophase)':p_d_acrophase, 'CI(d_acrophase)':CI_d_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    #df_results["q_reject"] = multi.multipletests(df_results["p_reject"], method = 'fdr_bh')[1]
    df_results["q(d_amplitude)"] = multi.multipletests(df_results["p(d_amplitude)"], method = 'fdr_bh')[1]
    df_results["q(d_amplification)"] = multi.multipletests(df_results["p(d_amplification)"], method = 'fdr_bh')[1]
    df_results["q(d_lin_comp)"] = multi.multipletests(df_results["p(d_lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(d_acrophase)"] = multi.multipletests(df_results["p(d_acrophase)"], method = 'fdr_bh')[1]

    return df_results

# if the period is set to 0 it is evaluated in the regression process
def fit_generalized_cosinor_compare_pairs_independent(df, pairs, period1=24, period2=24, folder="", **kwargs):

    columns = ['test', 'period1', 'period2', 'd_amplitude', 'p(d_amplitude)', 'q(d_amplitude)', 'CI(d_amplitude)', 'd_acrophase', 'p(d_acrophase)','q(d_acrophase)', 'CI(d_acrophase)', 'd_amplification', 'p(d_amplification)', 'q(d_amplification)', 'CI(d_amplification)','d_lin_comp', 'p(d_lin_comp)', 'q(d_lin_comp)', 'CI(d_lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    for test1, test2 in pairs:    

        if folder:        
            save_to = os.path.join(folder,test1+'_'+test2)
        else:
            save_to = ""

        X1 = df[df.test == test1].x.values
        Y1 = df[df.test == test1].y.values

        X2 = df[df.test == test2].x.values
        Y2 = df[df.test == test2].y.values

        stats_params = fit_generalized_cosinor_compare_independent(X1, Y1, X2, Y2, period1=24, period2 = 24, save_to = save_to, test1 = test1, test2 = test2, **kwargs)

        if period1:
            per1 = period1
        else:
            per1 = stats_params['params']['period1']

        if period2:
            per2 = period2
        else:
            per2 = stats_params['params']['period2']


        d_amplitude, p_d_amplitude, CI_d_amplitude = stats_params['params']['d_B'], stats_params['p_values']['d_B'], stats_params['CIs']['d_B']
        d_amplification, p_d_amplification, CI_d_amplification = stats_params['params']['d_C'], stats_params['p_values']['d_C'], stats_params['CIs']['d_C']    
        d_lin_comp, p_d_lin_comp, CI_d_lin_comp = stats_params['params']['d_D'], stats_params['p_values']['d_D'], stats_params['CIs']['d_D']
        d_acrophase, p_d_acrophase, CI_d_acrophase = stats_params['params']['d_acrophase'], stats_params['p_values']['d_acrophase'], stats_params['CIs']['d_acrophase']

        d = {'test':test1 + ' vs. ' + test2, 'period1':per1, 'period2':per2,
             'd_amplitude': d_amplitude, 'p(d_amplitude)':p_d_amplitude, 'CI(d_amplitude)':CI_d_amplitude,
             'd_amplification': d_amplification, 'p(d_amplification)':p_d_amplification, 'CI(d_amplification)':CI_d_amplification,
             'd_lin_comp': d_lin_comp, 'p(d_lin_comp)':p_d_lin_comp, 'CI(d_lin_comp)':CI_d_lin_comp,
             'd_acrophase': d_acrophase, 'p(d_acrophase)':p_d_acrophase, 'CI(d_acrophase)':CI_d_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q(d_amplitude)"] = multi.multipletests(df_results["p(d_amplitude)"], method = 'fdr_bh')[1]
    df_results["q(d_amplification)"] = multi.multipletests(df_results["p(d_amplification)"], method = 'fdr_bh')[1]
    df_results["q(d_lin_comp)"] = multi.multipletests(df_results["p(d_lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(d_acrophase)"] = multi.multipletests(df_results["p(d_acrophase)"], method = 'fdr_bh')[1]

    return df_results



# if the period is set to 0 it is evaluated in the regression process
def population_fit_generalized_cosinor_compare_pairs(df, pairs, period1=24, period2 = 24, folder="", **kwargs):

    columns = ['test', 'period1', 'period2', 'd_amplitude', 'p(d_amplitude)', 'q(d_amplitude)', 'CI(d_amplitude)', 'd_acrophase', 'p(d_acrophase)','q(d_acrophase)', 'CI(d_acrophase)', 'd_amplification', 'p(d_amplification)', 'q(d_amplification)', 'CI(d_amplification)','d_lin_comp', 'p(d_lin_comp)', 'q(d_lin_comp)', 'CI(d_lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)


    for test1, test2 in pairs:    

        if folder:        
            save_to = os.path.join(folder,test1+'_'+test2)
        else:
            save_to = ""

        stats = population_fit_generalized_cosinor_compare_independent(df, test1, test2, period1=period1, period2 = period2, save_to = save_to, **kwargs)
        params, CIs, p_values = stats['params'], stats['CIs'], stats['p_values']
        
        if period1:
            per1 = period1
        else:
            per1 = params['period1']

        if period2:
            per2 = period2
        else:
            per2 = params['period2']


        d_amplitude, p_d_amplitude, CI_d_amplitude = params['d_B'], p_values['d_B'], CIs['d_B']
        d_amplification, p_d_amplification, CI_d_amplification = params['d_C'], p_values['d_C'], CIs['d_C']    
        d_lin_comp, p_d_lin_comp, CI_d_lin_comp = params['d_D'], p_values['d_D'], CIs['d_D']
        d_acrophase, p_d_acrophase, CI_d_acrophase = params['d_acrophase'], p_values['d_acrophase'], CIs['d_acrophase']

        d = {'test':test1 + ' vs. ' + test2, 'period1':per1, 'period2': per2, 
             'd_amplitude': d_amplitude, 'p(d_amplitude)':p_d_amplitude, 'CI(d_amplitude)':CI_d_amplitude,
             'd_amplification': d_amplification, 'p(d_amplification)':p_d_amplification, 'CI(d_amplification)':CI_d_amplification,
             'd_lin_comp': d_lin_comp, 'p(d_lin_comp)':p_d_lin_comp, 'CI(d_lin_comp)':CI_d_lin_comp,
             'd_acrophase': d_acrophase, 'p(d_acrophase)':p_d_acrophase, 'CI(d_acrophase)':CI_d_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q(d_amplitude)"] = multi.multipletests(df_results["p(d_amplitude)"], method = 'fdr_bh')[1]
    df_results["q(d_amplification)"] = multi.multipletests(df_results["p(d_amplification)"], method = 'fdr_bh')[1]
    df_results["q(d_lin_comp)"] = multi.multipletests(df_results["p(d_lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(d_acrophase)"] = multi.multipletests(df_results["p(d_acrophase)"], method = 'fdr_bh')[1]

    return df_results


# if the period is set to 0 it is evaluated in the regression process
def fit_generalized_cosinor_n_comp_group_best(df, period = 24, n_components = [1,2,3], folder = "", **kwargs):
    df_best_models = pd.DataFrame(columns = ['test', 'period', 'n_components', 
                                         'p', 'q', 'RSS', 
                                         'amplitude', 'acrophase', 'mesor', 
                                         'peaks', 'heights', 'troughs', 'heights2',
                                         'amplification', 'p(amplification)', 'q(amplification)', 
                                         'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)'], 
                              dtype=float)

    tests = df.test.unique()

    for test in tests:
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''    

        X = df[df.test == test].x.values
        Y = df[df.test == test].y.values
        best_comps, _, stats, stats_params, params = get_best_model(X,Y, period=period, n_components=n_components, save_to=save_to, test=test,**kwargs)

        if period:
            per = period
        else:
            per = stats_params['params']['period']

        n_comps = best_comps
        #p, p_reject, RSS = stats['p'], stats['p_reject'], stats['RSS']
        p, RSS = stats['p'], stats['RSS']
        amplitude, acrophase, mesor = params['amplitude'], params['acrophase'], params['mesor']
        peaks, heights, troughs, heights2, = params['peaks'], params['heights'], params['troughs'], params['heights2'],
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']

        d = {'test':test, 'period':per, 'n_components': n_comps, 'p':p, 'RSS': RSS, #'p_reject':p_reject, 
             'amplitude': amplitude, 'acrophase':acrophase, 'mesor':mesor,
             'peaks':peaks, 'heights':heights, 'troughs':troughs, 'heights2':heights2,
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_best_models = df_best_models.append(d, ignore_index=True)
        df_best_models = df_add_row(df_best_models, d)

    df_best_models["q"] = multi.multipletests(df_best_models["p"], method = 'fdr_bh')[1]
    #df_best_models["q_reject"] = multi.multipletests(df_best_models["p_reject"], method = 'fdr_bh')[1]
    df_best_models["q(amplification)"] = multi.multipletests(df_best_models["p(amplification)"], method = 'fdr_bh')[1]
    df_best_models["q(lin_comp)"] = multi.multipletests(df_best_models["p(lin_comp)"], method = 'fdr_bh')[1]
    
    return df_best_models


# if the period is set to 0 it is evaluated in the regression process
def population_fit_generalized_cosinor_n_comp_group_best(df, period = 24, n_components = [1,2,3], folder = "", **kwargs):
    df_best_models = pd.DataFrame(columns = ['test', 'period', 'n_components', 
                                         'p', 'q', 'RSS', 
                                         'amplitude', 'acrophase', 'mesor', 
                                         'peaks', 'heights', 'troughs', 'heights2',
                                         'amplification', 'p(amplification)', 'q(amplification)', 
                                         'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)'], 
                              dtype=float)

    tests = df.test.unique()
    tests = list({t.split("_")[0] for t in tests})
    tests.sort()


    for test in tests:
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''    

        df_pop = df[df.test.str.startswith(test)]          
        best_comps, stats, stats_params, params = get_best_model_population(df_pop, period=period, n_components=n_components, save_to=save_to, test=test, **kwargs)


        if period:
            per = period
        else:
            per = stats_params['params']['period']

        n_comps = best_comps
        #p, p_reject, RSS = stats['p'], stats['p_reject'], stats['RSS']
        p, RSS = stats['p'], stats['RSS']
        amplitude, acrophase, mesor = params['amplitude'], params['acrophase'], params['mesor']
        peaks, heights, troughs, heights2, = params['peaks'], params['heights'], params['troughs'], params['heights2'],
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']

        d = {'test':test, 'period':per, 'n_components': n_comps, 'p':p, 'RSS': RSS, #'p_reject':p_reject, 
             'amplitude': amplitude, 'acrophase':acrophase, 'mesor':mesor,
             'peaks':peaks, 'heights':heights, 'troughs':troughs, 'heights2':heights2,
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_best_models = df_best_models.append(d, ignore_index=True)
        df_best_models= df_add_row(df_best_models, d)

    df_best_models["q"] = multi.multipletests(df_best_models["p"], method = 'fdr_bh')[1]
    #df_best_models["q_reject"] = multi.multipletests(df_best_models["p_reject"], method = 'fdr_bh')[1]
    df_best_models["q(amplification)"] = multi.multipletests(df_best_models["p(amplification)"], method = 'fdr_bh')[1]
    df_best_models["q(lin_comp)"] = multi.multipletests(df_best_models["p(lin_comp)"], method = 'fdr_bh')[1]
    
    return df_best_models



def fit_generalized_cosinor_n_comp_group(df, period = 24, n_components = 3, folder = "", **kwargs):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 
                                         'p', 'q', 'RSS', 
                                         'amplitude', 'acrophase', 'mesor', 
                                         'peaks', 'heights', 'troughs', 'heights2',
                                         'amplification', 'p(amplification)', 'q(amplification)', 
                                         'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)'], 
                              dtype=float)

    tests = df.test.unique()

    for test in tests:
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''    

        X = df[df.test == test].x.values
        Y = df[df.test == test].y.values  
                
        res = fit_generalized_cosinor_n_comp(X, Y, period = period, n_components=n_components, save_to=save_to, test=test, **kwargs)

        try:
            _, stats, stats_params, params = res
        except:
            continue
        

        if period:
            per = period
        else:
            per = stats_params['params']['period']

                
            
            
        n_comps = n_components
        #p, p_reject, RSS = stats['p'], stats['p_reject'], stats['RSS']
        p, RSS = stats['p'], stats['RSS']
        amplitude, acrophase, mesor = params['amplitude'], params['acrophase'], params['mesor']
        peaks, heights, troughs, heights2, = params['peaks'], params['heights'], params['troughs'], params['heights2'],
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']

        d = {'test':test, 'period':per, 'n_components': n_comps, 'p':p, 'RSS': RSS, #'p_reject':p_reject, 
             'amplitude': amplitude, 'acrophase':acrophase, 'mesor':mesor,
             'peaks':peaks, 'heights':heights, 'troughs':troughs, 'heights2':heights2,
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    #df_results["q_reject"] = multi.multipletests(df_results["p_reject"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    
    
    return df_results

def population_fit_generalized_cosinor_n_comp_group(df, period = 24, n_components = 3, folder = "", **kwargs):
    df_results = pd.DataFrame(columns = ['test', 'period', 'n_components', 
                                         'p', 'q', 'RSS', 
                                         'amplitude', 'acrophase', 'mesor', 
                                         'peaks', 'heights', 'troughs', 'heights2',
                                         'amplification', 'p(amplification)', 'q(amplification)', 
                                         'CI(amplification)','lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)'], 
                              dtype=float)

    tests = df.test.unique()
    tests = list({t.split("_")[0] for t in tests})
    tests.sort()


    for test in tests:
        if folder:                    
            save_to = os.path.join(folder,test)
        else:
            save_to = ''    

        df_pop = df[df.test.str.startswith(test)]          
        try:
            stats, stats_params, params = population_fit_generalized_cosinor_n_comp(df_pop, period=period, n_components = n_components, save_to = save_to, test=test, **kwargs)
        except:
            continue

        if period:
            per = period
        else:
            per = stats_params['params']['period']

        n_comps = n_components
        #p, p_reject, RSS = stats['p'], stats['p_reject'], stats['RSS']
        p, RSS = stats['p'], stats['RSS']
        amplitude, acrophase, mesor = params['amplitude'], params['acrophase'], params['mesor']
        peaks, heights, troughs, heights2, = params['peaks'], params['heights'], params['troughs'], params['heights2'],
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']

        d = {'test':test, 'period':per, 'n_components': n_comps, 'p':p, 'RSS': RSS, #'p_reject':p_reject, 
             'amplitude': amplitude, 'acrophase':acrophase, 'mesor':mesor,
             'peaks':peaks, 'heights':heights, 'troughs':troughs, 'heights2':heights2,
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    #df_results["q_reject"] = multi.multipletests(df_results["p_reject"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    
    
    return df_results

# bootstrap using the best models
def bootstrap_generalized_cosinor_n_comp_group_best(df, df_best_models, **kwargs):


    columns = ['test', 'period', 'n_components', 'p', 'q',
               'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 
               'acrophase', 'p(acrophase)', 'q(acrophase)', 'CI(acrophase)',
               'amplification', 'p(amplification)', 'q(amplification)', 'CI(amplification)',
               'lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)']

    df_results = pd.DataFrame(columns=columns, 
                              dtype=float)

    tests = df.test.unique()

    for test in tests:
        X = df[df.test == test].x.values
        Y = df[df.test == test].y.values

        best = df_best_models[df_best_models['test'] == test]

        best_comps = int(best.iloc[0].n_components)
        best_per = best.iloc[0].period
        amp = best.iloc[0].amplitude
        acr = best.iloc[0].acrophase
        p, amplification, p_amplification, CI_amplification = best.iloc[0]['p'], best.iloc[0]['amplification'], best.iloc[0]['p(amplification)'], best.iloc[0]['CI(amplification)']
        lin_comp, p_lin_comp, CI_lin_comp = best.iloc[0]['lin_comp'], best.iloc[0]['p(lin_comp)'], best.iloc[0]['CI(lin_comp)']

        rhythm_params = eval_params_n_comp_bootstrap(X, Y, n_components=best_comps, period=best_per, **kwargs)


        d = {'test': test, 'period': best_per, 'n_components': best_comps, 'p':p,
             'amplitude': amp, 'p(amplitude)':rhythm_params['p(amplitude)'], 'CI(amplitude)':rhythm_params['CI(amplitude)'],
             'acrophase': acr, 'p(acrophase)':rhythm_params['p(acrophase)'], 'CI(acrophase)':rhythm_params['CI(acrophase)'],
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_results = df_results.append(d, ignore_index=True)    
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    df_results["q(amplitude)"] = multi.multipletests(df_results["p(amplitude)"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(acrophase)"] = multi.multipletests(df_results["p(acrophase)"], method = 'fdr_bh')[1]

    return df_results

# bootstrap using the same number of componetns for all the models
def bootstrap_generalized_cosinor_n_comp_group(df, period=24, n_components=3, amp_comp=True, lin_comp=True, **kwargs):

    columns = ['test', 'period', 'n_components', 'p', 'q', 
               'amplitude', 'p(amplitude)', 'q(amplitude)', 'CI(amplitude)', 
               'acrophase', 'p(acrophase)', 'q(acrophase)', 'CI(acrophase)',
               'amplification', 'p(amplification)', 'q(amplification)', 'CI(amplification)',
               'lin_comp', 'p(lin_comp)', 'q(lin_comp)', 'CI(lin_comp)']

    df_results = pd.DataFrame(columns=columns, 
                              dtype=float)

    tests = df.test.unique()
    for test in tests:
        X = df[df.test == test].x.values
        Y = df[df.test == test].y.values

        _, statistics, stats_params, rhythm_params = fit_generalized_cosinor_n_comp(X, Y, period = period, n_components=n_components, amp_comp=amp_comp, lin_comp=lin_comp)

        if period:
            per = period
        else:
            per = stats_params['params']['period']

        p = statistics['p']
        amp = rhythm_params['amplitude']
        acr = rhythm_params['acrophase']
        amplification, p_amplification, CI_amplification = stats_params['params']['C'], stats_params['p_values']['C'], stats_params['CIs']['C']    
        lin_comp, p_lin_comp, CI_lin_comp = stats_params['params']['D'], stats_params['p_values']['D'], stats_params['CIs']['D']

        rhythm_params = eval_params_n_comp_bootstrap(X, Y, n_components=n_components, period=per, amp_comp=amp_comp, lin_comp=lin_comp, **kwargs)


        d = {'test': test, 'period': per, 'n_components': n_components, 'p': p,
             'amplitude': amp, 'p(amplitude)':rhythm_params['p(amplitude)'], 'CI(amplitude)':rhythm_params['CI(amplitude)'],
             'acrophase': acr, 'p(acrophase)':rhythm_params['p(acrophase)'], 'CI(acrophase)':rhythm_params['CI(acrophase)'],
             'amplification':amplification, 'p(amplification)':p_amplification, 'CI(amplification)':CI_amplification,
             'lin_comp':lin_comp, 'p(lin_comp)':p_lin_comp, 'CI(lin_comp)':CI_lin_comp}

        #df_results = df_results.append(d, ignore_index=True)    
        df_results = df_add_row(df_results, d)

    df_results["q"] = multi.multipletests(df_results["p"], method = 'fdr_bh')[1]
    df_results["q(amplitude)"] = multi.multipletests(df_results["p(amplitude)"], method = 'fdr_bh')[1]
    df_results["q(amplification)"] = multi.multipletests(df_results["p(amplification)"], method = 'fdr_bh')[1]
    df_results["q(lin_comp)"] = multi.multipletests(df_results["p(lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(acrophase)"] = multi.multipletests(df_results["p(acrophase)"], method = 'fdr_bh')[1]

    return df_results

# df_best_models: if this is specified, number of components and periods are taken from here. Otherwise, n_components and period parameters are used
# df_boostrap_single: results of the basic bootstrap resutls. If this is not specified or if a certain measurement is missing in the results, bootstrap is ran on individual fits
def compare_pairs_n_comp_bootstrap_group(df, pairs, df_best_models = None, df_bootstrap_single = None, period=24, n_components=3, folder ="", bootstrap_size=1000, bootstrap_type="std", t_test=True, **kwargs):
    
    parameters_to_analyse = ['amplitude', 'acrophase']#, 'amplification', 'lin_comp']

    columns = ['test', 'period1', 'period2', 'n_components1', 'n_components2', 'd_amplitude', 'p(d_amplitude)', 'q(d_amplitude)', 'CI(d_amplitude)', 'd_acrophase', 'p(d_acrophase)','q(d_acrophase)', 'CI(d_acrophase)', 'd_amplification', 'p(d_amplification)', 'q(d_amplification)', 'CI(d_amplification)','d_lin_comp', 'p(d_lin_comp)', 'q(d_lin_comp)', 'CI(d_lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    for test1, test2 in pairs:    

        if folder:        
            save_to = os.path.join(folder,test1+'_'+test2)
        else:
            save_to = ""

        X1 = df[df.test == test1].x.values
        Y1 = df[df.test == test1].y.values

        X2 = df[df.test == test2].x.values
        Y2 = df[df.test == test2].y.values

        if type(df_best_models) != pd.DataFrame:
            n_components1 = n_components
            n_components2 = n_components
            period1 = period
            period2 = period
        else:
            best1 = df_best_models[df_best_models['test'] == test1]
            n_components1 = int(best1.iloc[0].n_components)
            period1 = best1.iloc[0].period

            best2 = df_best_models[df_best_models['test'] == test2]
            n_components2 = int(best2.iloc[0].n_components)
            period2 = best2.iloc[0].period

        stats_params = compare_pairs_n_comp_basic(X1, Y1, X2, Y2, n_components= n_components1, n_components2 = n_components2, period1 = period1, period2 = period2, save_to=save_to, test1 = test1, test2 = test2, **kwargs)

        if not period:
            period1 = stats_params['params']['period1']
            period2 = stats_params['params']['period2']
        
        if type(df_bootstrap_single) != pd.DataFrame:
            rhythm_params1 = {}
            rhythm_params2 = {}
        else:
            rhythm_params1 = {}
            rhythm_params2 = {}

            rp1 = df_bootstrap_single[df_bootstrap_single['test'] == test1].iloc[0]
            rp2 = df_bootstrap_single[df_bootstrap_single['test'] == test2].iloc[0]

            try:
                for param in parameters_to_analyse:
                    rhythm_params1[f'CI({param})'] = rp1[f'CI({param})']
                    rhythm_params1[param] = rp1[param]
            except:
                rhythm_params1 = {}

            try:
                for param in parameters_to_analyse:
                    rhythm_params2[f'CI({param})'] = rp2[f'CI({param})']    
                    rhythm_params2[param] = rp2[param]
            except:
                rhythm_params2 = {}

        rhythm_params = compare_pairs_n_comp_bootstrap(X1, Y1, X2, Y2, n_components=n_components1, n_components2=n_components2, period=period1, period2=period2, rhythm_params1=rhythm_params1, rhythm_params2=rhythm_params2, parameters_to_analyse=parameters_to_analyse, bootstrap_size=bootstrap_size, bootstrap_type=bootstrap_type, t_test=t_test)

        d_amplitude, p_d_amplitude, CI_d_amplitude = rhythm_params['params']['d_amplitude'], rhythm_params['p_values']['d_amplitude'], rhythm_params['CIs']['d_amplitude']
        d_acrophase, p_d_acrophase, CI_d_acrophase = rhythm_params['params']['d_acrophase'], rhythm_params['p_values']['d_acrophase'], rhythm_params['CIs']['d_acrophase']

        if 'amplification' in parameters_to_analyse: # if bootstrap should be used for this parameter
            d_amplification, p_d_amplification, CI_d_amplification = rhythm_params['params']['d_amplification'], rhythm_params['p_values']['d_amplification'], rhythm_params['CIs']['d_amplification']    
        else:
            d_amplification, p_d_amplification, CI_d_amplification = stats_params['params']['d_C'], stats_params['p_values']['d_C'], stats_params['CIs']['d_C']    

        if 'lin_comp' in parameters_to_analyse: # if bootstrap should be used for this parameter
            d_lin_comp, p_d_lin_comp, CI_d_lin_comp = rhythm_params['params']['d_lin_comp'], rhythm_params['p_values']['d_lin_comp'], rhythm_params['CIs']['d_lin_comp']
        else:
            d_lin_comp, p_d_lin_comp, CI_d_lin_comp = stats_params['params']['d_D'], stats_params['p_values']['d_D'], stats_params['CIs']['d_D']


        d = {'test':test1 + ' vs. ' + test2, 
             'period1':period1, 'period2':period2,
             'n_components1':n_components1, 'n_components2':n_components2,
             'd_amplitude': d_amplitude, 'p(d_amplitude)':p_d_amplitude, 'CI(d_amplitude)':CI_d_amplitude,
             'd_amplification': d_amplification, 'p(d_amplification)':p_d_amplification, 'CI(d_amplification)':CI_d_amplification,
             'd_lin_comp': d_lin_comp, 'p(d_lin_comp)':p_d_lin_comp, 'CI(d_lin_comp)':CI_d_lin_comp,
             'd_acrophase': d_acrophase, 'p(d_acrophase)':p_d_acrophase, 'CI(d_acrophase)':CI_d_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q(d_amplitude)"] = multi.multipletests(df_results["p(d_amplitude)"], method = 'fdr_bh')[1]
    df_results["q(d_amplification)"] = multi.multipletests(df_results["p(d_amplification)"], method = 'fdr_bh')[1]
    df_results["q(d_lin_comp)"] = multi.multipletests(df_results["p(d_lin_comp)"], method = 'fdr_bh')[1]
    df_results["q(d_acrophase)"] = multi.multipletests(df_results["p(d_acrophase)"], method = 'fdr_bh')[1]
    
    return df_results


# df_best_models: if this is specified, number of components and periods are taken from here. Otherwise, n_components and period parameters are used

def population_compare_pairs_n_comp_group(df, pairs, df_best_models = None, period=24, n_components=3, folder ="", bootstrap_size=1000, bootstrap_type="std", t_test=True, **kwargs):
    
    columns = ['test', 'period1', 'period2', 'n_components1', 'n_components2', 'd_amplitude', 'd_acrophase', 'd_amplification', 'p(d_amplification)', 'q(d_amplification)', 'CI(d_amplification)','d_lin_comp', 'p(d_lin_comp)', 'q(d_lin_comp)', 'CI(d_lin_comp)']
    df_results = pd.DataFrame(columns=columns, dtype=float)

    for test1, test2 in pairs:    

        if folder:        
            save_to = os.path.join(folder,test1+'_'+test2)
        else:
            save_to = ""

        df_pop1 = df[df.test.str.startswith(test1)]
        df_pop2 = df[df.test.str.startswith(test2)]

        if type(df_best_models) != pd.DataFrame:
            n_components1 = n_components
            n_components2 = n_components
            period1 = period
            period2 = period
        else:
            best1 = df_best_models[df_best_models['test'] == test1]
            n_components1 = int(best1.iloc[0].n_components)
            period1 = best1.iloc[0].period

            best2 = df_best_models[df_best_models['test'] == test2]
            n_components2 = int(best2.iloc[0].n_components)
            period2 = best2.iloc[0].period

        stats_params =  population_compare_pairs_n_comp_basic(df_pop1, df_pop2, n_components= n_components1, n_components2 = n_components2, period1 = period1, period2 = period2, save_to = save_to, test1=test1, test2=test2, **kwargs)
        
        if not period:
            period1 = stats_params['params']['period1']
            period2 = stats_params['params']['period2']

        d_amplification, p_d_amplification, CI_d_amplification = stats_params['params']['d_C'], stats_params['p_values']['d_C'], stats_params['CIs']['d_C']    
        d_lin_comp, p_d_lin_comp, CI_d_lin_comp = stats_params['params']['d_D'], stats_params['p_values']['d_D'], stats_params['CIs']['d_D']

        d_amplitude = stats_params['rhythm_params']['d_amplitude']
        d_acrophase = stats_params['rhythm_params']['d_acrophase']

        d = {'test':test1 + ' vs. ' + test2, 
             'period1':period1, 'period2':period2,
             'n_components1':n_components1, 'n_components2':n_components2,
             'd_amplitude': d_amplitude, 
             'd_amplification': d_amplification, 'p(d_amplification)':p_d_amplification, 'CI(d_amplification)':CI_d_amplification,
             'd_lin_comp': d_lin_comp, 'p(d_lin_comp)':p_d_lin_comp, 'CI(d_lin_comp)':CI_d_lin_comp,
             'd_acrophase': d_acrophase}

        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    df_results["q(d_amplification)"] = multi.multipletests(df_results["p(d_amplification)"], method = 'fdr_bh')[1]
    df_results["q(d_lin_comp)"] = multi.multipletests(df_results["p(d_lin_comp)"], method = 'fdr_bh')[1]
    
    return df_results


    

###################################
###################################
###################################

def amp_phase_fit(predictor, Amp, Per, Phase, Base, dAmp, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp)*np.cos((2*np.pi*X/Per) + Phase + H * dPhase) + Base + H*dBase

def phase_fit(predictor, Amp, Per, Phase, Base, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return Amp*np.cos((2*np.pi*X/Per) + Phase + H * dPhase) + Base + H*dBase
    
def amp_fit(predictor, Amp, Per, Phase, Base, dAmp, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp)*np.cos((2*np.pi*X)/Per + Phase) + Base + H*dBase

def per_fit(predictor, Amp, Per, Phase, Base, dPer, dBase):
    X = predictor[0]
    H = predictor[1]
    return Amp*np.cos((2 * np.pi *X)/(Per+ H*dPer) + Phase) + Base + H*dBase

def all_fit(predictor, Amp, Per, Phase, Base, dAmp, dPer, dPhase, dBase):
    X = predictor[0]
    H = predictor[1]
    return (Amp + H*dAmp) * np.cos((2 * np.pi *X)/(Per + H*dPer)  + Phase + H * dPhase) + Base + H*dBase

def compare_phase_pairs(df, pairs, min_per = 18, max_per = 36, folder = '', prefix='', plot_residuals=False):
    df_results = pd.DataFrame()

    for test1, test2 in pairs: 
        if folder:
            save_to = os.path.join(folder, prefix + test1 + '-' + test2)
        else:
            save_to = ''
        
        X1, Y1 = df[df.test == test1].x.values, df[df.test == test1].y.values
        X2, Y2 = df[df.test == test2].x.values, df[df.test == test2].y.values
        
        statistics, d = compare_nonlinear(X1, Y1, X2, Y2, test1 = test1, test2 = test2, min_per = min_per, max_per=max_per, compare_phase = True, compare_period = False, compare_amplitude = False, save_to = save_to, plot_residuals=plot_residuals)
        
        d['test'] = test1 + ' vs. ' + test2
        d['p'] = statistics['p']
        d['p_reject'] = statistics['p_reject']
        d['ME'] = statistics['ME']
        d['resid_SE'] = statistics['resid_SE']
      
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)

    for v in d:
        if v.startswith('p'):
            if v == "p_reject":
                q = "q_reject"
            elif v == "p":
                q = "q"
            else:
                q = v[2:].replace(")","")
                q = "q("+str(q)+")"
            
            df_results[q] = multi.multipletests(df_results[v], method = '')[1]
    
 
    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where((columns == 'test')|(columns == 'p')|(columns == 'q')|(columns == 'p_reject')|(columns == 'q_reject')))
    columns = ['test', 'p', 'q', 'p_reject', 'q_reject'] + list(columns)
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]


def compare_nonlinear_pairs(df, pairs, min_per = 18, max_per = 36, folder = '', prefix='', plot_residuals=False):
    df_results = pd.DataFrame()

    for test1, test2 in pairs: 
        if folder:            
            save_to = os.path.join(folder, prefix + test1 + '-' + test2)
        else:
            save_to = ''
        
        X1, Y1 = df[df.test == test1].x.values, df[df.test == test1].y.values
        X2, Y2 = df[df.test == test2].x.values, df[df.test == test2].y.values
        
        statistics, d = compare_nonlinear(X1, Y1, X2, Y2, test1 = test1, test2 = test2, min_per = min_per, max_per=max_per, compare_phase = True, compare_period = False, compare_amplitude = True, save_to = save_to, plot_residuals=plot_residuals)
        
        d['test'] = test1 + ' vs. ' + test2
        d['p'] = statistics['p']
        d['p_reject'] = statistics['p_reject']
        d['ME'] = statistics['ME']
        d['resid_SE'] = statistics['resid_SE']
     
        #df_results = df_results.append(d, ignore_index=True)
        df_results = df_add_row(df_results, d)
  
    for v in d:
        if v.startswith('p'):
            if v == "p_reject":
                q = "q_reject"
            elif v == "p":
                q = "q"
            else:
                q = v[2:].replace(")","")
                q = "q("+str(q)+")"
            
            df_results[q] = multi.multipletests(df_results[v], method = 'fdr_bh')[1]
    
 
    columns = df_results.columns
    columns = columns.sort_values()
    columns = np.delete(columns, np.where((columns == 'test')|(columns == 'p')|(columns == 'q')|(columns == 'p_reject')|(columns == 'q_reject')))
    columns = ['test', 'p', 'q', 'p_reject', 'q_reject'] + list(columns)
    
    df_results = df_results.reindex(columns, axis=1)
    
    return df_results

    #return multi.multipletests(P, method = 'fdr_bh')[1]


def compare_nonlinear(X1, Y1, X2, Y2, test1 = '', test2 = '', min_per = 18, max_per=36, compare_phase = False, compare_period = False, compare_amplitude = False, save_to = '', plot_residuals=False):
    H1 = np.zeros(X1.size)
    H2 = np.ones(X2.size)
    
    Y = np.concatenate((Y1, Y2))    
    
    X = np.concatenate((X1, X2))    
    H = np.concatenate((H1, H2))
    
    predictor = np.array([X,H])
    
    X_full = np.linspace(min(min(X1), min(X2)), max(max(X1), max(X2)), 100)
    H1 = np.zeros(X_full.size)
    H2 = np.ones(X_full.size)
    
    minimum_bounds = {'Amp':0, 
                      'Per':min_per, 
                      'Phase':0, 
                      'Base':0, 
                      'dAmp':0, 
                      'dPer':0, 
                      'dPhase':-np.pi, 
                      'dBase':0}
    maximum_bounds = {'Amp':max(Y), 
                      'Per':max_per, 
                      'Phase':2*np.pi, 
                      'Base':max(Y), 
                      'dAmp':max(Y), 
                      'dPer':max_per/2, 
                      'dPhase':np.pi, 
                      'dBase':max(Y)}
    
    if compare_phase and compare_amplitude:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dPhase', 'dBase']        
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]               
        popt, pcov = curve_fit(amp_phase_fit, predictor, Y, bounds=(min_bounds, max_bounds))        
        
        Y1_full = amp_phase_fit(np.array([X_full, H1]), *popt)
        Y2_full = amp_phase_fit(np.array([X_full, H2]), *popt)    
        Y_fit = amp_phase_fit(np.array([X, H]), *popt)              
    elif compare_phase:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dPhase', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(phase_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = phase_fit(np.array([X_full, H1]), *popt)
        Y2_full = phase_fit(np.array([X_full, H2]), *popt)    
        Y_fit = phase_fit(np.array([X, H]), *popt)
    elif compare_period:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dPer', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(per_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = per_fit(np.array([X_full, H1]), *popt)
        Y2_full = per_fit(np.array([X_full, H2]), *popt)    
        Y_fit = per_fit(np.array([X, H]), *popt)
    elif compare_amplitude:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(amp_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = amp_fit(np.array([X_full, H1]), *popt)
        Y2_full = amp_fit(np.array([X_full, H2]), *popt)    
        Y_fit = amp_fit(np.array([X, H]), *popt)
    else:
        parameters = ['Amp', 'Per', 'Phase', 'Base', 'dAmp', 'dPer', 'dPhase', 'dBase']
        min_bounds = [minimum_bounds[name] for name in parameters]
        max_bounds = [maximum_bounds[name] for name in parameters]   
        popt, pcov = curve_fit(all_fit, predictor, Y, bounds=(min_bounds, max_bounds))
        Y1_full = all_fit(np.array([X_full, H1]), *popt)
        Y2_full = all_fit(np.array([X_full, H2]), *popt)    
        Y_fit = all_fit(np.array([X, H]), *popt)
    
    statistics = calculate_statistics_nonlinear(X, Y, Y_fit, len(popt), popt[1])    
    
    # Compute standard errors of parameter estimates
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # http://reliawiki.com/index.php/Multiple_Linear_Regression_Analysis
    perr = np.sqrt(np.diag(pcov))
    DoF = len(Y) - len(popt)
    p=np.zeros(len(popt))
    for i in np.arange(len(perr)):
        T0 = popt[i]/perr[i]
        p[i] = 2 * (1 - stats.t.cdf(abs(T0), DoF))
    
    p_dict = {}
    for param, val, p_val in zip(parameters, popt, p):
        p_dict[param] = val
        p_dict["p("+param+")"] = p_val
    
    
    
    plt.plot(X1, Y1, 'ko', markersize=1, label = test1)
    plt.plot(X2, Y2, 'ro', markersize=1, label = test2)
   
    #Y_fit1 = Y_fit[H == 0]
    #Y_fit2 = Y_fit[H == 1]
    
    plt.plot(X_full, Y1_full, 'k', label = 'fit '+test1)    
    plt.plot(X_full, Y2_full, 'r', label = 'fit '+test2)    
      
    plt.title(test1 + ' vs. ' + test2)
    plt.xlabel('time [h]')
    plt.ylabel('measurements')
    plt.legend()
    
    if save_to:
        plt.savefig(save_to+'.png')
        plt.savefig(save_to+'.pdf')
        plt.close()
    else:
        plt.show()
    
    if plot_residuals:
        
        resid = Y-Y_fit
        sm.qqplot(resid)
        plt.title(test1 + ' vs. ' + test2)
        save_to_resid = save_to + '_resid' 
        if save_to:
            plt.savefig(save_to_resid)
            plt.close()
        else:
            plt.show()
    
    
    return statistics, p_dict
    





