## `cosinor` module
The main functions of `cosinor` module are devoted to single- or multi-component cosinor analysis and visualization.


### `cosinor.periodogram(X, Y, per_type='per', sampling_f = '', logscale = False, name = '', save_to = '', prominent = False, max_per = 240)`
Plots the periodogram with threshold of significance marked.
#### Parameters
* `X`: iterable of timepoints 
* `Y`: iterable of measurements

### `cosinor.plot_data(df, names = [], folder = '', prefix = '')`
Plots the raw data.
#### Parameters
* `df`: pandas dataframe of measurements
* `names`: list of names to plot; if empty all names will be plotted
* `folder`: folder to which the files are stored; if empty, plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored


### `cosinor.plot_data_pairs(df, names, folder = '', prefix ='')`
Plots the raw data pairs (two groups of measurement on the same plot).
#### Parameters
* `df`: pandas dataframe of measurements
* `names`: pairs of names to plot together
* `folder`: folder to which the files are stored; if empty, plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored

### `cosinor.plot_tuples_best_models(df, df_best_models, tuples, colors = ['black', 'red'], folder = '', **kwargs)`
Plots the tests (raw data and fits) from `tuples` into the same graph using the data from `df` and models from `df_best_models`. If more than two tests are given in `tuples` colors should be specified. 


### `cosinor.plot_phases(acrs, amps, tests, period=24, colors = ("black", "red", "green", "blue"), folder = "", prefix="", legend=True, CI_acrs = [], CI_amps = [], linestyles = [], title = "", labels = [])`
Plots the phases in a polar coordinate system.
#### Parameters
* `acrs`: acrophase value or list of acrophases
* `amps`: amplitude or list of amplitudes
* `test`: list of names of tests
* `period`: period value
* `colors`: colors to use; if list number of phases to plot is larger than 4, black color is used in default setting
* `folder`: folder to which the files are stored; if empty plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored
* `legend`: plot legend?
* `CI_acrs`: list of acrophase confidence intervals to plot
* `CI_amps`: list of amplitude confidence intervals to plot
* `linestyles`: linestyles to use with each plot
* `title`: if not empty used this label as a plot title
* `labels`: labels assigned to each plot

### `cosinor.plot_heatmap(df, merge_repeats=True, z_score=True, clustermap=True, df_results=False, sort_by="p", ascending=True, xlabel='Time [h]', dropnacols=False):`
Plots a heatmap of the raw data.
#### Parameters
* `df`: pandas dataframe of measurements
* `merge_repeats`: if `True`, repeats in the same timepoint are averaged; else, all repeats are plotted
* `z_score`: if `True`, the measurements for each experiment are scaled to z-scores
* `clustermap`: if `True`, clustermap is also plotted 
* `df_results`: the results of the fitting process can also be specified and used as sorting
* `sort_by`: the column of `df_results` that should be used for sorting
* `ascending`: if `True`, sorting is performed in an ascending order
* `x_label`: label to put on the x-axis
* `dropnacols`: if `True`, columns containing `nan` will be dropped
* `folder`: folder to which the files are stored; if empty, plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored

### `cosinor.fit_me(X, Y, n_components = 2, period = 24, model_type = 'lin', lin_comp = False, alpha = 0, name = '', save_to = '', plot=True, plot_residuals=False, plot_measurements=True, plot_margins=True, return_model = False, color = False, plot_phase = True, hold=False, x_label = "", y_label = "", bootstrap=False))`
Perform the basic Cosinor regression
#### Parameters
* `X`: iterable of timepoints 
* `Y`: iterable of measurements
* `n_components`: number of cosinor components to use
* `period`: period to fit
* `lin_comp`: if `True` linear componet (y ~ x) will also be included in the model
* `model_type`: by default linear model is used; if working with count data `poisson`, `gen_poisson` (generalized Poisson) or `nb`(negative binomial) can be specified.
* `alpha`: dispersion parameter - only with `nb` models - if the parameter is omitted it is assessed using the basic Poisson model ([more](https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/#cameron))
* `name`: name to diplay
* `folder`: folder to which the files are stored; if empty plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be storedorms the fitting process on all measurement in the dataframe.
* `plot`: if False plotting is not performed
* `plot_residuals`: if True qq-plot will be made
* `plot_measurements`: if True raw measurement are plotted
* `plot_margins`: if True confidence intervals are plotted
* `return_model`: if True model will be returned
* `color`: the color to use when plotting; if False, black color will be used
* `plot_phase`: if True phase plot will be made
* `hold`: allows to plot multiple graphs in the same figure
* `x_label`, `y_label`: if specified, these labels are used in the figure
* `bootstrap`: if set to `int` it defines the number of bootstrap samples used to assess the confidence intervals of amplitude, acrophase and mesor as well as their p-values
#### Returns
* tuple which includes
  * `model`: if `return_model` was set to True
  * `statistics`:
     * `p`: significance of the model
     * `p_reject`: goodness of fit of the model (only if `model_type=='lin'`)
     * `SNR`: signal to noise ratio (only if `model_type=='lin'`)
     * `RSS`: residual sum of squares 
     * `resid_SE`: standard error of residuals (only if `model_type=='lin'`)
     * `ME`: margin of error (only if `model_type=='lin'`)
     * `count`: total sum of values within the measurements (only if `model_type!='lin'`)
  * `rhythm_params`: parameters describing the rhythmicity
  * `X_test`: timepoints for additional plotting
  * `Y_test`: measurement for additional plotting


### `cosinor.population_fit(df_pop, n_components = 2, period = 24, lin_comp= False, model_type = 'lin', plot_on = True, plot_measurements=True, plot_individuals=True, plot_margins=True, save_to = '', x_label='', y_label='', **kwargs)
Population fit (according to Cornelissen). For parameters and returns see `cosinor_fit.fit_me`


### `cosinor.fit_group(df, n_components = 2, period = 24, names = "", folder = '', prefix='', **kwargs)`
Perform the multi-component cosinor fit to the measurements in the dataframe. Calls `cosinor.fit_me`.
#### Parameters
* `df`: pandas dataframe composed of three columns labeled as:
  * `'x'`: timepoints
  * `'y'`: measurements
  * `'test'`: labels of datasets (a separate cosinor model is returned for each dataset)
* `n_components`: number of cosinor components or list with numbers of cosinor components; if list of components is given, cosinor fits will be performed on cosinor models with different numbers of cosinor components
* `period`: period or list of periods to fit
* `names`: list of groups names to fit; if empty all groups will be used
* `folder`: folder to which the files are stored; if empty plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored
##### Keyword arguments:
See arguments of `cosinor.fit_me`.

#### Returns
* dataframe with statistics and parameters of each fit

### `cosinor.population_fit_group(df, n_components = 2, period = 24, folder = '', prefix='', names = [], **kwargs)`
Perform the multi-component population-mean cosinor fit to the measurements in the dataframe. Parameters and returns are the same as in `cosinor.fit_group`. Calls `cosinor.population_fit`.

### `cosinor.get_best_fits(df_results, criterium = 'R2_adj', reverse = False, n_components = [])`
Accepts fitting results and returns the best fits for each model complexity (number of components). Returns the best fit among all fits with a given number of cosinor components. Can be used to find the periods that best fit the data for a cosinor model.
#### Parameters
* `df_results`: pandas dataframe returned by* `df`: pandas dataframe of measurements
* `criterium`: criterium for finding the best models
* `reverse`: if `True` lower is better
* `n_components`: list with numbers of cosinor components to analyse; if list is empty all components in the dataframe will be used
#### Returns
* dataframe with the best fitting model (the model with the best fitting period) for certain number of components

### `cosinor.get_best_models(df, df_models, n_components = [1,2,3], lin_comp = False, criterium='p', reverse = True)`
Find the best models for each measurement. Get the best models for all fits (accepts the models obtained with `fit_group` and calls `get_best_fits` for each experiment).
#### Parameters
* `df`: pandas dataframe of measurements
* `df_models`: pandas dataframe of fitted models as returned by `cosinor.fit_group`
* `n_components`: number of cosinor components or list with numbers of cosinor components; if list of components is given, cosinor fits will be performed on cosinor models with different numbers of cosinor components
* `lin_comp`: if `True` linear componet (y ~ x) is included in the models
* `criterium`: criterium for finding the best models
* `reverse`: if `True` lower is better
#### Returns
* dataframe with the best fitting models

### `cosinor.get_best_models_population(df, df_models, n_components = [1,2,3], lin_comp = False, criterium = 'RSS', reverse = True)`
Find the best population-mean models for each measurement. See description of `cosinor.get_best_models`.

### `cosinor.plot_df_models(df, df_models, plot_residuals=True, folder ="")`
Plots the given models with the given data.

### `cosinor.plot_df_models_population(df, df_models, plot_residuals=True, folder ="")`
Plots the given population-mean models with the given data.

### `cosinor.compare_pairs_limo(df, pairs, n_components = 3, period = 24, folder = "", prefix = "", **kwargs)`
Perform the LimoRhyde analysis of differential expression between the given pairs of measurements.
#### Parameters
* `df`: pandas dataframe of measurements
* `pairs`: list of pairs to compare
* `n_components`: number of cosinor components to be used in the model for comparison
* `period`: period to fit
* `folder`: folder to which the plots are stored; if empty plots will be displayed instead of stored
* `prefix`: prefix to the file names in which plots will be stored 
##### Keyword arguments:
See arguments of `cosinor.fit_me`.

#### Returns
* dataframe with the results of comparison. These include p values for each added parameter and p value for the F statistic - should the more complex model be accepted? In the plot, the latter value is reported.

### `cosinor.compare_pairs_best_models_limo(df, df_best_models, pairs, folder = "", prefix = "", **kwargs)`
Compares pairs in a similar manner as `cosinor.compare_pairs_limo`. Compares pairs defined in `pairs` using the data from `df` and an optimal number of components from `df_best_models`.

### `cosinor.compare_pair_df_extended(df, test1, test2, n_components = 3, period = 24, n_components2 = None, period2 = None, lin_comp = False, model_type = 'lin', alpha = 0, save_to = '', non_rhythmic = False, plot_measurements=True, plot_residuals=False, plot_margins=True, x_label = '', y_label = '', bootstrap = False)`
Compare two tests from `pair` using `n_components` cosinor with period equal to `period`. If `n_components2` or `period2` are specified, using a different number of components and/or period for the second model. 

### `cosinor.compare_nonlinear(X1, Y1, X2, Y2, test1 = '', test2 = '', min_per = 18, max_per=36, compare_phase = False, compare_period = False, compare_amplitude = False, save_to = '', plot_residuals=False)`
Perform the analysis of differential expression on the basis of non-linear cosinor model.

### `cosinor.calculate_significance_level(N, **kwargs)`
When the number of samples is large, the 0.05 significance level should be decreased. `calculate_significance_level` allows you to define a significance level in such cases

#### Parameters
* `N`: the number of samples
##### Keyword arguments:
Should include
* `n_params`: number of params in a model

or

* `n_components`: number of components in a cosinor model 

Optional arguments:

* `lin_comp`: additional linear component (`bool`)
* `return_T`: By default the function returns a significance level for the F-test used in a regression process. If `return_T` is `True`, the function returns a significance level for the T-test.
For the explanation of background and references see https://davegiles.blogspot.com/2019/10/everythings-significant-when-you-have.html
