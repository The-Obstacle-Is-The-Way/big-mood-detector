## `file_parser` module
The main functions of `file_parser` module are devoted to data import/export and generation of synthetic data.

### `file_parser.read_excel(file_name, trim=False, diff=False, rescale_x=False, independent=True, remove_outliers=False)`
Reads the measurent from excel (xlsx) file. Each group of measurement (e.g., each gene) should be written in a separate sheet. First columns corresponds to time points and the second column to the measured quantities. Replications should be given in a consecutive order, i.e. first replication, seconds replication, etc. Within each replication, times should be sorted. See an example of a valid file [here](https://github.com/mmoskon/CosinorPy/blob/master/test_data/dependent_data.xlsx).

#### Parameters
* `file_name` (`str`): name of the file
* `trim` (`bool`, `default=False`): if `True` first line of the file should include the times describing the interval of the measurement to include in the analysis; other measurement are removed; if the interval is set to [-1, -1] all measurement are included in the analysis.
* `diff` (`bool`, `default=False`): if `True` differentials of the measurement are calculated (e.g. to remove the accumulation of luminescence)
* `rescale_x` (`bool`, `default=False`): if `True` timepoints with the same values within the same replicate are rescaled to an interval between the current timepoint and the next timepoint.
* `independent` (`bool`, `default=True`): if `False` replicates are stored sepparately under names `x_rep1`, `x_rep2`, etc., where `x` corresponds to the current measurement.
* `remove_outliers` (`bool`, `default=False`): if `True` removes outliers.

#### Returns
* `pandas` `DataFrame` with three columns
  * `test`: names of groups of measurements 
  * `x`: timepoints
  * `y`: measurements

### `file_parser.read_csv(file_name, (file_name, sep="\t")`
Reads the measurent from csv file as used by other methods, such as JTK_CYLCE and RAIN. The first column should include the names of groups of measurement (e.g., genes). The first row should include the data describing the timepoints and replicates. If replicates are available, they should be labeled with increasing numbers (starting with 1) and these should be separated with timepoints using the underscore symbol (`_`). See an example of a valid file [here](https://github.com/mfcovington/jtk-cycle/raw/develop/Example2_data.txt).

#### Parameters
* `file_name` (`str`): name of the file
* `sep` (`str`): separator
#### Returns
* `pandas` `DataFrame` with three columns
  * `test`: names of groups of measurements 
  * `x`: timepoints
  * `y`: measurements

### `file_parser.export(df, file_name, independent = True)`
Exports the measurements into an xlsx file.
#### Parameters
* `df` (`DataFrame`): with columns `test`, `x` and `y`
* `file_name` (`str`): name of the file
* `independent` (`bool`, `default=True`): if `False` replicates that are stored sepparately under names `x_rep1`, `x_rep2`, etc., are merged together to a single measurement group

### `file_parser.export_csv(df, file_name, independent = True)`
Exports the measurements into a csv file.
#### Parameters
* `df` (`DataFrame`): with columns `test`, `x` and `y`
* `file_name` (`str`): name of the file
* `independent` (`bool`, `default=True`): if `False` replicates that are stored sepparately under names `x_rep1`, `x_rep2`, etc., are merged together to a single measurement group

### `file_parser.export_csv(df, file_name, independent = True)`
Exports the measurements into a csv file.
#### Parameters
* `df` (`DataFrame`): with columns `test`, `x` and `y`
* `file_name` (`str`): name of the file
* `independent` (`bool`, `default=True`): if `False` replicates that are stored sepparately under names `x_rep1`, `x_rep2`, etc., are merged together to a single measurement group

### `file_parser.export_cosinor2(input_file_name, output_file_name, period = 24, trim=False, diff=False, remove_outliers=False, rescale_median = False, remove_lin_comp = False)`
Convert the xlsx file with the name `input_file_name` into an xlsx file with the name `output_file_name` that can be used in a combination with the population-mean cosinor tests implemented in the cosinor2 R package (see https://cran.r-project.org/web/packages/cosinor2/).
#### Parameters
* `input_file_name` (`str`): input file name (xlsx); see `file_parser.read_excel` for the description of required file formatting. 
* `output_file_name` (`str`): name of the output file (xlsx).
* `period` (`int`, `default=24`): presumed period; only used if `remove_lin_comp` is set to `True`.
* `trim` (`bool`, `default=False`): if `True` first line of the file should include the times describing the interval of the measurement to include in the analysis; other measurement are removed; if the interval is set to [-1, -1] all measurement are included in the analysis.
* `diff` (`bool`, `default=False`): if `True` differentials of the measurement are calculated (e.g. to remove the accumulation of luminescence)
* `remove_outliers` (`bool`, `default=False`): if `True` removes outliers.
* `rescale_median` (`bool`, `default=False`): if `True` timepoints with the same values within the same replicate are rescaled to an interval between the current timepoint and the next timepoint using the median values of each timepoint.
* `remove_lin_comp` (`bool`, `default=False`): if `True` linear component is identified and removed from the data.



### `file_parser.generate_test_data(n_components=1, period = 24, amplitudes = 0, baseline = 0, phase = 0, min_time = 0, max_time = 48, time_step = 2, replicates = 1, independent = True, name="test", noise = 0)`
Synthetic test-data generator.
#### Parameters
* `n_components` (`int`, `default=1`): number of components in the cosinor data generator
* `period` (`int`, `default=24`): period of generated data
* `amplitudes` (`list`, `default=0`): amplitudes of each cosinor, default sets amplitudes to `1, 1/2, 2/3, 1/4`.
* `baseline` (`int`, `default=0`): baseline of the cosinor
* `phase` (`int`, `default=0`): acrophase
* `min_time` (`int`, `default=0`): minimal time
* `max_time` (`int`, `default=48`): maximal time
* `time_step` (`int`, `default=2`): time between measurements
* `replicates` (`int`, `default=1`): number of replicates
* `independent` (`bool`, `default=True`): if `False` replicates are stored sepparately under names `x_rep1`, `x_rep2`, etc., where `x` corresponds to the current measurement.
* `name` (`str`, `"test"`): name of the group of measurements
* `noise` (`int`, `default=0`): noise amplitude
#### Returns
* `pandas` `DataFrame` with three columns
  * `test`: name of group of measurements 
  * `x`: timepoints
  * `y`: measurements