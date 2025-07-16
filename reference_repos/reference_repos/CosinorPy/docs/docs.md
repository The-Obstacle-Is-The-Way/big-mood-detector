# CosinorPy: documentation

CosinorPy presents a python package for cosinor based rhythmomethry. It is composed of three modules:
* [`file_parser`](docs_file_parser.md): reading and writting xlsx or csv files and generating synthetic data

* [`cosinor`](docs_cosinor.md): single- or multi-component cosinor functions

* [`cosinor1`](docs_cosinor1.md): single-component cosinor specific functions

To use these three modules include the following code in your python file:

`from CosinorPy import file_parser, cosinor, cosinor1`

CosinorPy can be installed using `pip` with the command

`pip install CosinorPy`

This documentation describes the main functions for data import/export, cosinor regression and testing of differential rhythmicity.