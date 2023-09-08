## polars_compare

A tool to compare and find the differences between two Polars DataFrames. 


### To DO:
- [x] Linting (Ruff)
- [] strict MyPy type checking
- [x] Make into python package
- [x] Add makefile for easy linting and tests
- [x] Statistics should indicate which statistics are referencing columns
- [x] Add all statistics frame to tests
- [x] Add schema differences to schema summary
- [x] Make row examples alternate between base only and compare only so that it is more readable.
- [x] Add limit value to the examples.
- [x] Updated value differences summary so that Statistic is something that makes sense.
- Raise error and print examples if duplicates are present.
- [x] Add difference criterion.
- [] Add a count of the number of rows that have any differences to the value differences summary.
- [] Add total number of value differences to the value differences summary.
- [] Add parameter to hide column differences with 0 differences.
- [] Update report so that non differences are not displayed.
- [] Add table name labels that can replace 'base' and 'compare'.
- [] Change id_columns to be named 'join_on' and add a test that checks that abritrary join conditions work.
- [] Update code to use a config dataclass that can be passed between the class and functions.
- [x] Add license
- Test for large amounts of data
- Benchmark for different sizes of data.
- Write up docstrings
- Write up readme (with code examples)
- Publish package to pypi
- Github actions
