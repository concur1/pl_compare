## polars_compare

A tool to compare and find the differences between two Polars DataFrames. 


### To DO:
- [x] Linting (Ruff)
- [x] Make into python package
- [x] Add makefile for easy linting and tests
- [x] Statistics should indicate which statistics are referencing columns
- [x] Add all statistics frame to tests
- [x] Add schema differences to schema summary
- [x] Make row examples alternate between base only and compare only so that it is more readable.
- [x] Add limit value to the examples.
- [x] Updated value differences summary so that Statistic is something that makes sense.
- [x] Publish package to pypi
- [x] Add difference criterion.
- [x] Add license
- [] Make package easy to use (i.e. so you only have to import pl_compare and then you can us pl_compare)
- [] Raise error and print examples if duplicates are present.
- [] Add a count of the number of rows that have any differences to the value differences summary.
- [] Add total number of value differences to the value differences summary.
- [] Add parameter to hide column differences with 0 differences.
- [] Update report so that non differences are not displayed.
- [] Add table name labels that can replace 'base' and 'compare'.
- [] Change id_columns to be named 'join_on' and add a test that checks that abritrary join conditions work.
- [] Update code to use a config dataclass that can be passed between the class and functions.
- [] Test for large amounts of data
- [] Benchmark for different sizes of data.
- [] Write up docstrings
- [] Write up readme (with code examples)
- [] strict MyPy type checking
- [] Github actions for testing
- [] Github actions for linting
- [] Github actions for publishing

