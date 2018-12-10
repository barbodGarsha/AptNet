# Here we will handle different errors
# this will help us to understand the problems of our neural network better
import sys

ERROR_ARRAY_SIZES = 0
ERROR_ARRAY_SIZES_MSG = "Error: array sizes don't match "


def error(e, extra_explanation=None):
    if e == ERROR_ARRAY_SIZES:
        error_arrays_sizes(extra_explanation)


def error_arrays_sizes(extra_explanation=None):
    sys.exit(ERROR_ARRAY_SIZES_MSG + ": " + extra_explanation)
