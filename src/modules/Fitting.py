# coding: UTF-8
# This is created 2023/06/12 by Y. Shinohara
# This is lastly modified YYYY/MMD/DD by Y. Shinohara
import sys
from modules.constants import *
import numpy as np
import matplotlib.pyplot as plt
class  Fitting:
    """Description for Fitting class
    Class for frequenctly used functions for fitting.
    """
    def __init__(self):
        self.var1 = None #A template for the variable

    @classmethod
    def polynomialfit(self, x, y, norder):
        """Description for polynomialfit function
        A Function to get fitting parameters for `norder`th-order polynomial function evaluated by the data set (x,y). 
        The returned arrays are the fitting parameter by polyfit function for polval function and residual sum of squares (RSS).
        """
        mt, plist = np.polynomial.polynomial.polyfit(x, y, norder, full=True)
        residuals = plist[0]
        rank = plist[1]
        singular_values = plist[2]
        rcond = plist[3]
        return mt, residuals
    
    @classmethod
    def polynomialfunc(self, x, mt, N4fit = 100):
        """Description for polynomialfunc function
        A Function to get the fitted function by polynomialfit function.
        The returned arrays are the fitted dataset covering the original data.
        """
        xrange = np.amax(x) - np.amin(x)
        x4fit = np.linspace(np.amin(x)-0.2*xrange, np.amax(x)+0.2*xrange, N4fit)
        y4fit = np.polynomial.polynomial.polyval(x4fit, mt)
        return x4fit, y4fit
    
    @classmethod
    def polynomialerror(self, mt, x, y):
        """Description for polynomialerror function
        A Function to get error of the fitting function.
        The returned array is error array by the fitting.
        """
        error = y - np.polynomial.polynomial.polyval(x, mt)
        return error
