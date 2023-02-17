# coding: UTF-8
# This is created 2023/02/17 by Y. Shinohara
# This is lastly modified YYYY/MMD/DD by Y. Shinohara
from modules.constants import *
import numpy as np
import matplotlib.pyplot as plt
class FourierAnalysis:
    """Description for FourierAnalysis class
    Class for frequenctly used functions based on Fourier analysis
    """
    def __init__(self):
        self.var1 = None #A template for the variable

    @classmethod
    def Fourier_interpolation_1D(self, x, f, Nfiner = 1, plot_option = False, endpoint = False):
        """Description for Fourier_interpolation_1D function
        A Function is for taking first- and second-derivative of a periodic function. 
        The returned arrays can be interpolated by a finer mech by Nfiner times denser mesh by Fourier transformaion.
        The endepoint option allows us to append data to the end of arrays that is equivalent to initial data which is redundant information but useful sometime for plotting.
        """
        Nx = len(x)
        dx = x[1] - x[0]
        xmin = x[0]
        xmax = xmin + dx*Nx
        #Preparation
        array_shape = f.shape
        if (len(array_shape)) == 1:
            Naxis1 = array_shape[0]
            workfiner = np.zeros([Nfiner*Naxis1,1], dtype='complex128')
        if (len(array_shape)) == 2:
            Naxis1 = array_shape[0]
            Naxis2 = array_shape[1]
            workfiner = np.zeros([Nfiner*Naxis1,Naxis2], dtype='complex128')
        else:
            print('The array shape, '+str(array_shape)+'is not supported.')
        #FFT
        work = np.fft.fft(f, axis=0)
        #
        if (Naxis1%2 == 0):
            workfiner[:Naxis1//2,:] = work[:Naxis1//2,:]*Nfiner
            workfiner[Nfiner*Naxis1 - Naxis1//2 : Nfiner*Naxis1,:] = work[Naxis1//2 :,:]*Nfiner
        elif (Naxis1%2 == 1):
            print('The odd number Naxis1,'+str(Naxis1)+', is not supported.')
        k = np.fft.fftfreq(Nx*Nfiner)*(tpi/dx)*Nfiner
        workfiner_deriv = 0.0*workfiner
        workfiner_2ndderiv = 0.0*workfiner
        for iaxis2 in range(Naxis2):
            workfiner_deriv[:,iaxis2] = zI*k[:]*workfiner[:,iaxis2]
            workfiner_2ndderiv[:,iaxis2] = -k[:]**2*workfiner[:,iaxis2]
        if (len(array_shape)) == 1:
            Naxis1 = array_shape[0]
            workfiner = np.zeros([Nfiner*Naxis1,1], dtype='complex128')
        #
        x = np.linspace(xmin, xmax, Nx*Nfiner, endpoint = False)
        f = np.real( np.fft.ifft(workfiner, axis = 0) )
        dfpdx = np.real( np.fft.ifft(workfiner_deriv, axis = 0) )
        d2fpdx2 = np.real( np.fft.ifft(workfiner_2ndderiv, axis = 0) )
        if (endpoint):
            x = np.append(x, xmax)
            f = np.append(f, f[0:1,:], axis=0)
            dfpdx = np.append(dfpdx, dfpdx[0:1,:], axis=0)
            d2fpdx2 = np.append(d2fpdx2, d2fpdx2[0:1,:], axis=0)
        if (len(array_shape)) == 1:
            f = f[: 0]
            dfpdx = dfpdx[:, 0]
            d2fpdx2 = d2fpdx2[:, 0]
        return x, f, dfpdx, d2fpdx2
