# coding: UTF-8
# This is created 2023/02/17 by Y. Shinohara
# This is lastly modified YYYY/MMD/DD by Y. Shinohara
import sys
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
        #Grid info
        Nx = len(x)
        dx = x[1] - x[0]
        xmin = x[0]
        xmax = xmin + dx*Nx
        
        #Preparation
        array_shape = f.shape
        if (len(array_shape)) == 1:  #For 1D-array
            Nblockaxis = 1
            Nfftaxis = array_shape[0]
            ftemp = np.zeros([Nblockaxis,Nfftaxis])
            ftemp[0, :] = 1.0*f
        elif (len(array_shape)) == 2:
            Nblockaxis = array_shape[0]
            Nfftaxis = array_shape[1]
            ftemp = 1.0*f
        else:
            print('The array shape, '+str(array_shape)+'is not supported.')
        if (Nfftaxis != Nx):
            print('ERROR:Number of space grid and array size do not match!')
            print('Check the array shape, transpose of the array might be considered.')
            sys.exit()
        workfiner = np.zeros([Nblockaxis,Nfiner*Nfftaxis], dtype='complex128')
        
        #FFT
        work = np.fft.fft(ftemp)
        #Constructing Fourier component for finer real-space mesh
        if (Nfftaxis%2 == 0):
            workfiner[:, :Nfftaxis//2] = work[:, :Nfftaxis//2]*Nfiner
            workfiner[:, Nfiner*Nfftaxis - Nfftaxis//2 : Nfiner*Nfftaxis] = work[:, Nfftaxis//2 :]*Nfiner
        elif (Nfftaxis%2 == 1):
            print('The odd number Nfftaxis,'+str(Nfftaxis)+', is not supported.')
        k = np.fft.fftfreq(Nx*Nfiner)*(tpi/dx)*Nfiner
        workfiner_deriv = 0.0*workfiner
        workfiner_2ndderiv = 0.0*workfiner
        for i in range(Nblockaxis):
            workfiner_deriv[i, :] = zI*k[:]*workfiner[i, :]
            workfiner_2ndderiv[i, :] = -k[:]**2*workfiner[i, :]
        
        #Back transform to the original grid
        x = np.linspace(xmin, xmax, Nx*Nfiner, endpoint = False)
        f = np.real( np.fft.ifft(workfiner) )
        dfpdx = np.real( np.fft.ifft(workfiner_deriv) )
        d2fpdx2 = np.real( np.fft.ifft(workfiner_2ndderiv) )
        
        #Data padding to the endpoint
        if (endpoint):
            x = np.append(x, xmax)
            f = np.append(f, f[:, 0:1], axis=1)
            dfpdx = np.append(dfpdx, dfpdx[:, 0:1], axis=1)
            d2fpdx2 = np.append(d2fpdx2, d2fpdx2[:, 0:1], axis=1)
        
        #Post process for 1D array
        if (len(array_shape)) == 1:
            f = f[0, :]
            dfpdx = dfpdx[0, :]
            d2fpdx2 = d2fpdx2[0, :]
        
        return x, f, dfpdx, d2fpdx2
