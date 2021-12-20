# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:35:43 2019

@author: Wenjun Zhang
"""

##############################################################################
# import packages

import os

import numpy as np
from numpy import log as ln
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

##############################################################################
# Functions for reading in data

def readInImages(imgDir, numOfImgsInEachRun, parameter, \
                trapRegion=(slice(0, 65535), slice(0, 65535)), \
                noiseRegion=(slice(0, 65535), slice(0, 65535))):
    """
    Read in images, and calculate optical depth (OD).
    It selects two regions, one with atoms, one without atoms. The latter is
    served as an estimation of noise and to be subtracted in Fourier space.

    ----------
    parameters

    imgDir: string, the directory where the images of atoms are stored.
    numOfImgsInEachRun: int, the number of images you want to include.
    trapRegion: slice, mark the position of the atoms. 
        [(xmin, xmax), (ymin, ymax)] (pixel)
    noiseRegion: slice, mark the position chosen to serve as an estimation 
        of noise. [(xmin, xmax), (ymin, ymax)] (pixel)

    -------
    returns

    atomODs: list, each element of which is a numpy.ndarray, the matrix 
        for optical depth of the region with atoms.
    atomODAvg: numpy.ndarray, the average of all images in `atomODs`.
    noiseODs: list, each element of which is a numpy.ndarray, the matrix 
        for optical depth of the region without atoms.
    noiseODAvg: numpy.ndarray, the average of all images in `noiseODs`.
    imgIndexMin, imgIndexMax: string, the minimum and maximum index of the
        images included
    """

    # get filenames and the minimum and maximum index of the images
    imgFileNames = []
    for _, _, files in os.walk(imgDir):
        # select file name
        for file in files:
            # check the extension of files
            if file.startswith("rawimg_"):
                # print whole path of files
                imgFileNames.append(imgDir + "\\" + file)
#     paraFile = open(imgDir + "\\parameters.txt", "r")
#     lines = paraFile.readlines()

#     #for line in lines[-numOfImgsInEachRun:]:
#     for line in lines[1:]:
#         if line.split()[0] == "img#":
#             raise ValueError(\
#                 "Total number of images is less than required: {}<{}".format(\
#                 len(lines)-1, numOfImgsInEachRun))
#         if float(line.split()[1]) == parameter:
#             imgFileNames.append(imgDir + "\\rawimg_" + line.split()[0])

#     #imgIndexMin = lines[-numOfImgsInEachRun].split()[0]
#     imgIndexMin = lines[1].split()[0]
#     imgIndexMax = lines[-1].split()[0] # these are strings
    imgIndexMin = 0
    imgIndexMax = 1

#     paraFile.close()

    # read in images
    atomODs = []
    noiseODs = []
    for filename in imgFileNames:
        dim = np.fromfile(filename, '>u2')[0:4]
        print(dim)
        img_temp = np.fromfile(filename, '>u2')[4:]
        print(np.shape(img_temp))
        img_temp = np.fromfile(filename, '>u2')[4:].reshape((dim[1], dim[3]))
        atomImg_WithAtom_temp = \
            img_temp[:dim[1]//2, :].astype(int)[trapRegion[::-1]]
        atomImg_WithoutAtom_temp = \
            img_temp[dim[1]//2:, :].astype(int)[trapRegion[::-1]]
        noiseImg_WithAtom_temp = \
            img_temp[:dim[1]//2, :].astype(int)[noiseRegion[::-1]]
        noiseImg_WithoutAtom_temp = \
            img_temp[dim[1]//2:, :].astype(int)[noiseRegion[::-1]]
        
        # 
        atomODs.append(-ln(atomImg_WithAtom_temp/atomImg_WithoutAtom_temp))
        noiseODs.append(-ln(noiseImg_WithAtom_temp/noiseImg_WithoutAtom_temp))
    
    atomODAvg = sum(atomODs) / len(atomODs)
    atomODAvg = np.nan_to_num(atomODAvg)
    noiseODAvg = sum(noiseODs) / len(noiseODs)
    
    return atomODs, atomODAvg, noiseODs, noiseODAvg, imgIndexMin, imgIndexMax

##############################################################################
# Functions for data process

def OD2AtomNum(OD, imgSysData=None):
    """
    Calculate surface density from optical density.
    """

    if imgSysData["ODtoAtom"] == 'beer':
        px = imgSysData["CCDPixelSize"] / imgSysData["magnification"]
        AtomNum = OD * 2*np.pi / (3*imgSysData["wavelen"]**2) * px**2
    else:
        AtomNum = OD * imgSysData["ODtoAtom"]
    return AtomNum


def __calcNPS(ODs, ODAvg, norm=False, imgSysData=None):
    """
    Calculate the noise power spectrum (NPS), from a set of images and their
    average. Note that `ODs` is considered as an ensemble, which means each
    image of `ODs` should be taken under identical conditions, and `ODAvg`
    is considered as the ensemble average.

    ODs: list, each element of which is a numpy.ndarray, the matrix of each 
        image.
    ODAvg: numpy.ndarray, the ensemble average of `ODs`.
    norm: bool. If true, use the atom number to normalize the noise power 
        spectrum. If false, use OD to calculate. In the latter case, the 
        absolute value of the noise power spectrum is meaningless.

    noisePowSpecs: list, each element of which is the noise power spectrums of
        an image in `ODs`. 
    NPS: numpy.ndarray, the average of noisePowSpecs, which is taken as 
        ensemble average
    """

    noisePowSpecs = []
    for k in range(len(ODs)):
        if len(ODs) > 1:
            noise = ODs[k] - ODAvg
        else:
            noise = ODs[k]
        if norm:
            noise = OD2AtomNum(noise, imgSysData)
        noiseFFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(noise)))
        noisePowSpec = np.abs(noiseFFT)**2
        # noisePowSpec = noisePowSpec / ODs[k].sum()
        noisePowSpecs.append(noisePowSpec)
    
    NPS = sum(noisePowSpecs) / len(noisePowSpecs)
    #NPS = NPS / ODAvg
    return NPS, noisePowSpecs


def calcNPS(imgDir, numOfImgsInEachRun, parameter, trapRegion, noiseRegion, \
    norm=False, imgSysData=None):
    """
    This is a combination of `ReadInImages` and `calcNPS__`. 
    Calculate the experimental imaging response function.

    ----------
    parameters

    same as `ReadInImages`

    -------
    returns

    M2k_Exp: numpy.ndarray, the calculated imaging response function 
        aalready subtracted by pure noise result).
    M2k_Exp_atom: numpy.ndarray, the imaging response function calculated 
        from images of atoms (suffered from shot noise).
    M2k_Exp_noise: numpy.ndarray, the imaging response function calculated 
        from images of pure noise (only shot noise affects)
    imgIndexMin, imgIndexMax, atomODs, noiseODs, atomODAvg, noiseODAvg: 
        same as `readInImages` 
    """

    atomODs, atomODAvg, noiseODs, noiseODAvg, imgIndexMin, imgIndexMax = \
        readInImages(imgDir, numOfImgsInEachRun, parameter, trapRegion, noiseRegion)
    
    M2k_Exp_atom, _ = __calcNPS(atomODs, atomODAvg, norm=norm, imgSysData=imgSysData) 
    M2k_Exp_noise, _ = __calcNPS(noiseODs, noiseODAvg, norm=norm, imgSysData=imgSysData)

    M2k_Exp_atom = M2k_Exp_atom / atomODAvg.sum()
    M2k_Exp_noise = M2k_Exp_noise / atomODAvg.sum()
    M2k_Exp_atom[M2k_Exp_atom.shape[0]//2, M2k_Exp_atom.shape[1]//2] = 0
    M2k_Exp = M2k_Exp_atom #- M2k_Exp_noise 
    bkg = np.mean(M2k_Exp[-11:-1])
    M2k_Exp -= bkg
    M2k_Exp[M2k_Exp < 0] = 0
    
    return M2k_Exp, M2k_Exp_atom, M2k_Exp_noise, imgIndexMin, imgIndexMax, \
        atomODAvg, noiseODAvg

##############################################################################
# Functions for physics and fitting

def pupilFunc(R_p, Theta_p, tau, S0, alpha, phi, beta):
    """
    Given the polar coordinates, aberration parameters, 
    calculate the exit pupil function.

    ----------
    parameters

    R_p: radial coordinate
    Theta_p: azimuthal coordinate
    tau: describe the radial transmission decaying of the pupil. 
        $T(r) = T_0 \\exp\\left( -r^2 / tau^2 \\right)$
    S0: spherical aberration, $S_0 r^4$
    alpha, phi: astigmatism, 
        $\\alpha r^2 \\cos\\left(2\\theta - 2\\phi\\right)$
    beta: defocus, $\\beta r^2$

    ------
    return

    Exit pupil function
    """
    U = np.exp(-(R_p/tau)**4) * np.array(R_p <= 1, dtype=float)
    Phase = S0 * (R_p**4) + \
            alpha * (R_p**2) * np.cos(2*Theta_p - 2*phi) + \
            beta * (R_p**2)
    return U * np.exp(1j*Phase)


def M2kFuncAnal(K_X, K_Y, d, tau, S0, alpha, phi, beta, delta_s):
    """
    Given the spatial frequency, aberration parameters and the phase 
    introduced by the atom scattering process, 
    calculate the imaging response function

    ----------
    parameters

    K_X, K_Y: spacial frequencies
    d: the ratio for converting the spacial frequencies into 
        coordinates at exit pupil plane
    tau: describe the radial transmission decaying of the pupil. 
        $T(r) = T_0 \\exp\\left( -r^2 / tau^2 \\right)$
    S0: spherical aberration, $S_0 r^4$
    alpha, phi: astigmatism, 
        $\\alpha r^2 \\cos\\left(2\\theta - 2\\phi\\right)$
    beta: defocus, $\\beta r^2$
    delta_s: the phase introduced by the atom scattering process

    ------
    return

    Exit pupil function
    """

    R_p, Theta_p = np.abs(K_X + 1j*K_Y) * d, np.angle(K_X + 1j*K_Y)
    p1 = pupilFunc(R_p, Theta_p + np.pi, tau, S0, alpha, phi, beta)
    p2 = np.conj(pupilFunc(R_p, Theta_p, tau, S0, alpha, phi, beta)) * \
            np.exp(-2*1j*delta_s)
    PSF = (p1 + p2) / (2 * np.cos(delta_s))
    M2k = np.abs(PSF)**2
    
    return M2k


def getFreq(CCDPixelSize, magnification, M2k_mat_shape):
    """
    Calculate the corresponding spacial frequency $k$.
    """

    pixelSize = CCDPixelSize / magnification # pixel size in object plane, in micron
    px_x = 1.0 * pixelSize # micron
    px_y = 1.0 * pixelSize # micron
    sampleNum_x = M2k_mat_shape[1]
    sampleNum_y = M2k_mat_shape[0]

    k_x = np.fft.fftshift(np.fft.fftfreq(sampleNum_x, px_x)) * 2*np.pi
    k_y = np.fft.fftshift(np.fft.fftfreq(sampleNum_y, px_y)) * 2*np.pi

    K_x, K_y = np.meshgrid(k_x, k_y)

    return k_x, k_y, K_x, K_y


def fitM2k(M2k_Exp, imgSysData, paras_guess=None, paras_bounds=None):
    """
    Fit the imaging response function using the model provided in 
    Chen-Lung Hung et al. 2011 New J. Phys. 13 075019
    """
    
    k_x, k_y, K_x, K_y = getFreq(imgSysData["CCDPixelSize"], imgSysData["magnification"], M2k_Exp.shape)
    d = imgSysData["wavelen"] / (2*np.pi*imgSysData["NA"]) 

    def fitFunc(M, *args):
        k_x, k_y = M
        A, tau, S0, alpha, phi, beta, delta_s = args
        return A * M2kFuncAnal(k_x, k_y, d, tau, S0, alpha, phi, beta, delta_s)

    if paras_guess == None:
        #                A ,     tau,   S0,   alpha,  phi,  beta, delta_s
        paras_guess = [ 
                        [.23, 1.22, .0, 1.24, -1.94, 0.12, 0], \
                        [.15, 1.22, .0, 1.43, -1.9,   .36, 0], \
                        [.25, 1.22, .0, 2.04, -1.9, -1.76, 0], \
                        [.15, 1.22, .0, 1.76, -1.9, -0.32, 0], \
                        [.15, 1.22, .0, 1.54, -1.9, -0.14, -0], \
                      ]
    elif paras_guess == 'focus':
        paras_guess = [1 ,  0.8,    0,    1, -1.9,    0,    0]
    elif paras_guess == 'defocus':
        paras_guess = [1 ,  0.8,  -10  -0.5, -2.3,   17,  3.3]

    if paras_bounds == None:
        #                  A,   tau,   S0, alpha,   phi, beta, delta_s       
        paras_bounds = ([ .1,  1.00,   .0,  1.00,-1.94,  -2.,     -2],
                        [ .5,  2.00,  .01,  2.46,-1.83,   4.,      2])

    M2k_Exp_cut = np.clip(M2k_Exp, 0, M2k_Exp.max()) 
        # cut the negative values, which are meaningless

    xdata = np.vstack((K_x.ravel(), K_y.ravel())) 
        # prepare the independent variables, i.e., the coordinates
    xdata = np.delete(xdata, (K_x.shape[0]//2)*K_x.shape[1]+K_x.shape[1]//2, axis=1)
    ydata = np.delete(M2k_Exp_cut.ravel(), (K_x.shape[0]//2)*K_x.shape[1]+K_x.shape[1]//2, axis=0)
        # leave out the center bright point

    rms_min = np.inf
    for pg in paras_guess:
        popt_temp, pcov = curve_fit(fitFunc, xdata, ydata, \
                                p0=pg, max_nfev=500000, bounds=paras_bounds)

        A_fit, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit = popt_temp

        M2k_Fit_temp = A_fit * M2kFuncAnal(K_x, K_y, d, tau_fit, \
                            S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit)

        rms = np.sqrt( np.mean( (M2k_Fit_temp - M2k_Exp_cut)**2 ) )
        if rms < rms_min:
            rms_min = rms
            popt = popt_temp
            M2k_Fit = M2k_Fit_temp
        elif rms == rms_min:
            print("Warning from 'FitM2k': "
                    "    Similar optimal rms error for several sets of initial parameters!")
    
    return M2k_Fit, rms_min, popt, pcov, k_x, k_y, K_x, K_y, d

##############################################################################
# Further data processing

def azmAvgSq(X, Y, Z):
    """
    Do the azimuthal average for a square matrix given in Cartesian 
    coordinates. The origin is considered to be at the center.
    
    ----------
    parameters
    
    X: numpy.ndarray, x-coordinate.
    Y: numpy.ndarray, y-coordinate.
    Z; numpy.ndarray, value at (x, y). X, Y and Z should have same shapes.
    
    ------
    return
    
    sorted_r: numpy.ndarray, radial coordinate, sorted from small to large.
    sorted_v: numpy.ndarray, averaged value at r
    """
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError('X, Y and Z have different shapes!')
        
    if X.shape[0] != X.shape[1]:
        raise ValueError("X isn't a square! Consider to use `azmAvgNonSq`")

    ind_max = X.shape[0] - 1

    R = np.sqrt(X**2 + Y**2)

    r_list = []
    v_list = []

    cind = ind_max // 2

    for ii in range(cind + 1):
        for jj in range(ii + 1):
            r_list.append(R[ii, jj])
            v_temp = [Z[ii, jj], Z[jj, ii], \
                      Z[ind_max - ii, jj], Z[ind_max - jj, ii], \
                      Z[ii, ind_max - jj], Z[jj, ind_max - ii], \
                      Z[ind_max - ii, ind_max - jj], \
                      Z[ind_max - jj, ind_max - ii] ]
            v_list.append(np.mean(v_temp))

    r_list = np.array(r_list)
    v_list = np.array(v_list)

    sorted_indices = np.argsort(r_list)
    sorted_r = r_list[sorted_indices]
    sorted_v = v_list[sorted_indices]
    
    if len(sorted_r.tolist()) != len(set(sorted_r.tolist())):
        print("Warning from `azmAvgSq`: there're duplicate elements in output!")
        
    return sorted_r, sorted_v

def azmAvgNonSq(X, Y, Z):
    """
    Do the azimuthal average for a non-square matrix given in Cartesian 
    coordinates. The origin is considered to be at the center.
    
    ----------
    parameters
    
    X: numpy.ndarray, x-coordinate.
    Y: numpy.ndarray, y-coordinate.
    Z; numpy.ndarray, value at (x, y). X, Y and Z should have same shapes.
    
    ------
    return
    
    sorted_r: numpy.ndarray, radial coordinate, sorted from small to large.
    sorted_v: numpy.ndarray, averaged value at r
    """
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError('X, Y and Z have different shapes!')
        
    if X.shape[0] == X.shape[1]:
        raise ValueError('X is a square matrix! Consider use `azmAvgSq`.')

    row_ind_max = X.shape[0] - 1
    col_ind_max = X.shape[1] - 1

    R = np.sqrt(X**2 + Y**2)

    r_list = []
    v_list = []

    row_cind = row_ind_max // 2
    col_cind = col_ind_max // 2

    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            r_list.append(R[ii, jj])
            v_temp = [Z[ii, jj], Z[row_ind_max - ii, jj], \
                      Z[ii, col_ind_max - jj], \
                      Z[row_ind_max - ii, col_ind_max - jj]]
            v_list.append(np.mean(v_temp))

    r_list = np.array(r_list)
    v_list = np.array(v_list)

    sorted_indices = np.argsort(r_list)
    sorted_r = r_list[sorted_indices]
    sorted_v = v_list[sorted_indices]
    
    if len(sorted_r.tolist()) != len(set(sorted_r.tolist())):
        print("Warning from `azmAvgNonSq`: there're duplicate elements in output!")
    
    return sorted_r, sorted_v

def azmAvg_0(X, Y, Z):
    """
    Do the azimuthal average for any matrix given in Cartesian coordinates. 
    The origin is considered to be at the center.
    
    ----------
    parameters
    
    X: numpy.ndarray, x-coordinate.
    Y: numpy.ndarray, y-coordinate.
    Z; numpy.ndarray, value at (x, y). X, Y and Z should have same shapes.
    
    ------
    return
    
    sorted_r: numpy.ndarray, radial coordinate, sorted from small to large.
    sorted_v: numpy.ndarray, averaged value at r
    """
    
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError('X, Y and Z have different shapes!')
        
    if X.shape[0] == X.shape[1]:
        return azmAvgSq(X, Y, Z)
    else:
        return azmAvgNonSq(X, Y, Z)


def azmAvg(X, Y, Z):
    """
    Do the azimuthal average for a matrix given in Cartesian coordinates.
    
    ----------
    parameters
    
    X: numpy.ndarray, x-coordinate.
    Y: numpy.ndarray, y-coordinate.
    Z; numpy.ndarray, value at (x, y). X, Y and Z should have same shapes.
    
    ------
    return
    
    sorted_r: numpy.ndarray, radial coordinate, sorted from small to large.
    sorted_v: numpy.ndarray, averaged value at r
    """
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError('X, Y and Z have different shapes!')

    row_ind_max = X.shape[0] - 1
    col_ind_max = X.shape[1] - 1

    R = np.sqrt(X**2 + Y**2)

    row_cind = row_ind_max // 2
    col_cind = col_ind_max // 2

    res_dict = {}
    
    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            res_dict[R[ii, jj]] = []

    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            v_temp = [Z[ii, jj], Z[row_ind_max - ii, jj], \
                      Z[ii, col_ind_max - jj], Z[row_ind_max - ii, col_ind_max - jj]]
            res_dict[R[ii, jj]].append(np.mean(v_temp))
            
    for ii in range(row_cind + 1):
        for jj in range(col_cind + 1):
            res_dict[R[ii, jj]] = np.mean(res_dict[R[ii, jj]])
    
    r_list = []
    v_list = []
    
    for r in res_dict:
        r_list.append(r)
        v_list.append(res_dict[r])

    r_list = np.array(r_list)
    v_list = np.array(v_list)

    sorted_indices = np.argsort(r_list)
    sorted_r = r_list[sorted_indices]
    sorted_v = v_list[sorted_indices]
    
    if len(sorted_r.tolist()) != len(set(sorted_r.tolist())):
        print("Warning: there're duplicate elements in output!")
    
    return sorted_r, sorted_v


def calcSk(NPS, M2k, imgSysData, if_plot=False): 
    """
    Given the normalized density fluctuation power spectrum, calculate the 
    static structure factor and do the azimuthal average.
    """
    M2k[M2k==0] = np.inf
    S = NPS / M2k
    _1, _2, K_x, K_y = getFreq(imgSysData["CCDPixelSize"], imgSysData["magnification"], S.shape)

    k, S_azmAvg = azmAvg(K_x, K_y, S)

    # extrapolation = InterpolatedUnivariateSpline(k[k.shape[0]//4:3*k.shape[0]//4], S_azmAvg[k.shape[0]//4:3*k.shape[0]//4], k=2)
    # S_k0 = extrapolation(0)

    return K_x, K_y, S, k, S_azmAvg

##############################################################################
# Functions for making plots
# all the functions for making plots return a figure handle and an axes 
# handle for further manipulations of the plot

def fplot_Intensity(imgDir, num):
    """
    Show raw data of intensity.

    imgDir: string, the directory where the images with atoms are stored.
    num: int, you want to show the num-th image.
    """
    
    paraFile = open(imgDir + "\\parameters.txt", "r")
    line = paraFile.readlines()[num]
    filename = imgDir  + "\\rawimg_" + line.split()[0]
    dim = np.fromfile(filename, '>u2')[0:4]
    img = np.fromfile(filename, '>u2')[4:].reshape((dim[1], dim[3]))

    fig_intens = plt.figure()
    ax_intens = fig_intens.add_subplot(111)
    pc = ax_intens.pcolor(img, cmap=cm.jet)
    ax_intens.set_aspect(1)
    ax_intens.set_title("Raw data (intensity)")
    plt.colorbar(pc)
    return fig_intens, ax_intens


def showExampleImg(imgDir, numOfImgsInEachRun, parameter, \
                   trapRegion=(slice(0, 65535), slice(0, 65535)), \
                   noiseRegion=(slice(0, 65535), slice(0, 65535)), \
                   vRange=[0, 0.5]):
    
    atomODs, atomODAvg, noiseODs, noiseODAvg, imgIndexMin, imgIndexMax = readInImages(imgDir, numOfImgsInEachRun, parameter)
    
    fig_atom = plt.figure()
    ax_atom = fig_atom.add_subplot(111)
    pc = ax_atom.pcolor(atomODAvg, cmap=cm.jet, vmin=vRange[0], vmax=vRange[1])
    ax_atom.set_aspect(1)
    ax_atom.set_title("Example of OD")
    plt.colorbar(pc)
    
    xmin = range(atomODAvg.shape[1])[trapRegion[0]][0]
    xmax = range(atomODAvg.shape[1])[trapRegion[0]][-1]
    ymin = range(atomODAvg.shape[0])[trapRegion[1]][0]
    ymax = range(atomODAvg.shape[0])[trapRegion[1]][-1]
    
    xroute = [xmin, xmin, xmax, xmax, xmin]
    yroute = [ymin, ymax, ymax, ymin, ymin]
    ax_atom.plot(xroute, yroute, '-r', linewidth=2)
    
    xmin = range(atomODAvg.shape[1])[noiseRegion[0]][0]
    xmax = range(atomODAvg.shape[1])[noiseRegion[0]][-1]
    ymin = range(atomODAvg.shape[0])[noiseRegion[1]][0]
    ymax = range(atomODAvg.shape[0])[noiseRegion[1]][-1]
    
    xroute = [xmin, xmin, xmax, xmax, xmin]
    yroute = [ymin, ymax, ymax, ymin, ymin]
    ax_atom.plot(xroute, yroute, '-m', linewidth=2)
    plt.show()
    
    return fig_atom, ax_atom


def fplot_OD(atomOD, if_Save=False, saveDir=None, X=None, Y=None, axes=None, cMap=cm.jet, vRange=None):
    """
    Plot the averaged image (OD) of the 2D thermal atmoic gas
    """
    if axes == None:
        fig_atom = plt.figure(figsize=(6, 4.5))
        ax_atom = fig_atom.add_subplot(111)
    else:
        ax_atom = axes
    
    if X == None or Y == None:
        if vRange == None:
            pc_atom = ax_atom.pcolor(atomOD, cmap=cMap)
        else:
            pc_atom = ax_atom.pcolor(atomOD, cmap=cMap, \
                vmin=max(vRange[0], atomOD.min()), \
                vmax=min(vRange[1], atomOD.max()))
        ax_atom.set_xlabel('$x$ (px)')
        ax_atom.set_ylabel('$y$ (px)')
    else:
        if vRange == None:
            pc_atom = ax_atom.pcolor(X, Y, atomOD, cmap=cMap)
        else:
            pc_atom = ax_atom.pcolor(X, Y, atomOD, cmap=cMap, \
                vmin=max(vRange[0], atomOD.min()), \
                vmax=min(vRange[1], atomOD.max()))
        ax_atom.set_xlabel('$x$ ($\\mu$m)')
        ax_atom.set_ylabel('$y$ ($\\mu$m)')
    plt.colorbar(pc_atom, extend='both')
    ax_atom.set_aspect(1)
    ax_atom.set_title("2D thermal gas (OD)")
    if if_Save:
        plt.savefig(saveDir + "\\2Dgas.png", dpi='figure')
    return fig_atom, ax_atom


def fplot_NPS_Exp(K_x, K_y, M2k_Exp, if_Save=False, saveDir=None, axes=None, cMap=cm.jet, vRange=None):
    """
    Plot noise power spectrum (only experimental result)
    """
    if axes == None:
        fig_NPS_Exp = plt.figure(figsize=(6, 4.5))
        ax_NPS_Exp = fig_NPS_Exp.add_subplot(111)
    else:
        ax_NPS_Exp = axes
    if vRange == None:
        pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x, K_y, M2k_Exp, cmap=cMap)
    else:
        pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x, K_y, M2k_Exp, cmap=cMap, vmin=vRange[0], vmax=vRange[1])
    plt.colorbar(pc_NPS_Exp)
    ax_NPS_Exp.set_aspect(1)
    ax_NPS_Exp.set_title("Noise Power Spectrum (Exp.)")
    ax_NPS_Exp.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Exp.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    if if_Save:
        plt.savefig(saveDir + "\\NoisePowerSpec_Exp.png", dpi='figure')
    return fig_NPS_Exp, ax_NPS_Exp


def fplot_NPS_ExpAndFit(K_x, K_y, M2k_Exp, M2k_Fit=None, if_Save=False, saveDir=None, fig=None, cMap=cm.jet, \
    vRange_Exp=None, vRange_Fit=None):
    """
    Plot noise power spectrum (experimental result and fit result)
    """
    if fig == None:
        fig_NPS = plt.figure('NPS', figsize=(12, 5))
    else:
        fig_NPS = fig

    if vRange_Exp == None:
        vMin_Exp = 0
        vMax_Exp = M2k_Fit.max()
    if vRange_Fit == None:
        vMin_Fit = 0
        vMax_Fit = M2k_Fit.max()  

    ax_NPS_Exp = fig_NPS.add_subplot(121)
    pc_NPS_Exp = ax_NPS_Exp.pcolor(K_x, K_y, M2k_Exp, cmap=cMap, vmin=vMin_Exp, vmax=vMax_Exp)
    divider = make_axes_locatable(ax_NPS_Exp)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc_NPS_Exp, cax=cax, extend='both')
    ax_NPS_Exp.set_aspect(1)
    ax_NPS_Exp.set_title("Noise Power Spectrum (Exp.)")
    ax_NPS_Exp.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Exp.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    
    ax_NPS_Fit = fig_NPS.add_subplot(122)
    pc_NPS_Fit = ax_NPS_Fit.pcolor(K_x, K_y, M2k_Fit, cmap=cMap, vmin=vMin_Fit, vmax=vMax_Fit)
    divider = make_axes_locatable(ax_NPS_Fit)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc_NPS_Fit, cax=cax, extend='both')
    ax_NPS_Fit.set_aspect(1)
    ax_NPS_Fit.set_title("Noise Power Spectrum (Fit)")
    ax_NPS_Fit.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
    ax_NPS_Fit.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
    if if_Save:
        plt.savefig(saveDir + "\\NoisePowerSpec_ExpAndFit.png", dpi='figure')
    return fig_NPS, ax_NPS_Exp, ax_NPS_Fit


def fplot_NPS_LineCut(k_x, k_y, M2k_Exp, M2k_Fit, if_Save=False, saveDir=None):
    """
    Plot noise power spectrum - line cut
    """
    ll = M2k_Exp.shape[0]
    k_r = np.linspace(-4, 4, ll)
    t_scan = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    linetype = [['.-b', '-b'], ['.-k', '-k'], ['.-r', '-r'], ['.-m', '-m']]
    title = ['0', 'pi/4', 'pi/2', '3*pi/4']
    
    fig_NPS_LineCut = plt.figure('NPS_LineCut', figsize=(12, 9))
    for k in range(len(t_scan)):
        t = t_scan[k]
        
        kx, ky = k_r * np.cos(t), k_r * np.sin(t)
        ix= np.array(k_x.shape[0]*(kx-k_x.min()) / (k_x.max()-k_x.min()), dtype=int)
        iy= np.array(k_y.shape[0]*(ky-k_y.min()) / (k_y.max()-k_y.min()), dtype=int)
            
        M2k_r_exp = M2k_Exp[iy, ix]
        M2k_r_fit = M2k_Fit[iy, ix]
        
        axNPS_LineCut = fig_NPS_LineCut.add_subplot(2, 2, k+1)
        axNPS_LineCut.plot(k_r, M2k_r_exp, linetype[k][0], linewidth=0.5, label=title[k]+'-Exp.')
        axNPS_LineCut.plot(k_r, M2k_r_fit, linetype[k][1], linewidth=2, label=title[k]+'-Fit')
        axNPS_LineCut.legend()
        axNPS_LineCut.set_xlabel('$k$ ($\\mu$m$^{-1}$)')
        axNPS_LineCut.set_ylabel('Noise Power Spectrum (a.u.)')
        axNPS_LineCut.set_ylim([0, 1.5*M2k_Fit.max()])
        #axNPS_LineCut.set_ylim([0, 1])
    if if_Save:
        plt.savefig(saveDir + "\\NoisePowerSpec_LineCut.png", dpi='figure')
    return fig_NPS_LineCut


def fplot_pupil(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, if_Save=False, saveDir=None):
    """
    Plot the pupil
    """ 

    r_p_pupilplt = np.linspace(0, 1, 200)
    theta_p_pupilplt = np.linspace(-np.pi, np.pi, 300)
    
    R_p_pupilplt, Theta_p_pupilplt = np.meshgrid(r_p_pupilplt, theta_p_pupilplt)
    
    pupilplt = pupilFunc(R_p_pupilplt, Theta_p_pupilplt, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit)
    
    X_pupilplt = R_p_pupilplt * np.cos(Theta_p_pupilplt)
    Y_pupilplt = R_p_pupilplt * np.sin(Theta_p_pupilplt)
    
    fig_pupil = plt.figure('pupil', figsize=(12, 6))
    ax_pupil = fig_pupil.add_subplot(121)
    pc_pupil = ax_pupil.pcolor(X_pupilplt, Y_pupilplt, np.angle(pupilplt), cmap=cm.twilight_shifted, vmin=-np.pi, vmax=np.pi)
    ax_pupil.set_aspect(1)
    ax_pupil.set_title('Phase of exit pupil (radian)')
    ax_pupil.axis('off')
    divider = make_axes_locatable(ax_pupil)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(pc_pupil, cax=cax)
    
    ax_pupil_2 = fig_pupil.add_subplot(122)
    pc_pupil_2 = ax_pupil_2.pcolor(X_pupilplt, Y_pupilplt, np.angle(pupilplt), cmap=cm.RdYlGn)
    ax_pupil_2.set_aspect(1)
    ax_pupil_2.set_title('Phase of exit pupil (radian)')
    ax_pupil_2.axis('off')
    divider = make_axes_locatable(ax_pupil_2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(pc_pupil_2, cax=cax)
    if if_Save:
        plt.savefig(saveDir + "\\Pupil.png", dpi='figure')
    return fig_pupil, ax_pupil, ax_pupil_2


def fplot_PSF(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save=False, saveDir=None):
    """
    Plot the PSF as defined in Chen-Lung Hung et al 2011 New J. Phys. 13 075019
    """

    rg_PSF = 8
    px_PSF = 0.05
    
    px_Pupil = 1 / (2*rg_PSF)
    rg_Pupil = 1 / (2*px_PSF)
    sampleNum = np.int(2*rg_Pupil / px_Pupil)
    
    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j*Eta), np.angle(Xi + 1j*Eta)
    
    Pupil = pupilFunc(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit)
    
    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.real(np.exp(1j*delta_s_fit) * U)
    
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx*d*2*np.pi, fx*d*2*np.pi)
    
    
    fig_PSF = plt.figure('PSF', figsize=(6, 4.5))
    ax_PSF = fig_PSF.add_subplot(111)
    MM = np.max([np.abs(PSF.max()), np.abs(PSF.min())])
    pc_PSF = ax_PSF.pcolor(X, Y, PSF, cmap=cm.bwr, vmin=-MM, vmax=MM)
    ax_PSF.set_aspect(1)
    ax_PSF.set_xlim([-10, 10])
    ax_PSF.set_ylim([-10, 10])
    ax_PSF.set_title('Point spread function')
    ax_PSF.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF.set_ylabel('$y$ in object plane ($\\mu$m)')
    plt.colorbar(pc_PSF)
    if if_Save:
        plt.savefig(saveDir + "\\PSF.png", dpi='figure')
    return fig_PSF, ax_PSF


def fplot_PSF_LineCut(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save=False, saveDir=None):
    """
    Plot the linecut of PSF as defined in Chen-Lung Hung et al 2011 New J. Phys. 13 075019
    """

    rg_PSF = 8
    px_PSF = 0.05
    
    px_Pupil = 1 / (2*rg_PSF)
    rg_Pupil = 1 / (2*px_PSF)
    sampleNum = np.int(2*rg_Pupil / px_Pupil)
    
    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j*Eta), np.angle(Xi + 1j*Eta)
    
    Pupil = pupilFunc(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit)
    
    cind = x.shape[0]//2
    
    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.real(np.exp(1j*delta_s_fit) * U)
    
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx*d*2*np.pi, fx*d*2*np.pi)
    
    Pupil2 = np.array(R_p_PSFplt<=1, dtype=float)
    U2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil2)))
    PSF2 = np.real(np.exp(1j*delta_s_fit) * U2)
    
    PSF = PSF / PSF[cind, cind]
    PSF2 = PSF2 / PSF2[cind, cind]
    
    fig_PSF_LineCut = plt.figure('PSF_LineCut', figsize=(6, 4.5))
    ax_PSF_LineCut = fig_PSF_LineCut.add_subplot(111)
    ax_PSF_LineCut.plot(X[cind, :], PSF[cind, :], '-k', label='Fit')
    ax_PSF_LineCut.set_title('Point spread function')
    ax_PSF_LineCut.plot(X[cind, :], PSF2[cind, :], '--k', label='Ideal')
    ax_PSF_LineCut.xaxis.set_major_locator(MultipleLocator(1))
    ax_PSF_LineCut.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_PSF_LineCut.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF_LineCut.set_ylabel('Point spread function (a.u.)')
    ax_PSF_LineCut.legend()
    ax_PSF_LineCut.grid(True)
    ax_PSF_LineCut.set_xlim([-10, 10])
    if if_Save:
        plt.savefig(saveDir + "\\PSF_LineCut.png", dpi='figure')
    return fig_PSF_LineCut, ax_PSF_LineCut


def fplot_PSF_abs2(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save=False, saveDir=None):
    """
    Plot the PSF as traditionally defined
    """ 

    rg_PSF = 8
    px_PSF = 0.05
    
    px_Pupil = 1 / (2*rg_PSF)
    rg_Pupil = 1 / (2*px_PSF)
    sampleNum = np.int(2*rg_Pupil / px_Pupil)
    
    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j*Eta), np.angle(Xi + 1j*Eta)
    
    Pupil = pupilFunc(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit)
    
    cind = x.shape[0]//2
    
    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.abs(np.exp(1j*delta_s_fit) * U)**2
    
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx*d*2*np.pi, fx*d*2*np.pi)
    
    
    fig_PSF_abs2 = plt.figure('PSF_abs2', figsize=(6, 4.5))
    ax_PSF_abs2 = fig_PSF_abs2.add_subplot(111)
    MM = np.max([np.abs(PSF.max()), np.abs(PSF.min())])
    pc = ax_PSF_abs2.pcolor(X, Y, PSF, cmap=cm.bwr, vmin=-MM, vmax=MM)
    ax_PSF_abs2.set_aspect(1)
    ax_PSF_abs2.set_xlim([-10, 10])
    ax_PSF_abs2.set_ylim([-10, 10])
    ax_PSF_abs2.set_title('Point spread function (classical def.)')
    ax_PSF_abs2.set_xlabel('$x$ in object plane ($\\mu$m)')
    ax_PSF_abs2.set_ylabel('$y$ in object plane ($\\mu$m)')
    plt.colorbar(pc)
    if if_Save:
        plt.savefig(saveDir + "\\PSF_abs2.png", dpi='figure')
    return fig_PSF_abs2, ax_PSF_abs2

# plot the PSF_abs2 - line cut
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def uPercent_max_x(x, y):
    h = max(y)/100.0
    signs = np.sign(np.add(y, -h))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], h),
            lin_interp(x, y, zero_crossings_i[1], h)], h
    
def calcResolution(PSF_abs2_linecut, x):
    # find the two crossing points
    hmx, h = uPercent_max_x(x, PSF_abs2_linecut)
    res = (hmx[1] - hmx[0]) / 2
    return res, hmx[1], hmx[0], h

def fplot_PSF_abs2_LineCut(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save=False, saveDir=None):
    """
    Plot the linecut of PSF as traditionally defined. Estimate the resolution.
    """ 

    rg_PSF = 8
    px_PSF = 0.05
    
    px_Pupil = 1 / (2*rg_PSF)
    rg_Pupil = 1 / (2*px_PSF)
    sampleNum = np.int(2*rg_Pupil / px_Pupil)
    
    x = np.linspace(-rg_Pupil, rg_Pupil, sampleNum)
    Xi, Eta = np.meshgrid(x, x)
    R_p_PSFplt, Theta_p_PSFplt = np.abs(Xi + 1j*Eta), np.angle(Xi + 1j*Eta)
    
    Pupil = pupilFunc(R_p_PSFplt, Theta_p_PSFplt, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit)
    
    cind = x.shape[0]//2
    
    U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil)))
    PSF = np.abs(np.exp(1j*delta_s_fit) * U)**2
    
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[0], px_Pupil))
    X, Y = np.meshgrid(fx*d*2*np.pi, fx*d*2*np.pi)
    
    Pupil2 = np.array(R_p_PSFplt<=1, dtype=float)
    U2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Pupil2)))
    PSF2 = np.abs(np.exp(1j*delta_s_fit) * U2)**2
    
    PSF = PSF / PSF[cind, cind]
    PSF2 = PSF2 / PSF2[cind, cind]
    
    fig_PSF_abs2_LineCut = plt.figure('PSF_abs2_LineCut', figsize=(6, 4.5))
    ax_PSF_abs2_LineCut = fig_PSF_abs2_LineCut.add_subplot(111)
    ax_PSF_abs2_LineCut.plot(X[cind, :], PSF[cind, :], '-k', label='Fit')
    ax_PSF_abs2_LineCut.set_title('Point spread function (classical def.)')
    ax_PSF_abs2_LineCut.plot(X[cind, :], PSF2[cind, :], '--k', label='Ideal')
    ax_PSF_abs2_LineCut.xaxis.set_major_locator(MultipleLocator(1))
    ax_PSF_abs2_LineCut.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_PSF_abs2_LineCut.legend()
    ax_PSF_abs2_LineCut.grid(True)
    ax_PSF_abs2_LineCut.set_xlim([-10, 10])   
    ax_PSF_abs2_LineCut.set_xlabel('$x$ in object plane ($\\mu$m)')
    if if_Save:
        plt.savefig(saveDir + "\\PSF_abs2_LineCut.png", dpi='figure')

    resolution, b, a, h = calcResolution(PSF[cind, :], X[cind, :])
    ax_PSF_abs2_LineCut.plot([a, b], [h, h], '-b')
    return resolution, fig_PSF_abs2_LineCut, ax_PSF_abs2_LineCut


##############################################################################
# functions for the overall manipulation

def plotAndSave(imgDir, resDir, imgIndexMin, imgIndexMax, \
    K_x, K_y, k_x, k_y, M2k_Exp, M2k_Fit, popt,\
        atomODAvg, imgSysData, choices):
    
    plt.close('all')
    d = imgSysData["wavelen"] / (2*np.pi*imgSysData["NA"]) 
    if_Save = choices["if_Save"]
    
    saveDir = resDir + "\\{}_{}-{}".format(imgDir[-6:], imgIndexMin, imgIndexMax)
    if choices["if_Save"] and (not os.path.exists(saveDir)):
        # create the saveDir if not exist
        os.makedirs(saveDir)
        
    if choices["plot_2dGas"]:
        fplot_OD(atomODAvg, if_Save, saveDir)
        
    if choices["plot_NoisePowSpec"] and (not choices["do_Fit"]):
        fplot_NPS_Exp(K_x, K_y, M2k_Exp, if_Save, saveDir)
    
    if choices["do_Fit"]:
        A_fit, tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit = popt
        
        # write the fit parameters to file
        writeStr = \
            '[A, tau, S0, alpha, phi, beta, delta_s] =\n' + \
            'tau     := describe the decaying of transmission efficiency with radius\n' + \
            'S0      := spherical aberration\n' + \
            'alpha   := astigmatism\n' + \
            'beta    := defocus\n' + \
            'delta_s := phase imprint by atom scattering\n' + \
            str(popt.tolist())
        if if_Save:
            resFile = open(saveDir + "\\FitResults.txt", 'w')
            resFile.write(writeStr)
            resFile.close()
        
        # display the fit results
        dispStr = \
            'tau     = {: .4f} : describe the decaying of transmission efficiency with radius\n' + \
            'S0      = {: .4f} : spherical aberration\n' + \
            'alpha   = {: .4f} : astigmatism\n' + \
            'beta    = {: .4f} : defocus\n' + \
            'delta_s = {: .4f} : phase imprint by atom scattering\n'
        print(dispStr.format(tau_fit, S0_fit, alpha_fit, beta_fit, delta_s_fit))
        
        # if not choices["normalize"]:
        #     print('eta     = {: .4f} : the ratio between the atom number and the optical density\n'.format(1.3 / A_fit))
        #     print('Total atom number is around: {: .1f}\n'.format(atomODAvg.sum() * 1.3 / A_fit))

        # plots
        if choices["plot_Pupil"]:
            fplot_pupil(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, if_Save, saveDir)
        if choices["plot_PSF"]:
            fplot_PSF(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save, saveDir)
        if choices["plot_PSF_LineCut"]:
            fplot_PSF_LineCut(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save, saveDir)
        if choices["plot_PSF_abs2"]:
            fplot_PSF_abs2(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save, saveDir)
        if choices["plot_PSF_abs2_LineCut"]:
            resolution, _1, _2 =  fplot_PSF_abs2_LineCut(tau_fit, S0_fit, alpha_fit, phi_fit, beta_fit, delta_s_fit, d, if_Save, saveDir)
            print("The Rayleigh-criterion resolution is approximately {:.1f} micron".format(resolution))
        if choices["plot_NoisePowSpec"]:
            fplot_NPS_ExpAndFit(K_x, K_y, M2k_Exp, M2k_Fit, if_Save, saveDir)
    if choices["plot_NoisePowSpec_LineCut"]:
        fplot_NPS_LineCut(k_x, k_y, M2k_Exp, M2k_Fit, if_Save, saveDir)    
    plt.show()


def doCalibration(imgDir, resDir, trapRegion, noiseRegion, numOfImgsInEachRun, parameter, \
    imgSysData, choices):
    
    M2k_Exp, M2k_Exp_atom, M2k_Exp_noAtom, imgIndexMin, imgIndexMax, \
         atomODAvg, noiseODAvg = \
        calcNPS(imgDir, numOfImgsInEachRun, parameter, trapRegion, noiseRegion, norm=choices["normalize"], imgSysData=imgSysData)
   
    if choices["do_Fit"]:
        M2k_Fit, rms_min, popt, pcov, k_x, k_y, K_x, K_y, d = \
            fitM2k(M2k_Exp, imgSysData)
    else:
        M2k_Fit = M2k_Exp
        popt = None
        k_x, k_y, K_x, K_y = getFreq(imgSysData["CCDPixelSize"], imgSysData["magnification"], M2k_Exp.shape)

    plotAndSave(imgDir, resDir, imgIndexMin, imgIndexMax, \
        K_x, K_y, k_x, k_y, M2k_Exp, M2k_Fit, popt,\
        atomODAvg, imgSysData, choices)

    return K_x, K_y, M2k_Exp, M2k_Fit, popt, atomODAvg


def doAnalysis(popt, imgDir, resDir, trapRegion, noiseRegion, numOfImgsInEachRun, parameter, imgSysData, choices, M2k):

    NPS_Exp, NPS_Exp_atom, NPS_Exp_noAtom, imgIndexMin, imgIndexMax ,*_ = calcNPS(
        imgDir, numOfImgsInEachRun, parameter, trapRegion, noiseRegion, norm=choices["normalize"], imgSysData=imgSysData)
    *_, K_X, K_Y = getFreq(imgSysData["CCDPixelSize"], imgSysData["magnification"], NPS_Exp.shape)

    K_X, K_Y, S, k, S_azmAvg = calcSk(NPS_Exp, M2k, imgSysData)

    saveDir = resDir + "\\{}_{}-{}".format(imgDir[-6:], imgIndexMin, imgIndexMax)
    if choices["if_Save"] and (not os.path.exists(saveDir)):
        # create the saveDir if not exist
        os.makedirs(saveDir)

    if choices["plot_NPS"]:
        NPS_Exp[NPS_Exp == NPS_Exp.max()] = 0
        fig_NPS = plt.figure()
        ax_NPS = fig_NPS.add_subplot(111)
        pc = ax_NPS.pcolor(K_X, K_Y, NPS_Exp, cmap=cm.jet)
        plt.colorbar(pc)
        ax_NPS.set_aspect(1)
        ax_NPS.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
        ax_NPS.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
        ax_NPS.set_title('Noise power spectrum')
        if choices["if_Save"]:
            plt.savefig(saveDir + "\\NoisePowerSpectrum.png", dpi='figure')

    if choices["plot_S2d"]:
        S[S == S.max()] = 0
        fig_S2d = plt.figure()
        ax_S2d = fig_S2d.add_subplot(111)
        pc = ax_S2d.pcolor(K_X, K_Y, S, cmap=cm.jet)
        plt.colorbar(pc)
        ax_S2d.set_aspect(1)
        ax_S2d.set_xlabel('$k_x$ ($\\mu$m$^{-1}$)')
        ax_S2d.set_ylabel('$k_y$ ($\\mu$m$^{-1}$)')
        ax_S2d.set_title('Static structure factor')
        if choices["if_Save"]:
            plt.savefig(saveDir + "\\StaticStructureFactor.png", dpi='figure')

    if choices["plot_Sk_azmAvg"]:
        fig_S_azm = plt.figure('static structure factor', figsize=(6, 4))
        ax_S_azm = fig_S_azm.add_subplot(111)
        ax_S_azm.plot(k[1:], S_azmAvg[1:], '.b')
        ax_S_azm.set_xlabel('$k$ ($\\mu$m$^{-1}$)')
        ax_S_azm.set_ylabel('$S(k)$ (a.u.)')
        ax_S_azm.set_title('Static structure factor (azimuthal average)')
        if choices["if_Save"]:
            plt.savefig(saveDir + "\\StaticStructureFactor_AzimuthalAvg.png", dpi='figure')
    
    return K_X, K_Y, S, k, S_azmAvg
    