import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import plotly.graph_objects as go

'''
Eric Reinhardt, Purdue University
Plotting Variables: 
modelname: name of parent folder within ./Plots
modelorreco: name of subfolder within /modelname depending on type of plot e.g. Model, Reco, Comparison
space: name of subfolder within /modelorreco depending on type of kinematics e.g. Momentum/K, EtaPhi
true: "true" values e.g. generator level Monte Carlo data
pred: single set of predicted values for a variable
preds: an array of multiple sets of predicted values
varname: name of variable used for titles and axis labels e.g. Nu pT
prednames: names of prediction sets e.g. Reco MC, ML Model
lower: lower range for plots
upper: upper range for plots
bins: bins for plots
bincenters: centers values for bins in scatter plots
samples: bin-wise number of events
metrics: a collection of variables of same type e.g. FWHM, mean
metricname: name of metric e.g. 'FWHM', 'Mean'
Data Type Conversion Variables:
px, py, pz, e: components of momentum 4-vector
pt, eta, phi, m: components of eta-phi Lorentz 4-vector
Data Cut Variables:
datain: data set to be modified
datacheck: data set to be checked for two conditions
minval: lower value to cut datain along when evaluating datacheck
maxval: upper value to cut datain along when evaluating datacheck
Statistical Calculation Variables
values: a set of values for which to check the standard error of the mean
'''

#Get lower and upper limits for plots
def getlowerupper(pred, true):
    lower = min(min(pred), min(true))
    upper = max(max(pred), max(true))
    
    return lower, upper

#Histogram with ratio subplot
def histo(modelname, modelorreco, space, true, preds, varname, prednames, lower, upper, bins):
    #Get bins for true labels
    bins = np.histogram(true, 
                        bins=bins, 
                        range=[lower, upper])[1]
    
    #Find bin count statistics
    bincountsx, _, _ = stats.binned_statistic(preds[0], 
                                              preds[0], 
                                              statistic='count', 
                                              bins=bins, 
                                              range=[lower, upper])
    bincountsy, _, _ = stats.binned_statistic(true, 
                                              true, 
                                              statistic='count',
                                              bins=bins, 
                                              range=[lower, upper])
    
    #Create histo and ratio subplots
    fig,axs = plt.subplots(2, gridspec_kw={'height_ratios':[3, 1]})
    fig.suptitle(varname)
    
    #In first axis create set of histograms
    axs[0].hist(true, 
                bins=bins, 
                range=[lower, upper], 
                density=1, 
                histtype='step', 
                label='Ground Truth')
    axs[0].scatter([], 
                   [], 
                   marker='x',
                   color='red',
                   label='Missing Ground Truth')
    for i in range(len(preds)):
        axs[0].hist(preds[i], 
                    bins=bins, 
                    range=[lower, upper], 
                    density=1, 
                    histtype='step', 
                    label=prednames[i] + ' Solution')
    axs[0].scatter([], 
                   [], 
                   marker='x', 
                   color='blue', 
                   label='Missing ' + prednames[0])
    axs[0].set_ylabel('Density=1')
    axs[0].legend()
    axs[0].set_xlim([lower, upper])
    
    #Find bins with missing data
    binratio = bincountsx / bincountsy
    bincenters = (bins[1:] + bins[:-1]) / 2
    bincentersfinal = bincenters[binratio != 0]
    biniszero = binratio == 0
    binratio = binratio[binratio != 0]
    binisnan = np.isnan(binratio)
    binnonan = ~binisnan
    
    #Plot ratio subplot with markers for bins with missing data
    axs[1].scatter(bincentersfinal[binnonan], 
                   binratio[binnonan])
    axs[1].scatter(bincentersfinal[binisnan], 
                   np.ones(len(bincentersfinal[binisnan])), 
                   marker='x', color='red')
    axs[1].scatter(bincenters[biniszero], 
                   np.ones(len(bincenters[biniszero])), 
                   marker='x', 
                   color='blue')
    axs[1].axhline(y=1)
    axs[1].set_xlim([lower, upper])
    axs[1].set_ylim([0.5, 1.5])
    axs[1].set_ylabel(prednames[0] + ' / Ground Truth')
    _, pval = stats.ks_2samp(true, preds[0])
    axs[1].set_xlabel('KS-Test P-value: %.2e' % pval)
    
    #Save plot
    fig.savefig('./Plots/%s/%s/%s/%s histo' % (modelname, modelorreco, space, varname),bbox_inches='tight')
    plt.close()

#Cut items from an array outside of a given value range
def cuts(datain,datacheck,minval,maxval):
    dataout = datain[np.logical_and(datacheck > minval, datacheck < maxval)]
    
    return(dataout)

#Calculate the standard error of the mean
def sem(values):
    sem = stats.sem(values)
    
    return(sem)


def plotgaussian(modelname, modelorreco, space, true, pred, varname, lower, upper, bins):
    resmeans = []
#     fwhms = []
    stddevs = []
    bincenters = []
    samples = []
    
    #Compute residuals and bin width
    width = (upper - lower) / bins
    true = cuts(true, pred, lower, upper)
    pred = cuts(pred, pred, lower, upper)
    pred = cuts(pred, true, lower, upper)
    true = cuts(true, true, lower, upper)
    
    resids = true - pred
    
    #Create gaussian plot for each bin
    for i in range(bins):
        #Find bin dimensions
        lowertemp = lower + width * i
        uppertemp = lower + width * (i + 1)
        bincentertemp = (uppertemp + lowertemp) / 2
        
        #Cut the residuals to within lower and upper range along true values
        residscut = cuts(resids, 
                         true, 
                         lowertemp, 
                         uppertemp)
        
        varnametemp = '%s Residuals Distribution (%.2f to %.2f)' % (varname, 
                                                                    lowertemp, 
                                                                    uppertemp)
        residscut.sort()
        
        #Computer statistics for residuals
        resmeantemp = np.mean(residscut)
        resstdtemp = np.std(residscut)
        samplestemp = len(residscut)
#         fwhmtemp = 2 * np.sqrt(2 * np.log(2)) * resstdtemp
        pdf = stats.norm.pdf(residscut, resmeantemp, resstdtemp)
        
        #Plot histogram of residuals
        plt.hist(residscut, 
                 bins=bins, 
                 histtype='step', 
                 color='blue', 
                 density=1, 
                 label='Residuals')
        
        #Plot the normal curve fitted to the residuals
        plt.plot(residscut, 
                 pdf, 
                 label='Normal Curve', 
                 color='black')
        plt.title(varnametemp)
        
        #Plot the residuals mean as a vertical line
        plt.axvline(resmeantemp, 
                    label='Mean: %.2f' % resmeantemp, 
                    color='red')
        plt.xlabel('Stdev: %.2f (samples: %.i)' % (resstdtemp, 
                                                   samplestemp))
        
        #Plot the full-width half maximum range
        plt.axvspan(resmeantemp - resstdtemp / 2,
                    resmeantemp + resstdtemp / 2,
                    facecolor='g',
                    alpha=.3,
                    label='FWHM')        
        plt.legend()
        plt.savefig('./Plots/%s/%s/%s/%s_Resids_%.2f_%.2f.png' % (modelname,
                                                                  modelorreco, 
                                                                  space, 
                                                                  varname,
                                                                  lowertemp,
                                                                  uppertemp), bbox_inches='tight')
        plt.close()
        
        #Append values to lists to be used by scatter() function
        resmeans.append(resmeantemp)
        samples.append(samplestemp)
        stddevs.append(resstdtemp)
        bincenters.append(bincentertemp)
    
    resmean = np.mean(resids)
    stddev = np.std(resids)    
    
    return resmeans, stddevs, bincenters, samples, resmean, stddev

#Scatter plot with ratio subplot designed to compare FWHM or residuals mean values from a set of gaussians from plotgaussian
def scatter(modelname, space, metrics, bincenters, samples, varname, prednames, metricname):
    bincenters = np.array(bincenters)
    
    #Check that the bin centers match between the two datasets
    if (np.std(bincenters[:,0]) != 0) or (np.std(bincenters[:,-1]) != 0):
        print('Scatter plot failed due to different bincenters')
        print(bincenters)
        return
    else:
        #Create subplots for scatter and ratio
        fig,axs = plt.subplots(2, gridspec_kw={'height_ratios':[3,1]})
        axs[0].set_ylabel(metricname)
        axs[0].set_title('Gaussian %s vs %s' % (metricname,
                                                varname))
        
        for i in range(len(prednames)):
            #Find the width and range of bins
            width = (bincenters[i][-1] - bincenters[i][-2]) / 2
            lower = min(bincenters[i]) - width
            upper = max(bincenters[i]) + width        

            #Store first 2 metric vals
            if i == 0:
                metric1 = np.array(metrics[i].copy())
            elif i ==1:
                metric2 = np.array(metrics[i].copy())

            #Create the scatterplot
            axs[0].scatter(bincenters[i], 
                           metrics[i], 
                           label=prednames[i])
            axs[0].axhline(y=0)


            #Use standard error of the mean for y error and bin width for x error
            axs[0].errorbar(bincenters[i],
                            metrics[i],
                            xerr=(bincenters[i][-1] - bincenters[i][-2]) / 2,
                            yerr=metrics[i] / np.sqrt(samples[i]),
                            linestyle='')
        axs[0].legend()
        axs[0].set_xlim([lower,upper])
        
        #Plot ratio subplot for first 2 data sets
        axs[1].scatter(bincenters[i], metric1 / metric2)
        axs[1].axhline(y=1)
        axs[1].set_xlim([lower,upper])
        axs[1].set_ylim([0.5,2.0])
        axs[1].set_ylabel('%s / %s' % (prednames[0],
                                       prednames[1]))
        axs[1].set_xlabel('%s' % varname)
        
        #Save plot
        fig.savefig('./Plots/%s/Comparison/%s/%s_%s_scatter' % (modelname, 
                                                                space,
                                                                varname,
                                                                metricname), bbox_inches='tight')
        plt.close()

#95% confidence interval table
def confinttable(modelname, space, modelorreco, mean, std, varname, samples, lower, upper):
    center = (lower + upper) / 2
    lowconf = mean - std * 1.96
    uppconf = mean + std * 1.96

    fig = go.Figure(data = [go.Table(header=dict(values=[varname, "Value range: %.2f - %.2f" % (lower, upper)]),
                                     cells=dict(values=[['Number of Samples', 'Residuals Mean', 'Standard Deviation', 
                                                                 '95% Confidence Interval'], [samples, "%.4f" % mean, 
                                                                 '%.3f' % std, '(%.3f, %.3f)' % (lowconf, uppconf)]]))
                           ])
                            
    fig.write_image('./Plots/%s/%s/%s/%s_ConfInt.png' % (modelname,
                                                         modelorreco, 
                                                         space, 
                                                         varname))           
        
#Root mean squared error plot for a single set of data
def RMSEsingle(modelname, true, pred, varname, lower, upper, bins):
    #Find width of bins
    width = (upper - lower) / bins
    
    #Get residuals
    true = cuts(true, pred, lower, upper)
    pred = cuts(pred, pred, lower, upper)
    pred = cuts(pred, true, lower, upper)
    true = cuts(true, true, lower, upper)
    dif = true - pred
    
    #Compute bin-wise statistics for standard deviation and standard error
    xcenters = np.linspace(lower + .5 * width, 
                           upper + .5 * width, 
                           bins, 
                           endpoint=False)
    binstd, _, _ = stats.binned_statistic(true, 
                                          dif, 
                                          statistic='std', 
                                          bins=bins, 
                                          range=[lower, upper])
    binsem, _, _ = stats.binned_statistic(true, 
                                          dif, 
                                          statistic=sem,
                                          bins=bins, 
                                          range=[lower, upper])
    
    #Create RMSE plot
    fig, axs = plt.subplots(1)
    axs.scatter(xcenters, binstd)
    axs.set_xlim(lower, upper)
    axs.errorbar(xcenters, 
                 binstd, 
                 yerr=binsem, 
                 xerr=width * .5, 
                 ls='none')
    axs.set_title('RMSE vs %s' % varname)
    axs.set_ylabel('RMS (true - pred)')
    axs.set_xlabel('%s True' % varname)
    fig.savefig('./Plots/%s/%s/%s_RMSE Plot' % (modelname, 
                                                        space, 
                                                        varname), bbox_inches='tight')
    plt.close()
    
#RMSE plot for multiple datasets with ratio subplot comparing two sets of data
def RMSEcompare(modelname, space, true, preds, varname, prednames, lower, upper, bins):
    #Calculate bin width
    width = (upper - lower) / bins 

    
    #Create scatter plot and ratio subplot
    fig,axs=plt.subplots(2,gridspec_kw={'height_ratios':[3,1]})
    
    #Set scatterplot title and axis info
    axs[0].set_title('RMSE vs %s' % varname)
    axs[0].set_ylabel('RMS (true-pred)')
    axs[0].set_xlabel('%s True' % varname)
    axs[0].set_xlim(lower,
                    upper + width)
    
    #Plot RMSE scatter plots
    for i in range(len(preds)):
        truetemp = cuts(true, preds[i], lower, upper)
        predtemp = cuts(preds[i], preds[i], lower, upper)
        predtemp = cuts(predtemp, truetemp, lower, upper)
        truetemp = cuts(truetemp, truetemp, lower, upper)
        dif = truetemp - predtemp
        xcenters = np.arange(lower + .5 * width,
                             upper + .5 * width,
                             width)
        binstd, _, _ = stats.binned_statistic(truetemp, 
                                              dif, 
                                              statistic='std', 
                                              bins=bins,
                                              range=[lower,upper])
        #Store first 2 RMSE vals
        if i == 0:
            binstd1 = binstd.copy()
        elif i ==1:
            binstd2 = binstd.copy()
        binsem,_,_=stats.binned_statistic(truetemp,
                                          dif,
                                          statistic=sem,
                                          bins=bins,
                                          range=[lower,upper])
        
        #Plot scatterplot and errorbars
        axs[0].scatter(xcenters,
                       binstd,
                       label=prednames[i])
        axs[0].errorbar(xcenters,
                        binstd,
                        yerr=binsem,
                        xerr=width * .5,
                        ls='none')
        
    #Set scatterplot legend
    axs[0].legend()
    
    #Plot ratio subplot for first 2 data sets
    axs[1].scatter(xcenters, binstd1 / binstd2)
    axs[1].axhline(y=1)
    axs[1].set_xlim([lower,upper])
    axs[1].set_ylim([0.5,2.0])
    axs[1].set_ylabel('%s / %s' % (prednames[0],
                                   prednames[1]))
    axs[1].set_xlabel('%s' % varname)
    fig.savefig('./Plots/%s/Comparison/%s/%s_RMS Comparison Plot' % (modelname, 
                                                                     space, 
                                                                     varname), bbox_inches='tight')
    plt.close()

#2D Histogram plot or heatmap
def heatmap(modelname, modelorreco, space, true, pred, varname, lower, upper, bins):
    #Get histogram dimensions and data using numpy
    heatmap, xedges, yedges = np.histogram2d(true, 
                                             pred, 
                                             bins=bins, 
                                             range=[[lower, upper], [lower, upper]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    #Plot heatmap
    plt.imshow(heatmap.T, 
               extent=extent, 
               origin='lower')
    plt.plot([lower, upper], 
             [lower, upper], 
             color='blue')
    fig = plt.gcf()
    plt.set_cmap('gist_heat_r')
    plt.xlabel('%s True' % varname)
    plt.ylabel('%s Pred' % varname)
    plt.title('Frequency Heatmap')
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.colorbar()
    fig.savefig('./Plots/%s/%s/%s/%s Heatmap' % (modelname, 
                                                 modelorreco, 
                                                 space, 
                                                 varname))
    plt.close()