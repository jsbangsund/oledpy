# Function to calculate exciton density for a given:
# exciton lifetime (tau, s), PL efficiency (eta_PL)
# luminance (L, cd/m2), and eml width (w, nm)
import numpy as np
from scipy.optimize import curve_fit, leastsq
import pandas as pd
import os
module_dir = os.path.dirname(__file__)
import datetime
# Planck's constant (in J-s), speed of light (in m/s), elementary charge (in C)
from scipy.constants import h,c,e
q=e

def calc_exciton_density(L,tau,eta_PL,eta_OC,w,spectrum):
    '''
    Function to calculate exciton density within a device at a given luminance, L,
    for a device with:
    tau = exciton lifetime in seconds, typically ~1e-6 s
    eta_PL = photoluminescence efficiency, typically ~0.8-0.95
    eta_OC = outcoupling efficiency, typically ~0.2
    w = recombination zone width, assuming a flat exciton profile
        As a first approximation, this can be set equal to the emissive layer thickness
    spectrum = two column array of device EL spectrum (wavelength,intensity)
    **Assumptions**:
    - Exciton profile is flat
    - Device emission is lambertian
    '''
    k_r = eta_PL / tau # 1/s, radiative rate
    # wavelength range
    wavelength = np.arange(300,800,1) * 1e-9 # m
    # Import photopic response
    photopic = import_photopic_response()
    V = np.interp(wavelength * 1e9,photopic[:,0],photopic[:,1])
    # Normalize spectrum
    S = np.interp(wavelength,spectrum[:,0],spectrum[:,1])
    S = S / np.trapz(S,wavelength)
    # Convert L and w to cm
    L = L * 1e-4 # cd/cm^2
    w = w * 1e-7 # cm
    # Calculate photon flux, assuming lambertian profile
    F_p = L * np.pi / np.trapz(V * S * h * c / wavelength, wavelength)
    # Calculate exciton density assuming a flat recombination zone
    N_flat = F_p / (w * k_r * eta_OC)
    return N_flat

def fit_stretched_exp(time,EL):
    '''
    Fit OLED lifetime decay with stretched exponential decay
    '''
    def stretchedExp(t,tau,beta):
        return np.exp(-(t/tau)**beta)
    popt,pcov=curve_fit(stretchedExp,time,EL,p0=[100,0.5])
    tau,beta=popt
    fitEL = stretchedExp(time,*popt)
    return tau,beta,fit_EL

def calc_EQE(voltage,I_OLED,I_detector,deviceArea_cm2,
            spectrum,dev,sub):
    '''
    Calculate OLED external quantum efficiency from measurements of device current,
        photodetector current, and OLED EL spectrum
    This assumes lambertian emission and that the photodetector collects all outcoupled light
    If growthID is passed, fetch spectra from database
    If growthID not passed, must pass spectrum
    I_OLED and I_detector should be current in Amps
    Can supply dev and sub and use these to pull voltage, I_OLED, and I_detector
        In this case, pass None or 0 for voltage, I_OLED, I_detector, deviceArea_cm2
    spectrum is assumed to be a two column array of [wavelengths,intensities]
        note that units should be intensity (power/area/wavelength), NOT counts
    '''
    photopic = import_photopic_response()
    # Use photopic wavelength range as range of interest
    wavelength = photopic[:,0]
    # Import detector responsivity
    responsivity = import_responsivity()
    # Convert to numpy arrays and baseline
    voltage = np.array(voltage)
    I_OLED = np.array(I_OLED)
    I_detector = np.array(I_detector)
    # Subtract floor below 1 V
    # Find index of 1 V
    V_1V_idx = np.abs(voltage - 1).argmin()
    detectorFloor = np.mean(I_detector[:V_1V_idx])
    I_detector = I_detector-detectorFloor
    # interpolate data
    S = np.interp(wavelength,spectrum[:,0],spectrum[:,1])
    V = photopic[:,1]
    R = np.interp(wavelength,responsivity[:,0],responsivity[:,1])
    R_avg = (np.trapz(R*S*(h*c/(wavelength*1e-9))/q,wavelength*1e-9)
                    /np.trapz(S,wavelength*1e-9))
    photon_flux = (I_detector/q)/R_avg
    EQE = photon_flux / (I_OLED / q) * 100
    # Current Density mA/cm^2
    J_OLED = I_OLED * 1e3 / deviceArea_cm2; # mA/cm^2
    # Calculate Power Efficiency and Luminance
    # Scale spectrum to give units of photons / nm-s
    photons_scaling = photon_flux / np.trapz(S,wavelength * 1e-9)
    spectrum_photons = np.outer(photons_scaling,S) # vs. wavelength in columns
    luminous_flux = np.trapz(spectrum_photons * V *
                            h*c/(wavelength*1e-9),wavelength*1e-9)
    optical_power = np.trapz(spectrum_photons * h*c/(wavelength * 1e-9),
                               wavelength * 1e-9)# W
    luminous_power_efficiency = luminous_flux / (I_OLED * voltage) # lm/W (aka luminous efficacy)
    optical_power_efficiency = optical_power / (I_OLED * voltage) * 100 # (%)
    # factor of pi conversion from lm -> lm/sr (cd), assuming lambertian
    # emission. See Forrest, S.R., Bradley, D.D.C. & Thompson, M.E.
    # Measuring the Efficiency of Organic Light-Emitting Devices. Adv. Mater. 15, 1043–1048 (2003).
    luminance = luminous_flux / (deviceArea_cm2 * 1e-4) / np.pi # cd/m^2 (aka brightness)
    current_efficiency = luminance / (J_OLED * 1e4 * 1e-3) # cd/A
    return {'EQE':EQE,'luminance':luminance,'photon_flux':photon_flux,
            'luminous_flux':luminous_flux,'current_efficiency':current_efficiency,
            'optical_power_density':optical_power*1e3/deviceArea_cm2,#mW/cm2
            'optical_power_efficiency':optical_power_efficiency,
            'luminous_power_efficiency':luminous_power_efficiency}

def import_responsivity():
    return np.genfromtxt(os.path.join(module_dir,'detector_responsivity.csv'),
                            delimiter=',', skip_header=1)

def import_photopic_response():
    return np.genfromtxt(os.path.join(module_dir,'photopic_response.csv'),
                            delimiter=',', skip_header=1)

# Calculating confidence interval (CI) across a curve
import scipy.stats as stats
def get_mean_and_CI(x_list,y_list,CI=0.95,real_mean=True,
                    log_spaced=False,label=None,ax=None,color=None,legend_label=False,
                    return_rep_idx = False):
    '''
    Function to calculate continuous mean and CI
    for a list of x and y data sets
    If 'ax' is passed, mean and CI will be plotted
    if log_spaced=True, log-spaced x array will be used
        Be careful--if x data isn't logarithmic, issues
        can arise
    if real_mean=False, with find actual curve which is
        most representative of the mean. Otherwise, it
        will calculate the actual mean, which could appear
        unphysical
    '''
    # Remove inf and nan
    for idx in range(0,len(x_list)):
        x = np.array(x_list[idx])
        y = np.array(y_list[idx])
        # Addition is the equivalent of or, multiplication of and
        x_list[idx] = x[~(np.isinf(x)+np.isinf(y)+np.isnan(x)+np.isnan(y))]
        y_list[idx] = y[~(np.isinf(x)+np.isinf(y)+np.isnan(x)+np.isnan(y))]
    # Find lengths, and min and max values
    lengths = np.array([len(x) for x in x_list])
    min_x = [np.amin(x) for x in x_list]
    max_x = [np.amax(x) for x in x_list]
    # Get array of x values that range from the minimum
    # to maximum value in all of the x arrays
    # with a length equal to the max length
    # logarithmically space, since our J data is
    # roughly log spaced in general
    # (due to roughly exponential behavior J vs. V)
    if log_spaced:
        # re-find minimum positive value
        min_x = [np.amin(np.abs(x[x>0])) for x in x_list]
        x_interp = np.exp(np.linspace(np.log(np.amin(min_x)),
                                      np.log(np.amax(max_x)),
                                      np.amax(lengths)))
    else:
        x_interp = np.linspace(np.amin(min_x),np.amax(max_x),
                               np.amax(lengths))
    # Could just use the raw x data, but it may not be monotonic
    y_array = np.empty((len(x_list),len(x_interp)))
    for idx,x in enumerate(x_list):
        y_array[idx,:] = np.interp(x_interp,x,y_list[idx],left=np.nan,right=np.nan)
    y_mean = np.nanmean(y_array,axis=0) # ignore nan when calculating mean
    if not real_mean:
        # Find most representative curve, both in length and difference from mean
        diff_from_mean = np.abs(y_array-y_mean)
        diff_sums = np.nansum(diff_from_mean,axis=1)
        # Remove rows that are less than 80% of total length (based on nan)
        not_nan_count = np.count_nonzero(~np.isnan(y_array),axis=1)
        if type(not_nan_count) is int:
            not_nan_count = np.array([not_nan_count])
        min_length_idx = np.argwhere(not_nan_count/len(x_interp)>=0.8)
        min_length_idx = np.argwhere(not_nan_count/len(x_interp)>=0.8)
        diff_sums = diff_sums[min_length_idx]
        diff_sums_norm_by_length = diff_sums*len(x_interp)/(not_nan_count[min_length_idx])
        most_rep_idx=np.argmin(diff_sums_norm_by_length)
        y_mean = y_array[most_rep_idx,:]

    # get confidence intervals
    y_sem = stats.sem(y_array,axis=0,nan_policy='omit') # standard error
    # sample count (shape parameter df in stats.t.interval)
    count = np.count_nonzero(~np.isnan(y_array),axis=0) - 1
    # Find where count is one or less turn into nan
    one_idx = np.where(count <= 1)[0]
    upper,lower = stats.t.interval(CI, count, loc=y_mean, scale=y_sem)
    upper[one_idx]=np.nan
    lower[one_idx]=np.nan
    if not color:
        color = 'k'
    if ax:
        ax.semilogx(x_interp,y_mean,label=label,color=color)
        if legend_label:
            ax.fill_between(x_interp, upper, lower,
                            label='{:.0f}'.format(CI*100)+'% CI',
                            alpha=0.2, color=color)
        else:
            ax.fill_between(x_interp, upper, lower,
                            alpha=0.2, color=color)
    if return_rep_idx and not real_mean:
        return x_interp,y_mean,upper,lower,most_rep_idx
    else:
        return x_interp,y_mean,upper,lower

##### Cleaning, logspacing data
# Clean and filter helpers
# Log-spaced indices
def logspaced_indices(x, n=500):
    # generates n logarithmically spaced indices of vector x
    average_step = (np.amax(x) - np.amin(x))/len(x)
    min_exp = np.log(np.amin(x[x > 0]))
    max_exp = np.log(np.amax(x))
    idx = (np.exp(np.linspace(min_exp, max_exp, n))
             / average_step )
    # Round to integer
    idx = idx.astype(int)
    # Remove indices <0 or >len(x)
    idx = np.delete(idx,np.where(idx < 0) )
    idx = np.delete(idx,np.where(idx>x.size-1))
    # Add first and last indices
    idx = np.append(0,idx)
    idx = np.append(idx,x.size-1)
    # Remove duplicates
    idx = np.unique(idx)
    return idx

def logspaced_x(x,n=500):
    x=np.array(x)
    x_interp = np.exp(np.linspace(np.log(np.amin(x[x > 0])),
                                  np.log(np.amax(x)),
                                  n))
    return x_interp

def cleanEQE(EQE,threshold=4,return_index=False):
    # Similar to cleanSignal_curvature, but preserves the length of the array
    # i.e. end points are not trimmed. Threshold default is also better tuned to EQE.
    EQE_right = np.roll(EQE,1)
    EQE_right[0]=EQE_right[1]
    diff1 = EQE_right - EQE
    diff1_right = np.roll(diff1,1)
    diff1_right[0] = diff1_right[1]
    diff2 = np.abs(diff1_right - diff1)
    filter_idx = np.where(diff2<threshold)[0]
    if return_index:
        return EQE[filter_idx],filter_idx
    else:
        return EQE[filter_idx]

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except:
    print('statsmodels not installed. cleanEQE_lowess will not work')
def cleanEQE_lowess(x,EQE,diff_threshold=0.035,lowess_frac=0.09,x_lower_limit=3,
                  return_index=False,return_filter=False):
    '''
    This function uses Locally Weighted Scatterplot Smoothing (lowess) to filter out
    noise in a typical EQE curve. This should also work well for lifetime, but isn't tested
    See description at http://www.cs.sfu.ca/~ggbaker/data-science/content/filtering.html
    and https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    x is desired x-coordinate (e.g. Voltage, current density, luminance)
    EQE is EQE data
    diff_threshold is the maximum difference between the smoothed curve and the raw data
        that will be passed through the filter (returned)
    lowess_frac represents the window size
    x_lower_limit prevents filtering of points below some x-value
        This is intended to avoid filtering noisy data at very low currents and voltages
        and/or avoid filtering data points in the steep part of the roll-off
        
    This function can be a little unstable with large discontinuities, and is very sensitive
    to the lowess_frac parameter. For some reason in the range of 0.06 - 0.08 it is more 
    unstable. At and above 0.1, it can tend to start filtering out points in the roll-up
    This is where x_lower_limit is important.
    '''
    # Get the LOWESS filtered curve
    filtered = lowess(EQE, x, frac=lowess_frac)
    diff = np.abs(EQE-filtered[:,1])/np.amax(EQE)
    filter_idx = np.where(np.logical_not(
        np.logical_and(diff>diff_threshold,x>x_lower_limit)))[0]
    if return_index and not return_filter:
        return x[filter_idx],EQE[filter_idx],filter_idx
    elif return_index and return_filter:
        return x[filter_idx],EQE[filter_idx],filtered,filter_idx
    elif not return_index and return_filter:
        return x[filter_idx],EQE[filter_idx],filtered
    else:
        return x[filter_idx],EQE[filter_idx]
    
def cleanSignal_curvature(x,y,curvature_threshold = 0.004,
                          return_index=False,remove_less_than=None):
    # Could improve this function by filtering outliers using deviation from median rather than threshold
    # # see https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    # https://www.mathworks.com/matlabcentral/fileexchange/3961-deleteoutliers
    # Could also use np.roll so data isn't lost (right now two datapoints are removed)
    # May need to create separate cleaning routines for lifetime and IVL data,
    # as the types of noise that are seen differ in the two cases
    x = np.array(x)
    y = np.array(y)

    # Find regions where there is relatively little noise
    # Assess this by taking the second derivative to find
    # where there is limited change in slope
    low_curvature_idx = np.where(np.abs(np.diff(y,n=2))
                                 <curvature_threshold)[0]+1
    # Now find window where adjacent points satisfy low curvature
    # taking the diff of low_curvature_idx, adjacent points have diff = 1
    adjacent_idx = np.where(np.diff(low_curvature_idx)==1)[0]+1
    # Now specify the indices of the initial array to return
    filter_idx = low_curvature_idx[adjacent_idx]

    # If minimum threshold value is selected, remove points below:
    if remove_less_than:
        less_idx = np.where(y>remove_less_than)[0]
        filter_idx = np.intersect1d(filter_idx,less_idx)
    if return_index:
        return x[filter_idx],y[filter_idx],filter_idx
    else:
        return x[filter_idx],y[filter_idx]

def cleanSignal(x,y,threshold=0.05,return_index=False,remove_less_than=None):
    '''
    This function is meant to clean spurious points from lifetime curves due to contact 
    issues or external light exposure to the photodetector.
    '''
    x = np.array(x)
    y = np.array(y)
    # normalize by third highest point to avoid normalizing to outliers
    if len(y)>3:
        diff_array = np.diff(y)/np.sort(y)[-3]
    else:
        diff_array = np.diff(y)/np.sort(y)[-1]
    # Use np.roll to shift elements by one
    # Then substract by the sign, to identify a sign change
    # Catches sign change after outlier
    diff_sign = np.sign(diff_array)
    signchange = (( (np.roll(diff_sign, 1) - diff_sign) != 0 )
                    .astype(int))
    signchange[0]=0
    sign_idx = np.where(signchange==1)[0]
    rm_idx = np.where(np.abs(diff_array)>threshold)[0]+1
    # Find the overlap of where there is a sign change
    # in the diff array, and diff is above threshold
    rm_idx=np.intersect1d(sign_idx,rm_idx)

    # Now roll the other way, catching sign change before outlier
    signchange = (( (np.roll(diff_sign, -1) - diff_sign) != 0 )
                    .astype(int))
    signchange[0]=0
    sign_idx = np.where(signchange==1)[0]
    rm_idx_2 = np.where(np.abs(diff_array)>threshold)[0]+1
    rm_idx = np.append(rm_idx,np.intersect1d(sign_idx,rm_idx_2))
    # Also remove the last element if needed, since above doesn't
    # catch issues with the last point
    # This doesn't work right now
    if signchange[-2]==1 and np.abs(diff_array[-1])>threshold:
        rm_idx = np.append(rm_idx,len(x)-1)
    # If min threshold value is selected, remove:
    if remove_less_than:
        rm_idx_2 = np.where(y<remove_less_than)[0]
        rm_idx = np.append(rm_idx,rm_idx_2)
    y = np.delete(y,rm_idx)
    x = np.delete(x,rm_idx)
    if return_index:
        return x,y,rm_idx
    else:
        return x,y

# Get selected tx and add to dataframe
def  extractTx(OnTime,NormSignal,tx_points,test_type=''):
    '''
    Extract tx for lifetime curves, where tx is the time it takes to reach x% of the 
    initial brightness
    tx_points should be a list of percentages. E.g. [95,80,50]
    '''
    def stretchedExp(t,tau,beta):
        return np.exp(-(t/tau)**beta)
    # type should be a string of either 'EL', 'PL', or 'CB'
    tx = {}
    tx_fit = {}
    key_string_list=[]
    try:
        popt,pcov=curve_fit(stretchedExp,OnTime,NormSignal,p0=[100,0.5])
        tau,beta=popt
        fitSuccess = True
    except:
        tau=0
        beta=0
        fitSuccess = False
        print('Not Fit')
    # Find tx
    for x_idx,x in enumerate(tx_points):
        # key string for dict, e.g. 'ELt50'
        fraction = x/100
        key_string = test_type + 't' + "{:.0f}".format(x)
        key_string_list.append(key_string)
        # If tx exists, find last point that is closest to x
        # This way, if the device signal goes down temporarily
        # and comes back up, an artificially low lifetime
        # will not be recorded
        # Use a range of tolerances, starting at 0.1%, and 
        # if no points are found, increase tolerance up to 1%
        NormSignal=np.array(NormSignal)
        tolerances = np.arange(0.001,0.01,0.001)
        tx[key_string]= np.nan # If nothing is found, label as NaN
        for tol in tolerances:
            tx_idx = np.where(np.abs(NormSignal - fraction)<tol)[0]
            if tx_idx.size == 0:
                # If nothing found, move up to higher tolerance
                continue
            else:
                # If something is found, take last point and exit loop
                tx[key_string] = OnTime[tx_idx[-1]]
                break
        # If still nothing has been found, try previous method of finding
        # second point below value
        if np.isnan(tx[key_string]):
            indices = np.where(NormSignal<fraction)[0]
            if len(indices)>2:
                tx[key_string] = OnTime[indices[1]]
            elif len(indices):
                tx[key_string] = OnTime[indices[0]]
        # If fit was successful, record fit-extracted tx
        if fitSuccess:
            tx_fit[key_string] = tau*(-np.log(fraction))**(1/beta)
        else:
            tx_fit[key_string] = np.nan

    return {'tx':tx, 'tx_fit':tx_fit,
            'tau':tau,'beta':beta, 'key_string':key_string_list}
            
def power_law(L0,C,n):
    return C*L0**(-n)
def log_error_fun_tx_vs_L0(params,*args):
    # This function defines logarithmic error for tx vs L0 fits
    L0 = args[0]
    tx = args[1]
    C=params[0]
    n=params[1]
    fit = power_law(L0,C,n)
    return np.log(tx) - np.log(fit)
def fit_tx_vs_L0(L0,tx,error='log',p0=[100,1.8]):
    # error can be 'standard' or 'log'
    # 'log' error can help equalize weighting of lifetimes at high and low luminances,
    # since the magnitudes lifetimes at these ranges differs so widely
    # p0 = [C,n] are initial parameter guesses for the prefactor and acceleration factor
    if error == 'standard':
        popt,pcov=curve_fit(power_law,L0,tx,p0=p0)
        C,n=popt
    elif error == 'log':
        result=leastsq(log_error_fun_tx_vs_L0,p0,args=(L0,tx))
        C=result[0][0]
        n=result[0][1]
    return C,n
