#import oledpy.tmm
import oledpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# Planck's constant (in J-s), speed of light (in m/s), elementary charge (in C)
from scipy.constants import h,c,e
from numpy.lib.scimath import sqrt as csqrt # for complex arguments
# Import package for progressbar, and try to detect whether in notebook or console environment
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except ImportError:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        print('tqdm not available')
else:
    try:
        from tqdm import tqdm_notebook as tqdm
    except ModuleNotFoundError:
        print('tqdm not available')
'''
The most general use of this code is to calculate the light outcoupling
efficiency for a multilayer thin film stack, based on the emission spectrum and
emitter position within the stack.

In reporting outcoupling calculations, there are several common plots which are provided:
Power dissipation spectrum (single wavelength curve vs. u)
Power dissipation spectrum (2d heat map vs wavelength and u)
Mode analysis (Breakdown of mode losses: waveguided, surface plasmon, outcoupled, etc.)
    This is often plotted against a varied layer thickness within the stack
Outcoupling vs. dipole position within the active layer
    This is useful for calculating the overall outcoupling efficiency given a measured
    or otherwise simulated recombination zone profile (i.e. the spatial dependence of
    exciton density in an OLED)
Purcell factor (impact on excited state lifetime)
PL vs. angle for measuring emitter orientation

References:
1. Furno, M.; Meerheim, R.; Hofmann, S.; Lüssem, B.; Leo, K.
   Efficiency and Rate of Spontaneous Emission in Organic Electroluminescent Devices.
   Phys. Rev. B 2012, 85 (11), 115205.

2. Furno, M.; Meerheim, R.; Thomschke, M.; Hofmann, S.; Lüssem, B.; Leo, K.
   Outcoupling Efficiency in Small-Molecule OLEDs: From Theory to Experiment.
   In Proc. SPIE; 2010; Vol. 7617, p 761716.

3. Neyts, K. A. Simulation of Light Emission from Thin-Film Microcavities.
   J. Opt. Soc. Am. A, JOSAA 1998, 15 (4), 962–971.

4. Byrnes, Steven. Multilayer Optical Calculations, 2016.
   https://arxiv.org/abs/1603.02720

5. Pettersson, Leif AA, Lucimara S. Roman, and Olle Inganäs.
   Modeling photocurrent action spectra of photovoltaic devices based on organic thin films.
   Journal of Applied Physics 86.1 (1999): 487-496.

6. Kim, J., Kang, K., Kim, K. Y., & Kim, J. (2017).
   Origin of a sharp spectral peak near the critical angle in the spectral power
   density profile of top-emitting organic light-emitting diodes.
   Japanese Journal of Applied Physics, 57(1), 012101.
   https://iopscience.iop.org/article/10.7567/JJAP.57.012101/meta

7. Salehi, A., Ho, S., Chen, Y., Peng, C., Yersin, H., and So, F.
   Highly Efficient Organic Light-Emitting Diode Using A Low Refractive Index
   Electron Transport Layer. Adv. Optical Mater. 2017, 5, 1700197
   http://dx.doi.org/10.1002/adom.201700197
'''
def multiply_nd_by_1d(array_nd,array_1d,axis=0):
    ''' Multiplies n-dimensional array by 1D array along specified axis
    axis: axis along which elementwise multiplication with broadcasting
          is to be performed
    '''
    # Create an array which would be used to reshape 1D array, b to have
    # singleton dimensions except for the given axis where we would put -1
    # signifying to use the entire length of elements along that axis
    dim_array = np.ones((1,array_nd.ndim),int).ravel()
    dim_array[axis] = -1
    # Reshape b with dim_array and perform elementwise multiplication with
    # broadcasting along the singleton dimensions for the final output
    return array_nd*array_1d.reshape(dim_array)

def plot_power_dissipation_1d(u,K_total_array,wavelength_to_plot=510,
                              ax=None,fig=None,figsize=(4.5,4),
                              ylabel='Power Dissipation $uK$ (nm$^{-1}$)',
                              xlabel='Norm. in-plane wavevector $u$'):
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    if type(wavelength_to_plot) is int:
        w_idx = np.argmin(np.abs(wavelengths-wavelength_to_plot))
        handle=ax.semilogy(u,u*K_total_array[0,w_idx,:])
    elif hasattr(wavelength_to_plot, "__len__"):
        for w in wavelength_to_plot:
            w_idx = np.argmin(np.abs(wavelengths-w))
            handle=ax.semilogy(u,u*K_total_array[0,w_idx,:],label=str(w)+' nm')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax,handle

from matplotlib import colors, ticker, cm
try:
    default_cmap = plt.get_cmap('cividis')
except:
    default_cmap = plt.get_cmap('gray')
def plot_power_dissipation_2d(u,y,K_total_array,z_idx=0,ax=None,fig=None,figsize=(4.5,4),
        min_exp=None,max_exp=None,exp_step=0.01,
        show_cbar=True,ylabel='Wavelength (nm)',xlabel='Norm. in-plane wavevector $u$',
        cmap=default_cmap):
    '''
    Common cmap options: cividis,plasma,viridis,magma,gray
    z_idx is the dipole position slice to plot, K_total_array[z_idx,:,:]
    '''
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    uK = u*K_total_array[z_idx,:,:]
    if min_exp is None:
        min_exp = np.floor(np.log10(uK[uK>0].min())-1)
    if max_exp is None:
        max_exp = np.ceil(np.log10(uK.max())+1)
    uK[np.log10(uK)<min_exp]=np.power(10,min_exp*1.0)
    uK[np.log10(uK)>max_exp]=np.power(10,max_exp*1.0)
    lev_exp = np.arange(min_exp,max_exp,exp_step)
    levs = np.power(10, lev_exp)
    cs = ax.contourf(u,y,uK, levs,
                norm=colors.LogNorm(),cmap=cmap)
    if show_cbar:
        ticks = 10**np.arange(np.floor(np.log10(uK[uK>0].min())-1),np.ceil(np.log10(uK.max())+1),1)
        cbar = plt.colorbar(cs,ticks=ticks)
        cbar.set_label('Dissipated Power')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig,ax,cs,cbar

def emission_polar_plot(thetas_degrees=None,intensity=None,
                        ax=None,fig=None,leg_label=None,
                        theta_range=[0,90],theta_dir='clockwise',
                        zero_loc="N",plot_lambertian=True,
                        theta_grid_step=15,r_grid_step=0.25,
                        r_labels_angle=0,figsize=(3,3),
                        fs_tick_labels=12,fs_axis_label=14):
    '''
    Initialize and format a polar plot for emission profile visualization
    A single data series can optionally be passed to plot
    Otherwise, the fig and ax handles can be returned for the user to add
    data series and edit formatting as they wish.
    Inputs:
    theta_range: tuple or list of min and max angles to plot
        typically [0,90] or [-90,90]
    zero_loc: cardinal direction to start zero. By convention should be "N" for OLEDs
    theta_grid_step and r_grid_step specify the spacing between grid lines
    r_labels_angle specifies the angle at which the tick labels should be located
        This doesn't seem to work atm, and appears to be an open issue with matplotlib

    Documentation of how to edit polar plots at:
    https://matplotlib.org/api/projections_api.html
    '''
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize,subplot_kw=dict(polar=True))
    if fig is None:
        fig = plt.gcf()
    if not ((thetas_degrees is None) and (intensity is None)):
        # Note that matplotlib polar plots only accept angles in radians
        handle = ax.plot(thetas_degrees*np.pi/180,intensity/np.amax(intensity),
                         'o',label=leg_label)
    if plot_lambertian:
        thetas = np.arange(theta_range[0]*np.pi/180,theta_range[1]*np.pi/180,0.01)
        lambertian_handle = ax.plot(thetas,np.cos(thetas),'--k',label='Lambertian')
    ax.set_thetamin(theta_range[0])
    ax.set_thetamax(theta_range[1])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_thetagrids(np.arange(theta_range[0],theta_range[1]+0.1,theta_grid_step),
                     fontsize=fs_tick_labels)
    ax.set_rgrids(np.arange(0,1.01,r_grid_step),angle=r_labels_angle,
                 fontsize=fs_tick_labels)
    ax.set_ylabel('EL Intensity (norm.)',fontsize=fs_axis_label)
    return fig,ax,handle,lambertian_handle

def integrate(y,x=None,axis=-1,squeeze_axis=None):
    '''
    Returns numpy.trapz(y,x) if len(x)>1
    If len(x)==1, it will reduce the dimensionality of y using numpy.squeeze
    If squeeze_axis is provided, it will squeeze along this dimension
    axis is the axis term provided to numpy.trapz, defaults to -1

    This is meant to catch situations where the integration axis has a len==1,
    which would return an array of zeros, when the intended behavior is to average,
    and hence a reduced dimensionality array should be returned
    '''
    if len(x)>1:
        return np.trapz(y,x,axis=axis)
    else:
        if squeeze_axis is None:
            return np.squeeze(y)
        else:
            return np.squeeze(y,axis=squeeze_axis)

from oledpy.tmm import interface_t_cos, interface_r_cos, make_2x2_array
def partial_tmm(pol,n_list,d_list,active_layer,u,lam_vac,direction='both'):
    '''
    This approach is described in J Kim Applied Optics 2018 and
    Pettersson JAP 1999.
    See Eqn. 7,8,13-16 in Kim 2018
    See Eqn. 11-17 in Pettersson 1999

    pol = 'p' or 's'
    n_list = list of complex refractive indices of each layer
    d_list = list of thickness of each layer
    u = in-plane wavevector in the active layer (emissive layer)
    lam_vac = free-space wavelength of light to calculate with
    direction = 'both', 'top', or 'bottom', specifies whether to calculate the top
        or bottom half of the stack, or both
    '''
    n_list = np.array(n_list)
    d_list = np.array(d_list)
    n_e = n_list[active_layer].real
    num_layers = len(n_list)
    # Initialize variables
    r_down = 0
    t_down = 0
    r_up = 0
    t_up = 0
    t_list = np.zeros((num_layers, num_layers), dtype=complex)
    r_list = np.zeros((num_layers, num_layers), dtype=complex)
    I_list = np.zeros((num_layers, 2, 2), dtype=complex)
    L_list = np.zeros((num_layers, 2, 2), dtype=complex)
    M_list = np.zeros((num_layers, 2, 2), dtype=complex)
    cosTheta_list = csqrt(1 - (n_e/n_list)**2 * u**2)
    kz_list = 2 * np.pi / lam_vac * n_list * cosTheta_list
    delta = kz_list * d_list
    # Calculate bottom half of stack
    if (direction == 'both') or (direction == 'bottom'):
        S_down = make_2x2_array(1, 0, 0, 1, dtype=complex)
        for j in range(0,active_layer):
            # t and r are shared notation for pettersson and byrnes
            t_list[j,j+1] = interface_t_cos(pol, n_list[j], n_list[j+1],
                                        cosTheta_list[j], cosTheta_list[j+1])
            r_list[j,j+1] = interface_r_cos(pol, n_list[j], n_list[j+1],
                                        cosTheta_list[j], cosTheta_list[j+1])
            # interface matrix, eqn. 1 pettersson
            I_list[j] = 1/t_list[j,j+1] * make_2x2_array(1,r_list[j,j+1],
                                                         r_list[j,j+1],1,
                                                         dtype=complex)
            # M and L are not defined for the 0th layer
            # i.e. the substrate or ambient is incoherent
            if j==0:
                # Pre-factor in byrnes eqn 13
                S_down = np.dot(I_list[j],S_down)
            if j>0:
                # Layer matrix (phase matrix), eqn. 5 pettersson
                L_list[j] = make_2x2_array(np.exp(-1j*delta[j]),0,
                                    0,np.exp(1j*delta[j]),dtype=complex)
                # M matrix (byrnes eqn. 11)
                M_list[j] = np.dot(L_list[j],I_list[j])
                # Mtilde byrnes eqn. 13
                S_down = np.dot(S_down,M_list[j])
        # Kim 2018, Eqn. 13
        r_down = - S_down[0,1] / S_down[0,0]
        t_down = (S_down[0,0]*S_down[1,1]
                  - S_down[0,1]*S_down[1,0])/S_down[0,0]
    # Kim 2018 eqn. 7
    #S_down = np.dot(S_down, make_2x2_array(
    #        np.exp(-1j*delta[active_layer]*dipole_position),0,
    #        0,np.exp(1j*delta[active_layer]*dipole_position),dtype=complex))
    # Now top half of stack
    if (direction == 'both') or (direction == 'top'):
        S_up = make_2x2_array(1, 0, 0, 1, dtype=complex)
        for j in range(active_layer,num_layers-1):
            # t and r are shared notation for pettersson and byrnes
            t_list[j,j+1] = interface_t_cos(pol, n_list[j], n_list[j+1],
                                        cosTheta_list[j], cosTheta_list[j+1])
            r_list[j,j+1] = interface_r_cos(pol, n_list[j], n_list[j+1],
                                        cosTheta_list[j], cosTheta_list[j+1])

            # interface matrix, eqn. 1 pettersson
            I_list[j] = 1/t_list[j,j+1] * make_2x2_array(1,r_list[j,j+1],
                                                         r_list[j,j+1],1,
                                                         dtype=complex)
            # M and L are not defined for the 0th layer
            # i.e. the substrate or ambient is incoherent
            if j==active_layer:
                #delta_EML = kz_list[j]*d_list[j]*(1-dipole_position)
                #L = make_2x2_array(np.exp(-1j*delta_EML),0,
                #                        0,np.exp(1j*delta_EML),dtype=complex)
                #S_up = np.dot(L,I_list[j])
                S_up = np.dot(S_up,I_list[j])
            if j>active_layer:
                # Layer matrix (phase matrix), eqn. 5 pettersson
                L_list[j] = make_2x2_array(np.exp(-1j*delta[j]),0,
                                    0,np.exp(1j*delta[j]),dtype=complex)
                # M matrix (byrnes eqn. 11)
                M_list[j] = np.dot(L_list[j],I_list[j])
                # Mtilde byrnes eqn. 13
                S_up = np.dot(S_up,M_list[j])
        # Kim 2018, Eqn. 13
        r_up = S_up[1,0] / S_up[0,0]
        t_up = 1 / S_up[0,0]
    return {'r_down':r_down,'t_down':t_down,'r_up':r_up,'t_up':t_up}

class ThinFilmArchitecture(object):
    ############################################################################
    # Architecture and materials properties related functions
    def __init__(self,layer_names=[],d=[],n=[],doping=[],active_layer=0, coherence=[],
                 vac_wavelengths=[],pl_spectrum=[],thetas=[],dipole_positions=[],RZ=None,
                 k_r=None,k_nr=None,tau=None,PLQY=None,
                 n_medium=1,anisotropy_factor=0.3333,u_step=0.001,u_stop=2,layer_dict_list=None,
                 show_progress_bar=True,show_wavelength_progress_bar=True):
        '''
        n: array of complex indices of refraction for each layer
            should have n.shape=(len(layer_names),len(vac_wavelengths))
        d: list of thicknesses for each layer
        doping: Doping of each layer, used for calculating mixed layer
            optical constants by weighting their neat components
        dipole_positions: list of positions of dipoles as a fraction of the
            active layer thickness (ranges from 0 to 1)
            starts from the bottom of the layer i.e. the relative distance
            from the HTL in a conventional, bottom-emitting OLED
        active_layer: index (i.e. 0,1,2) for active layer position in stack
        anisotropy_factor: describes dipole orientation
            0 is horizontal, 0.33 is isotropic, and 1 is vertical
            See https://doi.org/10.1103/PhysRevApplied.8.037001 for discussion
        n_medium: index of refraction of the far-field medium (usually air)
        coherency: list of booleans for whether each layer is coherent or not
            this is not yet implemented, so specifying this won't have any effect
        RZ is the recombination zone profile, i.e. the spatial profile of the electrically
            generated exciton density.
            Must have same length as the dipole positions in self.dipole_positions
        pl_spectrum is the photoluminescence spectrum of the emitter
            same length as self.vac_wavelengths
        Pass at least one of these two pairs of variables:
        k_r and k_nr: The intrinsic radiative and non-radiative rates of the emitter outside a cavity
        tau and PLQY: The intrinsic exciton lifetime and photoluminescence quantum yield of the emitter
        '''
        # Initialize attributes
        if not layer_dict_list is None:
            self.load_from_dict_list(layer_dict_list)
        else:
            self.layer_names = layer_names
            self.d = np.array(d)
            self.set_doping(doping) 
            # Currently, calculating with incoherent layers is unsupported:
            self.coherence = coherence # List of layer coherency booleans
            self.set_active_layer(active_layer)
        if not hasattr(self,'active_layer'):
            print('No active layer specified.\n'+
                   ' To set, call arch.set_active_layer(active_layer) Indexing starts at 0.')
        self.n = np.array(n) # complex indices of refraction of each layer
        self.vac_wavelengths = np.array(vac_wavelengths) # vacuum wavelengths
        self.dipole_positions = np.array(dipole_positions)
        self.set_RZ(RZ)
        self.set_pl_spectrum(pl_spectrum)
        self.thetas = np.array(thetas) # Angles in exterior medium
        self.n_medium = n_medium
        self.anisotropy_factor = anisotropy_factor
        self.u_step = u_step
        self.u_stop = u_stop
        self.show_progress_bar = show_progress_bar
        self.show_wavelength_progress_bar = show_wavelength_progress_bar
        # Radiative and non-radiative rate intrinsic to emitter
        self.k_r = k_r
        self.k_nr = k_nr
        self.tau = tau
        self.PLQY = PLQY
        
    def load_from_dict_list(self,layer_dict_list):
        '''
       layer_dict_list should have the form of:
       layers = [
             {'name':'SiO2'     ,'d':0  ,'doping':1,'coherent':0},
             {'name':'ITO'      ,'d':100,'doping':1,'coherent':1},
             {'name':'HTL'      ,'d':70 ,'doping':1,'coherent':1},
             {'name':'ETL'      ,'d':20 ,'doping':1,'coherent':1,'active':1},
             {'name':'Al'       ,'d':100,'doping':1,'coherent':1},
             {'name':'Air'      ,'d':0  ,'doping':1,'coherent':0},
        ]
       '''
        self.layer_names = [l['name'] for l in layer_dict_list]
        self.d = np.array([l['d'] for l in layer_dict_list])
        self.set_doping([l['doping'] for l in layer_dict_list])
        self.coherence = [1]*len(self.layer_names)
        for i,layer in enumerate(layer_dict_list):
            if 'active' in layer.keys():
                if layer['active']==1 or layer['active']==True:
                    self.set_active_layer(i)
            if 'coherent' in layer.keys():
                self.coherence[i] = layer['coherent']
            
    def run_attribute_checks(self):
        '''
        Check that all required attributes exist and are of consistent length
        '''
        # Dipole positions should be fraction of active layer thickness
        if np.any(np.logical_or(self.dipole_positions>1,self.dipole_positions<0)):
            raise ValueError('Dipole positions should range from 0 to 1')
        # Anisotropy factor ranges from 0 to 1
        if not np.logical_and(self.anisotropy_factor<=1,self.anisotropy_factor>=0):
            raise ValueError('Anisotropy factor should range from 0 to 1')
        # Check that shapes matter across variables
        if not (self.n.shape[0] == len(self.d)):
            raise ValueError('n and d not the same length!')
        if not (self.n.shape[1] == self.vac_wavelengths.shape[0]):
            raise ValueError('n and vac_wavelengths not the same length! Try re-loading n')
        if not (len(self.doping) == len(self.d)):
            print('d and doping not same length! May want to check indices of refraction')
        if not (len(self.dipole_positions) == len(self.RZ)):
            self.set_RZ(self.RZ)
        if not self.pl_spectrum.shape == self.vac_wavelengths.shape:
            # TODO Could add in interpolation here if pl_spectrum has two columns
            print('PL spectrum not same length as wavelengths!')
            print('Interpolate the PL spectrum to the same shape')
        if not self.spectral_weights.shape == self.pl_spectrum.shape:
            self.set_pl_spectrum(self.pl_spectrum)
        if 'K_out_array' in self.__dict__.keys():
            if not self.K_out_array.shape[0]==len(self.dipole_positions):
                print('K_out, dipole_positions dim mismatch')
            if not self.K_out_array.shape[1]==len(self.vac_wavelengths):
                print('K_out, wavelength dim mismatch')
            if not self.K_out_array.shape[2]==len(self.u):
                print('K_out, u dim mismatch')
    def set_active_layer(self,active_layer):
        self.active_layer = active_layer
        # Check that user specified correct active_layer
        if self.layer_names==[]:
            print('Layer names not yet set.')
        else:
            try:
                print('Active layer is: ' + self.layer_names[self.active_layer] +
                      '. \n To change, call arch.set_active_layer(active_layer)'
                      + ' Indexing starts at 0.')
            except IndexError:
                print('Active layer index is out of range')
    def set_dipole_positions(self,dipole_positions,RZ=None):
        self.dipole_positions = np.array(dipole_positions)
        # Try setting the current RZ to see if the shape matches
        if RZ is None:
            print('Dipole positions changed. Remember to update RZ')
            self.set_RZ(self.RZ)
        else:
            self.set_RZ(RZ)
    def set_RZ(self,RZ):
        if not RZ is None:
            if len(RZ)==len(self.dipole_positions):
                self.RZ = np.array(RZ)
            else:
                print('RZ and dipole_positions not same length! Flat RZ will be assumed')
                self.RZ = np.ones(self.dipole_positions.shape)
        else:
            print('RZ not provided. Flat RZ will be assumed')
            self.RZ = np.ones(self.dipole_positions.shape)
        self.RZ_weights = self.RZ / integrate(self.RZ,self.dipole_positions)
    def set_vac_wavelengths(self,vac_wavelengths,pl_spectrum=None):
        print('Wavelengths changed. Remember to reload refractive indices and \n' +
              'a pl spectrum for same wavelength range!')
        self.vac_wavelengths = np.array(vac_wavelengths)
        # If no pl_spectrum is passed, try setting the current PL spectrum
        # If doesn't match shape, will assume flat emission and warn user
        if pl_spectrum is None:
            self.set_pl_spectrum(self.pl_spectrum)
        else:
            self.set_pl_spectrum(pl_spectrum)
    def set_pl_spectrum(self,pl_spectrum):
        if len(pl_spectrum) != len(self.vac_wavelengths):
            if len(pl_spectrum)==0:
                print('No PL spectrum provided. Flat (white) emission will be assumed')
            else:
                print('Shape mismatch between PL spectrum and vac_wavelengths. \n' +
                      'Check interpolation. Flat (white) emission will be assumed')
            pl_spectrum = np.ones(self.vac_wavelengths.shape)
        self.pl_spectrum = np.array(pl_spectrum)
        # Calculate normalized PL spectrum (integral over wavelength = 1)
        self.spectral_weights = self.pl_spectrum / integrate(self.pl_spectrum,self.vac_wavelengths)
    def set_doping(self,doping):
        if doping==[]:
            self.doping = [1]*self.d.shape[0]
            print('Doping was not defined. Assuming no doped layers')
        else:
            self.doping = doping
    def set_u(self,u_step=0.002,u_stop=2):
        print('Wavevector mesh changed. Remember to re-run init_pds_variables()')
        self.u_step = u_step
        self.u_stop = u_stop
        self.u = np.hstack((np.arange(0,0.999,self.u_step),
                           np.arange(1.001,self.u_stop,self.u_step)))
    def set_n(self,n_list):
        self.n = np.array(n_list)
        # If a list was passed, reshape for axes: layers,wavelengths
        if self.n.ndim==1:
            self.n.reshape([-1,1])
            if self.vac_wavelengths.shape[0]!=self.n.shape[1]:
                raise ValueError('n and vac_wavelengths have incompatible shape')
        elif self.n.ndim>2:
            raise ValueError('n has wrong dimensionality')
        # Check that index of the emissive layer is real and remove imaginary component
        # this model cannot account for absorption within the emissive layer
        if np.any(np.abs(self.n[self.active_layer,:].imag)>0):
           print("Provided imaginary active layer index. Discarding imaginary part.")
           self.n[self.active_layer,:] = self.n[self.active_layer,:].real
    def load_nk(self,df_nk=None):
        '''
        function to get table of complex refractive indices
        df_nk: pandas dataframe of optical constants
            each column should have key = 'MaterialName_n' or 'MaterialName_k'
            Wavelength column should have key = 'Wavelength (nm)'
        separate material names in doped layers with a hyphen
        doping: list of fraction of each material

        examples:
        doping = [1,1,[0.95,0.05],1]
        layer_names = ['ITO','TCTA','TCTA-TPBi','TPBi']
        '''
        # Make everything lower case to reduce chance for user error
        lower_names = [x.lower() for x in df_nk.columns]
        lower_names[0]='Wavelength (nm)'
        df_nk.columns = lower_names
        layer_names = [x.lower() for x in self.layer_names]
        # each row is a layer, col is wavelength
        n_list = np.zeros((self.d.shape[0],self.vac_wavelengths.shape[0]),
                         dtype=np.complex_)
        assert len(self.layer_names)==len(self.doping), \
            'self.doping and self.layer_names are not same length!'
        if df_nk is None:
            print('must supply df_nk')
        else:
            ref_wavelengths = df_nk['Wavelength (nm)']
            for lay_idx,layer in enumerate(layer_names):
                # If layer is doped, calculate weighted avg n and k
                if type(self.doping[lay_idx]) is list:
                    if np.abs(np.sum(self.doping[lay_idx])-1)>0.001:
                        print('Doping does not add to unity! Normalizing concentrations to 1.')
                        self.doping[lay_idx] = np.array(self.doping[lay_idx])/np.sum(self.doping[lay_idx])
                    n = np.zeros(self.vac_wavelengths.shape)
                    k = np.zeros(self.vac_wavelengths.shape)
                    for d_idx,doped_name in enumerate(layer.split('-')):
                        n = n + np.interp(self.vac_wavelengths,ref_wavelengths,
                                          df_nk[doped_name+'_n'])*self.doping[lay_idx][d_idx]
                        k = k + np.interp(self.vac_wavelengths,ref_wavelengths,
                                          df_nk[doped_name+'_k'])*self.doping[lay_idx][d_idx]
                else:
                    n = np.interp(self.vac_wavelengths,ref_wavelengths,df_nk[layer+'_n'])
                    k = np.interp(self.vac_wavelengths,ref_wavelengths,df_nk[layer+'_k'])
                n_list[lay_idx,:] = n+1j*k
            self.n = n_list
    ############################################################################
    # Dipole emission model functions
    def init_pds_variables(self,custom_u=None):
        '''
        This function initializes variables used for power dissipation spectrum
        calculations
        custom_u is an optional custom-defined wavevector grid to calculate over
            Can be useful if very fine spacing is desired near known peaks.
        '''
        # Check that index of the emissive layer is real and remove imaginary component
        # this model cannot account for absorption within the emissive layer
        if np.any(np.abs(self.n[self.active_layer,:].imag)>0):
           print("Provided imaginary active layer index. Discarding imaginary part.")
           self.n[self.active_layer,:] = self.n[self.active_layer,:].real
        # define in-plane wave vector (unit normalized)
        # Spans from 0 to self.u_stop, excluding 1 because of divergence
        if custom_u is None:
            self.u = np.hstack((np.arange(0,1,self.u_step),
                                np.arange(1+self.u_step,self.u_stop,self.u_step)))
        else:
            self.u = custom_u
        # Calculate angle in the emissive layer for transfer matrix
        # arcsin will only return a complex value if input is complex
        # The arccos method passes Byrnes's "is_forward_angle" test, so using that for now
        # TODO check if this is rigorous
        self.theta_e = np.arccos(csqrt(1-self.u**2))#np.arcsin(self.u.astype('complex'))#
        self.n_e = self.n[self.active_layer,:] # active layer index
        self.n_s = self.n[0,:] # substrate index
        # TODO make whether bottom layer is substrate or air an optional input
        # (for top-emitting or film compatibility)
        # u in substrate, correcting with snell's law
            # Note that multiplication of an n-dimensional array by a 1D array is
            # achieved along a specific axis by reshaping: array_1d.reshape([-1,1,1])
            # where -1 specifies the multiplication axis.
            # along dimensions of [wavelength,wavevector]
        self.u_s = self.u.reshape([1,-1]) * (self.n_e/self.n_s).reshape([-1,1])
        # calculate out-of-plane wavevectors in EML and substrate
        # See equations 4-7 of Neyts 1998 J Opt Soc Am Vol 15 No 4
        self.k_z_e = ( (2 * np.pi / self.vac_wavelengths * self.n_e).reshape([-1,1])
                * csqrt(1-self.u**2).reshape([1,-1]))
        # k_z in substrate, using snell's law above to correct u
        self.k_z_s = ((2 * np.pi / self.vac_wavelengths * self.n_s).reshape([-1,1])
                      *csqrt(1-self.u_s**2))
        # Each array will save a value for a particular
        # dipole position, wavelength, and wavevector
        array_dims = (self.dipole_positions.shape[0],
                      self.vac_wavelengths.shape[0],
                      self.u.shape[0])
        self.K_TM_v = np.zeros(array_dims)
        self.K_TM_h = np.zeros(array_dims)
        self.K_TE_h = np.zeros(array_dims)
        self.K_TM_v_p = np.zeros(array_dims)
        self.K_TM_h_p = np.zeros(array_dims)
        self.K_TE_h_p = np.zeros(array_dims)
        self.K_TM_v_out = np.zeros(array_dims)
        self.K_TM_h_out = np.zeros(array_dims)
        self.K_TE_h_out = np.zeros(array_dims)
        # Absorbed power densities
        self.K_TM_v_abs_up = np.zeros(array_dims)
        self.K_TM_h_abs_up = np.zeros(array_dims)
        self.K_TE_h_abs_up = np.zeros(array_dims)
        self.K_TM_v_abs_down = np.zeros(array_dims)
        self.K_TM_h_abs_down = np.zeros(array_dims)
        self.K_TE_h_abs_down = np.zeros(array_dims)
        # These values depend only on wavelength and u, not dipole position
        array_dims = (self.vac_wavelengths.shape[0],
                      self.u.shape[0])
        self.Rc_TM_down = np.zeros(array_dims)
        self.Rc_TE_down = np.zeros(array_dims)
        self.r_TM_up = np.zeros(array_dims,dtype=complex)
        self.r_TE_up = np.zeros(array_dims,dtype=complex)
        self.t_TM_up = np.zeros(array_dims,dtype=complex)
        self.t_TE_up = np.zeros(array_dims,dtype=complex)
        self.r_TM_down = np.zeros(array_dims,dtype=complex)
        self.t_TM_down = np.zeros(array_dims,dtype=complex)
        self.r_TE_down = np.zeros(array_dims,dtype=complex)
        self.t_TE_down = np.zeros(array_dims,dtype=complex)
        # To minimize number of variables that need to be initialized, could
        # use a structured array:
        # K = np.zeros(array_dims, dtype={'names':('TM_v', 'TM_h', 'TE_h'),
        #                                 'formats':('f8', 'f8', 'f8')}])
    def calc_r_and_t(self,direction='both'):
        '''
        This function loops across wavelength and wavevector to calculate the
        reflection coefficients for one or both halves of the stack.
        This is useful when changing one variable, e.g. HTL or ETL thickness,
        such that all of the coefficients do not have to be re-calculated for
        each thickness
        direction = 'down','up',or 'both'. Defaults to 'both'
        '''
        # Loop through wavelengths
        kwargs = {'unit':'lam','total':len(self.vac_wavelengths),'leave':False,
                  'disable':not self.show_wavelength_progress_bar}
        for w_idx,lam_vac in tqdm(enumerate(self.vac_wavelengths), **kwargs):
            # Check that index of the emissive layer is real and remove imaginary component
            # this model cannot account for absorption within the emissive layer
            n_list = self.n[:,w_idx]
            # Construct layer stacks for "top" and "bottom" halves of the stack
            # bottom_n_list = n_list[self.active_layer::-1]
            # bottom_d_list = self.d[self.active_layer::-1]
            # top_n_list = n_list[self.active_layer:]
            # top_d_list = self.d[self.active_layer:]
            # Loop through in-plane wavevectors
            for u_idx,u in enumerate(self.u):
                TM_tmm_data = partial_tmm(
                    'p',n_list,self.d,self.active_layer,u,lam_vac,direction)
                TE_tmm_data = partial_tmm(
                    's',n_list,self.d,self.active_layer,u,lam_vac,direction)
                # Calculate transfer matrix for bottom half of stack
                if (direction == 'both') or (direction == 'bottom'):
                    self.r_TM_down[w_idx,u_idx] = TM_tmm_data['r_down']
                    self.r_TE_down[w_idx,u_idx] = TE_tmm_data['r_down']
                    self.t_TM_down[w_idx,u_idx] = TM_tmm_data['t_down']
                    self.t_TE_down[w_idx,u_idx] = TE_tmm_data['t_down']
                    # TM_down_data = oledpy.tmm.coherent_tmm(
                    #     'p',bottom_n_list,bottom_d_list,self.theta_e[u_idx],lam_vac)
                    # TE_down_data = oledpy.tmm.coherent_tmm(
                    #     's',bottom_n_list,bottom_d_list,self.theta_e[u_idx],lam_vac)
                    # self.r_TM_down[w_idx,u_idx] = TM_down_data['r']
                    # self.r_TE_down[w_idx,u_idx] = TE_down_data['r']
                    # self.t_TM_down[w_idx,u_idx] = TM_down_data['t']
                    # self.t_TE_down[w_idx,u_idx] = TE_down_data['t']
                # Calculate transfer matrix for top half of stack
                if (direction == 'both') or (direction == 'top'):
                    self.r_TM_up[w_idx,u_idx] = TM_tmm_data['r_up']
                    self.r_TE_up[w_idx,u_idx] = TE_tmm_data['r_up']
                    self.t_TM_up[w_idx,u_idx] = TM_tmm_data['t_up']
                    self.t_TE_up[w_idx,u_idx] = TE_tmm_data['t_up']
                #     TE_up_data = oledpy.tmm.coherent_tmm(
                #         's', top_n_list, top_d_list, self.theta_e[u_idx], lam_vac)
                #     TM_up_data = oledpy.tmm.coherent_tmm(
                #         'p', top_n_list, top_d_list, self.theta_e[u_idx], lam_vac)
                #     self.r_TM_up[w_idx,u_idx] = TM_up_data['r']
                #     self.r_TE_up[w_idx,u_idx] = TE_up_data['r']
                #     self.t_TM_up[w_idx,u_idx] = TM_up_data['t']
                #     self.t_TE_up[w_idx,u_idx] = TE_up_data['t']
            # Calculate overall reflectance and transmittance of stack halves
            if (direction == 'both') or (direction == 'bottom'):
                # Reflectance of optically thin layers (Rc in Furno 2012 A14)
                self.Rc_TM_down = np.abs(self.r_TM_down)**2
                self.Rc_TE_down = np.abs(self.r_TE_down)**2
                # Calculate the energy transmittance of the bottom half of the stack
                # Note that these differ from Furno A12 and A13, due to errors there
                # See also Neyts, eqn. 11-12
                # TODO check this calc with byrnes for consistency
                # self.T_TM_down = (np.abs(self.t_TM_down)**2
                #                   * ((self.n_e/self.n_s)**2).reshape([-1,1])
                #                   * self.k_z_s / np.abs(self.k_z_e))
                # self.T_TE_down = np.abs(self.t_TE_down)**2 * self.k_z_s / np.abs(self.k_z_e)
                # Byrnes equations 21 and 22
                n_s = self.n_s.reshape([-1,1])# Re-shape n_s for array multiplication
                n_e = self.n_e.reshape([-1,1])
                cos_s = csqrt(1-self.u_s**2)
                cos_e = csqrt(1-self.u**2)
                self.T_TM_down = (np.abs(self.t_TM_down)**2
                                  * (n_s * np.conj(cos_s)).real
                                  / (n_e * np.conj(cos_e)).real)
                self.T_TE_down = (np.abs(self.t_TE_down)**2
                                  * (n_s * cos_s).real
                                  / (n_e * cos_e).real)
                # If substrate wavevector is imaginary, set T_down to 0
                # See Neyts, Eqn 12
                # Due to rounding issues, imaginary value is very small when
                # should be real. So use 1e-3 as a good cut-off
                self.T_TM_down[np.abs(self.k_z_s.imag) > 1e-3] = 0
                self.T_TE_down[np.abs(self.k_z_s.imag) > 1e-3] = 0
            if (direction == 'both') or (direction == 'top'):
                # Reflectance of top optically thin layers (Rc in Furno 2012 A14)
                self.Rc_TM_up = np.abs(self.r_TM_up)**2
                self.Rc_TE_up = np.abs(self.r_TE_up)**2
                # Calculate the energy transmittance of the top half of the stack
                # Byrnes equations 21 and 22
                n_f = self.n[-1,:].reshape([-1,1])# Re-shape n_final for array multiplication
                n_e = self.n_e.reshape([-1,1])
                cos_f = csqrt(1-self.u_s**2)
                cos_e = csqrt(1-self.u**2)
                self.T_TM_up = (np.abs(self.t_TM_up)**2
                                  * (n_f * np.conj(cos_f)).real
                                  / (n_e * np.conj(cos_e)).real)
                self.T_TE_up = (np.abs(self.t_TE_up)**2
                                  * (n_f * cos_f).real
                                  / (n_e * cos_e).real)
    def calc_K(self,store_a=False):
        '''
        Loop over dipole positions
        '''
        # Reshape u for array multiplication
        u = self.u.reshape([1,-1])
        # Loop through dipole positions
        if store_a:
            array_dims = (self.dipole_positions.shape[0],
                          self.vac_wavelengths.shape[0],
                          self.u.shape[0])
            self.a_TM_up = np.zeros(array_dims,dtype=complex)
            self.a_TM_down = np.zeros(array_dims,dtype=complex)
            self.a_TE_up = np.zeros(array_dims,dtype=complex)
            self.a_TE_down = np.zeros(array_dims,dtype=complex)
        for z_idx,z_down in enumerate(self.dipole_positions * self.d[self.active_layer]):
            # Distance of dipole from top EML interface
            z_up = self.d[self.active_layer] - z_down
            # Calculate phase-adjusted reflection coefficients (Furno 2012 A4 and A5)
            a_TM_up = self.r_TM_up * np.exp(2 * 1j * self.k_z_e * z_up)
            a_TM_down = self.r_TM_down * np.exp(2 * 1j * self.k_z_e * z_down)
            a_TE_up = self.r_TE_up * np.exp(2 * 1j * self.k_z_e * z_up)
            a_TE_down = self.r_TE_down * np.exp(2 * 1j * self.k_z_e * z_down)
            if store_a:
                self.a_TM_up[z_idx] = a_TM_up
                self.a_TM_down[z_idx] = a_TM_down
                self.a_TE_up[z_idx] = a_TE_up
                self.a_TE_down[z_idx] = a_TE_down
            # Calculate dipole emission densities (Furno 2012 A1-A3)
            self.K_TM_v[z_idx,:,:] = 3/4 * np.real((u**2/csqrt(1-u**2)) * (1+a_TM_up)*(1+a_TM_down)
                                       / (1-a_TM_up*a_TM_down))
            self.K_TM_h[z_idx,:,:] = 3/8 * np.real(csqrt(1-u**2) * (1-a_TM_up)*(1-a_TM_down)
                                       / (1-a_TM_up*a_TM_down))
            self.K_TE_h[z_idx,:,:] = 3/8 * np.real((1/csqrt(1-u**2)) * (1+a_TE_up)*(1+a_TE_down)
                                       / (1-a_TE_up*a_TE_down))
            # TODO calculate R_TM and R_TE so absorption loss can be quantified
            # Calculate dipole emission coupled to the substrate
            # See Jpn. J. Appl. Phys. 57, 012101 (2018), Eqn. 10 - 15
            # note that Eqn. A8 - A11 in Furno 2012 have flipped signs
            # note that Kim 2018 has incorrect K_TM_h_p with a |1+a_TM_up|**2 on top
            self.K_TM_v_p[z_idx,:,:] = (3/8 * (u**2/csqrt(1-u**2)) * np.abs(1+a_TM_up)**2
                                       / np.abs(1-a_TM_up*a_TM_down)**2 * self.T_TM_down)
            self.K_TM_h_p[z_idx,:,:] = (3/16 * csqrt(1-u**2) * np.abs(1-a_TM_up)**2
                                       / np.abs(1-a_TM_up*a_TM_down)**2 * self.T_TM_down)
            self.K_TE_h_p[z_idx,:,:] = (3/16 * (1/csqrt(1-u**2)) * np.abs(1+a_TE_up)**2
                                       / np.abs(1-a_TE_up*a_TE_down)**2 * self.T_TE_down)
            # Absorbed power densities
            self.K_TM_v_abs_up[z_idx,:,:] = (
                3/8 * (u**2/csqrt(1-u**2)) * np.abs(1+a_TM_down)**2
                / np.abs(1-a_TM_up*a_TM_down)**2
                * (1 - self.Rc_TM_up - self.T_TM_up))
            self.K_TM_h_abs_up[z_idx,:,:] = (
                3/16 * csqrt(1-u**2) * np.abs(1-a_TM_down)**2
                / np.abs(1-a_TM_up*a_TM_down)**2
                * (1 - self.Rc_TM_up - self.T_TM_up))
            self.K_TE_h_abs_up[z_idx,:,:] = (
                3/16 * (1/csqrt(1-u**2)) * np.abs(1+a_TE_down)**2
                / np.abs(1-a_TE_up*a_TE_down)**2
                * (1 - self.Rc_TE_up - self.T_TE_up))
            self.K_TM_v_abs_down[z_idx,:,:] = (
                3/8 * (u**2/csqrt(1-u**2)) * np.abs(1+a_TM_up)**2
                / np.abs(1-a_TM_up*a_TM_down)**2
                * (1 - self.Rc_TM_down - self.T_TM_down))
            self.K_TM_h_abs_down[z_idx,:,:] = (
                3/16 * csqrt(1-u**2) * np.abs(1-a_TM_up)**2
                / np.abs(1-a_TM_up*a_TM_down)**2
                * (1 - self.Rc_TM_down - self.T_TM_down))
            self.K_TE_h_abs_down[z_idx,:,:] = (
                3/16 * (1/csqrt(1-u**2)) * np.abs(1+a_TE_up)**2
                / np.abs(1-a_TE_up*a_TE_down)**2
                * (1 - self.Rc_TE_down - self.T_TE_down))
        # Note that the following correction will be a factor of 1 if n_s = n_medium
        # Transmission and reflection out of the substrate to air
        # This calculation assumes a lossless interface
        n_s = self.n_s.reshape([-1,1])# Re-shape n_s for array multiplication
        cos_s = csqrt(1-self.u_s**2)
        cos_medium = csqrt(1-(n_s/self.n_medium*self.u_s)**2)
        self.Rso_TE = np.abs((n_s*cos_s-self.n_medium*cos_medium)
                    /(n_s*cos_s+self.n_medium*cos_medium))**2
        self.Tso_TE = 1-self.Rso_TE
        self.Rso_TM = np.abs((n_s*cos_medium-self.n_medium*cos_s)
                    /(n_s*cos_medium+self.n_medium*cos_s))**2
        self.Tso_TM = 1 - self.Rso_TM
        # Fraction of power radiated to far-field medium (Furno 2012 A14)
        # Re-shaping is not needed here, because numpy will broadcast correctly
        # when the last two dimensions are the same
        self.K_TM_v_out = self.K_TM_v_p*(self.Tso_TM/(1-self.Rso_TM*self.Rc_TM_down))
        self.K_TM_h_out = self.K_TM_h_p*(self.Tso_TM/(1-self.Rso_TM*self.Rc_TM_down))
        self.K_TE_h_out = self.K_TE_h_p*(self.Tso_TE/(1-self.Rso_TE*self.Rc_TE_down))
        # Based on given anisotropy factor, calculate net dipole emission
        # Modified from the isotropic case in Furno A6
        K_total = (self.anisotropy_factor * self.K_TM_v
                    + (1 - self.anisotropy_factor) * (self.K_TM_h + self.K_TE_h))
        #K_prime = (self.anisotropy_factor * K_TM_v_p
        #            + (1 - self.anisotropy_factor) * (K_TM_h_p + K_TE_h_p))
        K_out = (self.anisotropy_factor * self.K_TM_v_out
                 + (1 - self.anisotropy_factor) * (self.K_TM_h_out + self.K_TE_h_out))
        K_abs = (self.anisotropy_factor * (self.K_TM_v_abs_up + self.K_TM_v_abs_down)
                 + (1 - self.anisotropy_factor)
                    * (self.K_TM_h_abs_up + self.K_TM_h_abs_down
                       + self.K_TE_h_abs_up + self.K_TE_h_abs_down))
        return {'u':self.u,'K_total':K_total,'K_out':K_out,'K_abs':K_abs}
    def loop_over_single_layer_prop(self,layer_idx,layer_prop_list,layer_prop='d'):
        '''
        This function allows user to calculate K_total and K_out while varying the
        properties of a single layer, e.g. the ETL thickness or index of refraction
        To vary layer optical constants, set layer_prop='n'
        To vary layer thickness, set layer_prop='d'
        TODO This function does not currently check the inputs for proper shape, type, etc.
        '''
        self.run_attribute_checks()
        print('Varied layer is: ' + self.layer_names[layer_idx])
        # Initialize variables
        array_dims = (len(layer_prop_list),self.dipole_positions.shape[0],
                      self.vac_wavelengths.shape[0],self.u.shape[0])
        K_total_loop = np.zeros(array_dims)
        K_out_loop = np.zeros(array_dims)
        # Check what needs to be re-calculated each loop
        calculate_once=False
        if layer_idx<self.active_layer:
            direction='bottom'
        if layer_idx>self.active_layer:
            direction='top'
        # If varying the active layer thickness r and t only need to be calculated once
        if layer_idx==self.active_layer and layer_prop=='d':
            calculate_once=True
        # If varying active layer index, everything needs to be recalculated every time
        if layer_idx==self.active_layer and layer_prop=='n':
            direction='both'
        kwargs = {'unit':layer_prop,'total':len(layer_prop_list),
                  'disable':not self.show_progress_bar}
        for prop_idx,prop_iter in tqdm(enumerate(layer_prop_list), **kwargs):
            if layer_prop=='d':
                self.d[layer_idx]=prop_iter
            elif layer_prop=='n':
                self.n[layer_idx]=prop_iter
            if layer_idx==self.active_layer:
                self.n_e = self.n[self.active_layer,:] # active layer index
                self.k_z_e = ( (2 * np.pi / self.vac_wavelengths * self.n_e).reshape([-1,1])
                        * csqrt(1-self.u**2).reshape([1,-1]))
            # On this first iteration, calculate r and t
            # for both top and bottom halves of stack
            if prop_idx==0:
                self.calc_r_and_t(direction='both')
            elif prop_idx>0 and not calculate_once:
                self.calc_r_and_t(direction)
            dissipation_data = self.calc_K()
            K_total_loop[prop_idx,:,:,:] = dissipation_data['K_total']
            K_out_loop[prop_idx,:,:,:] = dissipation_data['K_out']
        return {'K_total_loop':K_total_loop,'K_out_loop':K_out_loop}
    def calc_outcoupled_fraction(self,K_total,K_out,n_list):
        '''
        Calculates the fraction of light which escapes to air from the bottom
        half of the stack, based on the critical angle from Snell's law
        '''
        # Critical angle for total-internal reflection via Snell's law
        u_critical = self.n_medium / n_list[self.active_layer]
        int_range = self.u<u_critical # integrate up to critical angle
        out=np.trapz(2*self.u[int_range]*K_out[int_range],self.u[int_range]).real
        total=np.trapz(2*self.u*K_total,self.u)
        return (out/total)
    def calc_outcoupled_fraction_array(self,K_total_array,K_out_array):
        '''
        Wrapper for calc_outcoupled_fraction to loop over dipole position and
        wavelength
        '''
        outcoupling_array = np.zeros((self.dipole_positions.shape[0],
                                           self.vac_wavelengths.shape[0]))
        # There's probably a fancy way to vectorize this, but probably not worth the time
        for z_idx,z in enumerate(self.dipole_positions):
            for lam_idx,lam_vac in enumerate(self.vac_wavelengths):
                outcoupling_array[z_idx,lam_idx] = self.calc_outcoupled_fraction(
                    K_total_array[z_idx,lam_idx,:],K_out_array[z_idx,lam_idx,:],
                    self.n[:,lam_idx])
        return outcoupling_array
    def summarize_device(self):
        self.run_attribute_checks()
        self.calc_r_and_t(direction='both')
        dissipation_data = self.calc_K()
        self.K_total_array=dissipation_data['K_total']
        self.K_out_array=dissipation_data['K_out']
        self.outcoupling_array = self.calc_outcoupled_fraction_array(self.K_total_array,self.K_out_array)
        self.spatially_average_outcoupling()
        self.purcell_factor()
        #self.calc_efficiency_metrics()
    def summarize_loop_data(self,loop_data,layer_idx,layer_prop_list,layer_prop='d'):
        outcoupling_array=['']*len(layer_prop_list)
        eta_out = np.zeros(len(layer_prop_list))
        PLQYeff = np.zeros(len(layer_prop_list))
        U = np.zeros(len(layer_prop_list))
        F = np.zeros(len(layer_prop_list))
        U_over_F = np.zeros(len(layer_prop_list))
        for loop_idx,layer_prop_i in enumerate(layer_prop_list):
            K_total_array = loop_data['K_total_loop'][loop_idx]
            K_out_array = loop_data['K_out_loop'][loop_idx]
            outcoupling_array[loop_idx]= self.calc_outcoupled_fraction_array(
                K_total_array,K_out_array)
            self.outcoupling_array = outcoupling_array[loop_idx]
            lam_avg_eta_out,space_avg_eta_out = self.spatially_average_outcoupling(return_values=True)
            eta_out[loop_idx] = space_avg_eta_out
            self.purcell_factor(K_total_array=K_total_array,K_out_array=K_out_array)
            PLQYeff[loop_idx] = self.PLQY_effective_avg
            U[loop_idx],F[loop_idx]=self.U_avg,self.F_avg
            U_over_F[loop_idx] = np.trapz(self.spectral_weights*U[loop_idx]/F[loop_idx],
                                       self.vac_wavelengths)
        return {'U':U,'F':F,'U_over_F':U_over_F,'eta_out':eta_out,'PLQYeff':PLQYeff,
                'outcoupling_array':outcoupling_array}
    def spatially_average_outcoupling(self,outcoupling_array=None,return_values=False):
        self.run_attribute_checks()
        if outcoupling_array is None:
            outcoupling_array=self.outcoupling_array
        # Integrate along wavelength axis, weighted by PL specrtrum
        self.wavelength_averaged_eta_out = integrate(
            outcoupling_array*self.spectral_weights,self.vac_wavelengths,axis=1,squeeze_axis=1)
        # Integrate spatially, weighted by RZ
        self.spatially_averaged_eta_out = integrate(self.wavelength_averaged_eta_out*self.RZ_weights,
                                                self.dipole_positions)
        if return_values:
            return self.wavelength_averaged_eta_out,self.spatially_averaged_eta_out
    def angular_el_spectra(self,k_r=None,k_nr=None,tau=None,PLQY=None):
        self.run_attribute_checks()
        # From Furno 2012, Equation A19
        # Note the Erratum: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.88.119905
        # Calculate Purcell Factor and effective quantum efficiency
        self.purcell_factor(k_r=k_r,k_nr=k_nr,tau=tau,PLQY=PLQY)
        # The above function also calculates the spatially averaged K_out, weighting by RZ
        # Interpolate K_out to desired calculation angles
        K_out_interp = np.zeros((self.vac_wavelengths.shape[0],self.thetas.shape[0]))
        for l_idx,lam_vac in enumerate(self.vac_wavelengths):
            u_interp = (np.sin(self.thetas*np.pi/180)*self.n_medium/self.n_e[l_idx]).real
            K_out_interp[l_idx,:] = np.interp(u_interp,self.u,self.K_out_spatial_avg[l_idx,:])
        # Furno 2012, Eqn. A18
        self.P_out = ((self.n_medium / self.n_e.reshape([-1,1]))**2
                 * (np.cos(self.thetas*np.pi/180).reshape([1,-1]) / np.pi)
                 * K_out_interp)
        # Furno 2012, A19
        # Need to add charge balance, hc, and J/e if want absolute radiant intensity
        self.I_EL = (1/(self.vac_wavelengths)*self.spectral_weights*self.PLQY_effective
               /self.F).reshape([-1,1]) * self.P_out
    def calc_angular_emission_profile(self):
        intensity = integrate(self.I_EL,self.vac_wavelengths,axis=0,squeeze_axis=0)
        # Furno 2012, Eqn. A20
        radiance = integrate(self.I_EL/np.cos(self.thetas*np.pi/180),
                            self.vac_wavelengths,axis=0,squeeze_axis=0)
        # Furno 2012, Eqn. A21
        K_m = 683.002 #lm/W, constant converting optical power to lumens
        V = oledpy.import_photopic_response()
        # Interpolate the photopic response to same wavelengths and reshape
        # so that it can be multiplied by self.I_EL
        V_interp = np.interp(self.vac_wavelengths,V[:,0],V[:,1]).reshape([-1,1])
        luminance = K_m * integrate(V_interp*self.I_EL/np.cos(self.thetas*np.pi/180),
                                   self.vac_wavelengths,axis=0,squeeze_axis=0)
        return intensity,radiance,luminance
    def calc_efficiency_metrics(self,Va,gamma=1):
        '''Function to calculate EQE, power efficiency
        Units are not fully worked out on this
        '''
        ########################################################################
        # Calculate wavelength dependent, spatially averaged outcoupling efficiency
        self.eta_out_of_lambda = integrate(
            self.outcoupling_array*self.RZ_weights.reshape([-1,1]),
            self.dipole_positions,axis=0,squeeze_axis=0)
        # Furno 2012, Eqn. 30.
        # Need to check this with eta_out calculated with other methods. Should be same.
        self.eta_out = integrate(self.spectral_weights*self.U/self.F,self.vac_wavelengths)
        ########################################################################
        # Calculate power efficiency (lm/W)
        K_m = 683.002 #lm/W, constant converting optical power to lumens
        V = oledpy.import_photopic_response() # The photopic response of the eye
        # Interpolate the photopic response to same wavelengths and reshape
        # so that it can be multiplied by self.I_EL
        V_interp = np.interp(self.vac_wavelengths,V[:,0],V[:,1]).reshape([-1,1])
        # Furno 2012, Eqn. A16
        power_efficiency = gamma / (e*Va) * h * c * integrate(
            V_interp * (self.spectral_weights / self.vac_wavelengths) * self.PLQY_effective
            * self.eta_out_of_lambda, self.vac_wavelengths
        )
        ########################################################################
        # Calculate external quantum efficiency
        self.eqe = gamma * integrate(self.spectral_weights * self.PLQY_effective
                               * self.eta_out_of_lambda,self.vac_wavelengths)
    def analyze_modes(self,K_total,K_out,K_abs=None):
        self.run_attribute_checks()
        # Calculate spatial averages of recombination zone
        K_total_spatial_avg = integrate(K_total*self.RZ_weights.reshape([-1,1,1]),
                                     self.dipole_positions,axis=0,squeeze_axis=0)
        K_out_spatial_avg = integrate(K_out*self.RZ_weights.reshape([-1,1,1]),
                                     self.dipole_positions,axis=0,squeeze_axis=0)
        #K_abs_spatial_avg = integrate(K_abs*self.RZ_weights.reshape([-1,1,1]),
        #                             self.dipole_positions,axis=0,squeeze_axis=0)
        modes=pd.DataFrame(np.nan, index=np.arange(len(self.vac_wavelengths)), 
                       columns=['air','sub','wg','spp'])
        int_range={}
        for lam_idx,lam_vac in enumerate(self.vac_wavelengths):
            # Get index ranges for air, substrate, waveguide, and surface-plasmon-polariton modes
            # See Salehi 2017 or Nowy 2010
            int_range['air'] = self.u < (self.n_medium / self.n_e)[lam_idx]
            int_range['sub'] = np.logical_and(self.u>(self.n_medium / self.n_e)[lam_idx],
                                       self.u<(self.n_s / self.n_e)[lam_idx])
            int_range['wg'] =  np.logical_and(self.u>(self.n_s / self.n_e)[lam_idx],
                                       self.u<1)
            int_range['spp'] = self.u > 1
            for mode, range in int_range.items():
                modes[mode][lam_idx] = integrate(
                    2*self.u[range]*K_total_spatial_avg[lam_idx,range],
                    self.u[range]).real
            # absorption calculation not yet implemented
            #modes['abs'][lam_idx] = integrate(
            #    2*self.u[int_range['air']]*K_abs_spatial_avg[lam_idx,int_range['air']],
            #    self.u[int_range['air']]).real
        spectrally_integrated={}
        for name in modes.columns:
            spectrally_integrated[name] = integrate(modes[name]*self.spectral_weights,self.vac_wavelengths)
        return spectrally_integrated
    def analyze_modes_loop(self,loop_variable_list,loop_data):
        spectrally_integrated_modes=pd.DataFrame(
            np.nan, index=np.arange(len(loop_variable_list)),
            columns=['air','sub','wg','spp'])
        for idx in range(0,len(loop_variable_list)):
            spectrally_integrated_modes.iloc[idx] = self.analyze_modes(
                loop_data['K_total_loop'][idx],loop_data['K_out_loop'][idx])
        # Normalize to 1
        spectrally_integrated_modes=spectrally_integrated_modes.div(
            spectrally_integrated_modes.sum(axis=1), axis=0)
        return spectrally_integrated_modes
    def purcell_factor(self,K_total_array=None,K_out_array=None,k_r=None,k_nr=None,tau=None,PLQY=None):
        if K_total_array is None:
            K_total_array = self.K_total_array
        if K_out_array is None:
            K_out_array = self.K_out_array
        # Check if k_r and k_nr or tau and PLQY have been passed to the function
        if not (k_r is None and k_nr is None):
            self.k_r,self.k_nr = k_r,k_nr
        if not (tau is None and PLQY is None):
            self.tau,self.PLQY=tau,PLQY
        # If k_r and k_nr have not been defined, then calculate from PLQY and tau
        if (self.k_r is None and self.k_nr is None) and not (self.PLQY is None and self.tau is None):
            self.k_r = self.PLQY / self.tau
            self.k_nr = 1/self.tau - self.k_r
        # If PLQY and tau have not been defined, calculate them from k_r and k_nr
        elif not (self.k_r is None and self.k_nr is None) and (self.PLQY is None and self.tau is None):
            self.tau = 1 / (k_r+k_nr) # intrinsic emitter exciton lifetime
            self.PLQY = self.k_r * self.tau # intrinsic emitter photoluminescence quantum yield
        elif not (self.k_r is None and self.k_nr is None) and not (self.PLQY is None and self.tau is None):
            pass
        else:
            raise ValueError('Assign values of k_r and k_nr or tau and PLQY to the architecture object')
        # The Purcell factor, F(lambda), is the total radiated power
        # Furno 2012, Eqn. 16
        self.K_total_spatial_avg = integrate(K_total_array*self.RZ_weights.reshape([-1,1,1]),
                                     self.dipole_positions,axis=0,squeeze_axis=0)
        self.F = integrate(2*self.u*self.K_total_spatial_avg,self.u).real
        # Calculate spatially varying F, for different dipole positions
        self.F_spatial = integrate(2*self.u*K_total_array,self.u).real

        # Furno 2012, Eqn. 22
        # calculated outcoupled fraction
        self.K_out_spatial_avg = integrate(K_out_array*self.RZ_weights.reshape([-1,1,1]),
                                     self.dipole_positions,axis=0,squeeze_axis=0)
        # Critical angle for total-internal reflection via Snell's law
        u_critical = self.n_medium / self.n_e
        self.U=np.zeros(self.vac_wavelengths.shape)
        self.U_spatial=np.zeros((len(self.dipole_positions),len(self.vac_wavelengths)))
        for lam_idx,lam_vac in enumerate(self.vac_wavelengths):
            int_range = self.u<u_critical[lam_idx] # integrate up to critical angle
            self.U[lam_idx] = integrate(
                2*self.u[int_range]*self.K_out_spatial_avg[lam_idx,int_range],
                self.u[int_range]).real
            self.U_spatial[:,lam_idx] = integrate(
                2*self.u[int_range]*K_out_array[:,lam_idx,int_range],
                self.u[int_range]).real
        self.tau_effective = 1 / (self.k_nr + self.F*self.k_r) # Furno 2012 Eqn. 18
        self.PLQY_effective = self.PLQY * self.F / (1-self.PLQY+self.F*self.PLQY) # Furno 2012 Eqn. 20
        # Furno 2012, Eqn. 28 and 29
        self.F_avg = integrate(self.spectral_weights * self.F,self.vac_wavelengths)
        self.U_avg = integrate(self.spectral_weights * self.U,self.vac_wavelengths)
        # Spatial variation for different dipole positions:
        self.F_lamavg_spatial = integrate(self.spectral_weights * self.F_spatial,self.vac_wavelengths)
        self.U_lamavg_spatial = integrate(self.spectral_weights * self.U_spatial,self.vac_wavelengths)
        self.tau_effective_spatial = 1 / (self.k_nr + self.F_lamavg_spatial*self.k_r)
        self.PLQY_effective_spatial = (self.PLQY * self.F_lamavg_spatial 
                                    / (1-self.PLQY+self.F_lamavg_spatial*self.PLQY))
        # Overall average PLQY and tau
        self.tau_effective_avg = integrate(self.spectral_weights * self.tau_effective,
                                           self.vac_wavelengths)
        self.PLQY_effective_avg = integrate(self.spectral_weights * self.PLQY_effective,
                                           self.vac_wavelengths)
        return #{'PLQY_effective_avg':self.PLQY_effective_avg,'tau_effective_avg':self.tau_effective_avg}
