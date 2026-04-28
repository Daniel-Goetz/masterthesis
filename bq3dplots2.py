import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import matplotlib as mpl
from matplotlib import gridspec
import json

# astropy
import astropy
from astropy.io import fits
from astropy import constants
from astropy import units as u

# q3dfit
from q3dfit.q3dout import load_q3dout
from q3dfit import q3dutil

# iterative plot
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

# bruno
from bq3dfit import bq3dutils

# constants
c_kms = constants.c.to('km/s')

# --------------------------- Functions -------------------
def gaussian(x, *p):
    y = x*0
    for i in range(len(p)//3):
        y = y + p[0 + 3*i] * np.exp(-0.5*((x-p[1+3*i])/p[2+3*i])**2)
    return y

# --------------------------- Auxiliary plot funcitons-----
def get_norm(norm):
    '''
    Input
    -----
    norm: bool or str
        - if bool: True for LogNorm and False for Linear
        - if str: True for LogNorm and False for Linear
    '''
    
    if isinstance(norm, bool):
        if norm:
            norm = mpl.colors.LogNorm()
        else:
            norm = None
    elif isinstance(norm, str):
        if norm.lower() in ['log', 'lognorm']:
            norm = mpl.colors.LogNorm()
        elif norm.lower() in ['asinhnorm', 'asinh']:
            norm = mpl.colors.AsinhNorm()
        else:
            norm = None

    return norm

def set_vlims(data, vmin=None, vmax=None, vpercent=None):

    value = quantity_to_value

    # Default: minimum/maximum
    if vmin is None:
        vmin = np.nanmin(value(data))
    if vmax is None:
        vmax = np.nanmax(value(data))

    # Percent
    if vpercent is not None:
        if not is_iterable(vpercent):
            vpercent = [vpercent, vpercent]
        else:
            assert len(vpercent) == 2, "list <vpercent> size != 2."

        data2 = data[(data >= value(vmin)) & (data <= value(vmax))]
        if vpercent[0] is not None:
            vmin = percentile(data2, 50 - abs(50 - vpercent[0]))
        if vpercent[1] is not None:
            vmax = percentile(data2, abs(50 - vpercent[1]) + 50)

    return value(vmin), value(vmax)

def percentile(data, percent, mask=None):

    good = good_values(data, mask)

    if isinstance(data, u.quantity.Quantity):
        unit = data.unit
    else:
        unit = 1

    return np.percentile(good, percent) * unit

def good_values(data, mask=None):

    unmasked_values = quantity_to_value(data)

    if mask is None:
        if isinstance(data, np.ma.core.MaskedArray):
            mask = data.mask
        else:
            try:
                # For spectral_cube types
                mask = data.mask.view()
            except:
                mask = False

    mask_nan = np.isnan(unmasked_values)
    mask = mask + mask_nan

    return unmasked_values[~mask]

def quantity_to_value(x):
    '''
    Quantity -> value
    '''

    if isinstance(x, u.quantity.Quantity):
        return x.to_value()
    elif np.ma.isMaskedArray(x):
        if isinstance(x.data, u.Quantity):
            return np.ma.array(x.data.value, mask=x.mask)
        else:
            return np.ma.array(x.data, mask=x.mask)
    else:
        return x

def is_iterable(obj, check_not_str=True, check_not_header=True):
    '''
    check_str: Also check if it is not str?
               Remender that str is iterable by itself.
    '''
    #hasattr(obj, '__iter__'): FIXME: test this option

    try:
        null = iter(obj)
        logic = True
    except TypeError:
        logic = False

    # Check if it is not str (which is also iterable)
    if check_not_str:
        logic = logic and not isinstance(obj, str)
    
    if check_not_header:
        logic = logic and not isinstance(obj, astropy.io.fits.Header)

    return logic

# functions for linspace    
def f_log(x):
    return np.log10(x)
def f_log_inv(x):
    return 10**x
    
def f_asinh(x, a0=1):
    return a0 * np.arcsinh(x / a0)
def f_asinh_inv(x, a0=1):
    return a0 * np.sinh(x / a0)

def evenly_spaced_array(xi, xf, N, scale='linear', a0=1):
    '''
    Input
    -----
    xi: float
        Start of array
    xf: float
        End of array
    N: int  
        Number of points
    scale: string matplotlib.colors or  
        - 'linear', or matplotlib.colors.NoneNorm()
        - 'log'
        - 'asinh'
    a0: float
        Linear_width of the asinh scale 

    Output
    ------
    equal
    '''

    if isinstance(scale, str):
        scale = scale.lower()
    scale_norm = get_scale(scale)
    
    # Linear
    if isinstance(scale_norm, mpl.colors.NoNorm):
        equal_spaced_array = np.linspace(xi, xf, N)

    # Log
    elif isinstance(scale_norm, mpl.colors.LogNorm):
        logspace = np.linspace(f_log(xi), f_log(xf), N)
        equal_spaced_array = f_log_inv(logspace)

    # Asinh
    elif isinstance(scale_norm, mpl.colors.AsinhNorm):
        asinhspace = np.linspace(f_asinh(xi, a0), f_asinh(xf, a0), N)
        equal_spaced_array = f_asinh_inv(asinhspace, a0)

    return equal_spaced_array

def get_scale(scale=None, vmin=None, vmax=None, a0=1, opts={}):
    '''
    Create matplotlib.colors object with a given scale.
    
    Input
    -----
    scale: string or matplotlic.colors norm class, like matplotlic.colors.NoNorm
        None is returns a linear matplotlib.colors.NoNorm object.
        Example of accepted strings: log, linear, asinh.
    vmin: float
        Minimum value of the scale
    vmax: float
        Maximum value of the scale
    a0: float
    
    opts: dict
        Additional arguments to be passed to matplotlib.colors normalization
        classes.
        <opts> items overrides <vmin>, <vmax> and <a0>. 
    
    Output
    ------
     
    '''

    # if string, convert to lowercase
    if isinstance(scale, str):
        scale =  scale.lower()

    opts1 = dict(vmin=vmin, vmax=vmax)
    opts1.update(opts)

    # linear
    if (scale in [None, 'linear']) or isinstance(scale, mpl.colors.NoNorm):
        scale_colors = mpl.colors.NoNorm(**opts)
    # log
    elif (scale in ['log']) or isinstance(scale, mpl.colors.LogNorm):
        scale_colors = mpl.colors.LogNorm(**opts)
    # asinh
    elif (scale in ['asinh']) or isinstance(scale, mpl.colors.AsinhNorm):
        # avoid repeated 
        if 'linear_width' in opts1:
            a0 = opts1.pop('linear_width')
        if 'a0' in opts1:
            a0 = opts1.pop('a0')
            
        scale_colors = mpl.colors.AsinhNorm(linear_width=a0, **opts)
    else:
        # try to find by name
        if isinstance(scale, str):
            try:
                attr = mpl.colors.__dict__.keys()
                idx = list(map(str.lower, attr)).index(scale)
                scale_class = getattr(mpl.colors, list(attr)[idx])
                return scale_class(**opts)
            except:
                raise Exception(f"scale <{scale}> not valid.")
        # if class were already initialized
        else:
            try:
                # class must be initiallized (e.g. mpl.colors.NoNorm()) 
                #scale_class(vmin=None, vmax=None, **opts)
                for key, value in opts.items():
                    setattr(scale_class, key, value)
            except:
                raise Exception(f"scale <{scale}> not valid.")
                
    return scale_colors

# ------------------- plot functions -----------------------
def plot_single_fit(filename_q3di, col, row, wvel_line=None, rest=True,
                    axis=None, fig=None,  clean_axis=False, return_pars=False,
                    redshift=None, plot_guess=True, plot_model=True, plot_cont=True,
                    #plot_std=False,
                    plot_error=True, xlim=None, ylim=None, plottype_data='step', 
                    show=True, **args):
    '''
    Input
    -----
    filename_q3di: str
        Filename of the q3di configuration file
    col, row: int
        Column, Row (first index=1) of the data cube to be fitted.
    rest: bool
        Set spectral axis at the rest?
    wvel_line: str
        If not None, convert the spectral axis to velocities based on the 
        rest wavelenth of a given emission line. 
        - Example: If wvel_line='Hbeta' (and rest=True), the resulting spectral
          axis corresponding to the relative radia velocity of that the Hbeta
          would have at a given spectral position. Only works if Hbeta was one
          of the fitted components.
    fig: matplotlib.figure.Figure
        Figure object where the axis will be created
    axis: matplotlib.axes._axes.Axes
        Axis object where the plot will be drawn
    plottype_data: matplotlib.pyplot function name
        Plot type for the spectral data
        Example: "plot" or "step"
   
    '''

    # ====================================
    # Load
    q3di = q3dutil.get_q3dio(filename_q3di)
    q3do = load_q3dout(q3di, col, row)

    linelist = q3dutil.get_linelist(q3di)
    # only saved if checkcomp was used
    q3do.sepfitpars()

    # ====================================
    # redshift
    if redshift is None:
        redshift = q3di.zsys_gas

    # Spectral
    xlabel = "Wavelength ($\mu m$)"

    spectral = q3do.wave * 1
    if rest:
        spectral = spectral / (1 + redshift) 
        xlabel = "Rest wavelength ($\mu m$)"

    if wvel_line is not None:
        assert isinstance(wvel_line, str)
        try:
            rest_line = linelist[linelist['name'] == wvel_line]['lines']
            spectral = (spectral / rest_line - 1) * c_kms
        except:
            raise Exception(f"wvel_line not fitted: {wvel_line}")

        xlabel = f"Velocity ({wvel_line}) (km/s)"

    # flux densities, continuum index
    spec = q3do.spec
    spec_err = q3do.spec_err

    # Continuum
    if q3di.docontfit:
        cont_fit = q3do.cont_fit
        ct_indx = q3do.ct_indx

    else:
        cont_fit = q3do.spec * 0
        ct_indx = (q3do.spec * 0).astype(bool)

    # Model
    if plot_model:

        line_fit = q3do.line_fit
        model = cont_fit + line_fit

        # model components
        p = np.array([item for key, item in q3do.param.items()\
                      if 'SPECRES' not in key.upper()])
        specres = np.array([item for key, item in q3do.param.items()\
                            if 'SPECRES' in key.upper()])
        p[2::3] = p[2::3] / c_kms.value * p[1::3]# * linelist['lines']
        pp = p.reshape(len(p)//3, 3)

        # without convolving
        model_comps0 = np.array([gaussian(q3do.wave, *pi) for pi in pp])
        # convolving
        model_comps = np.array([ispecres.spect_convolver(q3do.wave, imodel0, pi[1])\
                       for pi, imodel0, ispecres in zip(pp, model_comps0, specres)])

    # FIXME: use q3dfit way of generating the gaussian profiles
    # if 0:# TODO from fitspec.py        
    #     q3do.line_fit = emlmod.eval(lmout.params, x=gdlambda)

    # Initial guess - second fit (result first fit)
    # FIXME also convolve the guess models
    if plot_guess:
        p1 = np.array([item.value for key, item in q3do.parinit.items()])
        p1[2::3] = p1[2::3] / c_kms.value  * p1[1::3]#linelist['lines']
        line_init1 = gaussian(q3do.wave, *p1)

    # ====================================
    # plots
    if fig is None:
        fig = plt.figure()
    if axis is None:

        gs = gridspec.GridSpec(5, 1)
        ax1 = fig.add_subplot(gs[0:4])
        ax2 = fig.add_subplot(gs[4], sharex=ax1)

    else:
        ax1, ax2 = axis

    if clean_axis:
        ax1.cla()
        ax2.cla()
    ax = axis 

    # data
    ax1.__getattribute__(plottype_data)(spectral, spec, label="data", **args)
    #ax1.axhline(0, color='gray', ls='-', lw=.5, alpha=.5)
    ax1.axhline(0, color='black', ls='-', lw=.9, alpha=.5, zorder=100)

    # full model (result from from first fit iteration)
    if plot_guess:
        ax1.plot(spectral, cont_fit + line_init1, label="guess1")


    # final full model
    if plot_model:
        ax1.plot(spectral, model, label="model", color='C2')

        # individual components
        for imodel, imodel0 in zip(model_comps, model_comps0):
            ax1.plot(spectral, cont_fit + imodel, ls="-", color='k', lw=.7, alpha=.7)
            #ax1.plot(spectral, cont_fit + imodel, ls="--", color='k', lw=1, alpha=.7)

        # == residuals == 
        ax2.__getattribute__(plottype_data)(spectral, spec - model)
        #ax2.plot(spectral, cont_fit*0, color='purple')
        #ax2.axhline(0, color='gray', ls='-', lw=.5, alpha=.5, zorder=100)
        ax2.axhline(0, color='black', ls='-', lw=.9, alpha=.5, zorder=100)

    # final fitted continuum
    if plot_cont:
        ax1.plot(spectral, cont_fit, color='purple', label="continuum")

        # data used for the second iteration continuum fit
        ax1.plot(spectral[ct_indx], spec[ct_indx], 'r.', label="cont. data", ms=10, zorder=-999)

        # == Residuals ==
        ax2.plot(spectral[ct_indx], (spec - model)[ct_indx], 'r.', ms=10)

    if plot_error:
        # from uncertainty in flux density from data reduction
        #ax1.plot(spectral, cont_fit + spec_err, color='grey', alpha=.7, lw=.5, ls=':', label="mean flux.dens. error")
        #ax1.plot(spectral, cont_fit + spec_err*2, color='grey', alpha=.7, lw=.5, ls=':')
        ax1.plot(spectral, cont_fit + spec_err*3, color='grey', alpha=.7, lw=.5, ls=':')

        # from stddev of residuals of the fit
        res_cont_std = np.nanstd(spec - model)
        #ax1.axhline(res_cont_std, color='red', alpha=.7, lw=.5, ls=':', label="stddev from residuals")
        #ax1.axhline(res_cont_std*2, color='red', alpha=.7, lw=.5, ls=':')
        ax1.axhline(res_cont_std*3, color='red', alpha=.7, lw=.5, ls=':', label="3*stddev from residuals")

        weight = 1/(spec_err/np.nanmean(spec_err))
        res_cont_std_weighted = np.nanstd(weight*(spec - model))
        ax1.axhline(res_cont_std_weighted*3, color='orange', alpha=.7, lw=.5, ls=':', label="3*stddev from residuals*weight")

        # from stdddev noise of the data in the continuum region (without lines)
        # in a continuum subtracted data
        #try:
        if 'contsub' in q3di.infile:
            outdir_noise = os.path.dirname(q3di.infile) + '/aux/'
            filename_noise = os.path.basename(q3di.infile).replace('.fits', '_noise.fits')
            filename_noise = f'{outdir_noise}/{filename_noise}'

            cont_std2D = fits.getdata(filename_noise)
            cont_std = cont_std2D[col-1, row-1]

            #ax1.axhline(cont_std, color='blue', alpha=.7, lw=.5, ls=':', 
            #            label="stddev in cont. region")
            #ax1.axhline(cont_std*2, color='blue', alpha=.7, lw=.5, ls=':')
            ax1.axhline(cont_std*3, color='blue', alpha=.7, lw=.5, ls=':',
                        label="3*stddev in cont. region")
        #except:
        #    1

        # == residuals == 
        ax2.fill_between(spectral, -spec_err, spec_err, alpha=.3)

    # axis limits
    if q3di.docontfit:
        res_cont_max = np.nanmax(abs(spec - model)[ct_indx])
        res_cont_std = np.nanstd(abs(spec - model)[ct_indx])
    else:
        res_cont_max = np.nanmax(abs(spec - model))
        res_cont_std = np.nanstd(abs(spec - model))

    cont_min = np.nanmin(cont_fit)
    model_max = np.nanmax(model)
    data_max = np.nanmax(spec)

    # ylim
    if ylim is not None:
        if is_iterable(ylim) & (len(ylim) == 2):
            ylim1 = ylim 
        #elif ylim == 'minmax':
        #    ylim(
    else:            
        ylim1 = (cont_min-res_cont_std*3, data_max+res_cont_std*3)

    ax1.set_ylim(*ylim1)
    ax2.set_ylim(-res_cont_std*7, res_cont_std*7)

    if xlim is not None:
        ax1.set_xlim(xlim)

    # legend
    ax1.legend(frameon=False, loc='best', fontsize="small")

    # axis labels 
    #ax1.xaxis.set_ticklabels([])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xlabel(xlabel)
    ax1.set_xlabel('')
    ax1.set_ylabel("Norm. flux density")
    ax2.set_ylabel("Residuals")


    # print string with values
    Amp_fit = q3do.line_fitpars['fluxpk'].to_pandas()
    wobs_fit = q3do.line_fitpars['wave'].to_pandas()
    vel_fit = (wobs_fit/(1+redshift)/linelist['lines']-1) * c_kms
    sig_fit = q3do.line_fitpars['sigma'].to_pandas()

    Amp_fit.index =  [f'amp_{i}' for i in range(len(Amp_fit))]
    vel_fit.index =  [f'dvel_{i}' for i in range(len(Amp_fit))]
    sig_fit.index =  [f'sig0_{i}' for i in range(len(Amp_fit))]
    pars = pandas.concat([Amp_fit, vel_fit, sig_fit], ignore_index=False)

    print(pars)

    # tight layout
    plt.tight_layout()

    if show:
        plt.show()

    if return_pars:
        return pars

def plot_quantity_map(
                  data, axis=None, fig=None, 
                  vmin=None, vmax=None, vpercent=None, cmap='inferno',
                  doLog=False, norm=None,
                  aspect_opts={}, 

                  do_image=True,                  
                  do_colorbar=False, 
                  colorbar_opts={},
                  do_contour=False, 
                  contour_opts={},

                  **plotargs):


    grid_dict = {}

    # auxiliary function to avoid plot u.Quantity objects
    value = quantity_to_value

    # plot args
    plotargs1 = {'origin':'lower', 'cmap':cmap, 'norm':get_norm(norm),
                 'rasterized':True, }
    plotargs1.update(plotargs)

    # fig / axis
    if fig is None:
        fig = plt.gcf()
    if axis is None:
        if len(fig.axes):
            axis = fig.gca()
        else:
            axis = fig.add_subplot()
    grid_dict['axis'] = axis

    # u.Quantity -> np.array
    data1 = np.ma.array(value(data)) * 1

    # Vlim
    if isinstance(plotargs1['norm'], mpl.colors.LogNorm):
        if (vmin is None):
            vmin = data1[data1 > 0].min()

    plotargs1['vmin'], plotargs1['vmax'] = set_vlims(data1, vmin, vmax,
                                                     vpercent)

    # Fix norm issue
    if plotargs1['norm'] is not None:
        vmin1 = plotargs1.pop('vmin')
        vmax1 = plotargs1.pop('vmax')
        plotargs1['norm'].vmin = vmin1
        plotargs1['norm'].vmax = vmax1
    else:
        vmin1 = plotargs1['vmin']
        vmax1 = plotargs1['vmax']    

    # Plot
    if do_image:
        grid_dict['img'] = axis.imshow(data1, **plotargs1)

    # Contour
    if do_contour:

        contour_opts1 = {'colors':'grey', 'norm':plotargs1['norm'], 'nlevels':6,
                         'levels':None}
        contour_opts1.update(contour_opts)

        if contour_opts1['levels'] is None:
             levels = evenly_spaced_array(vmin1, vmax1, 
                                          contour_opts1['nlevels'],
                                          scale=contour_opts1['norm'])

             contour_opts1['levels'] = levels

        grid_dict['ctour'] = axis.contour(data1, **contour_opts1)

    # aspect
    aspect_opts1 = {'aspect':'equal', 'adjustable':'box'}
    aspect_opts1.update(aspect_opts)
    if aspect_opts1 != {}:
        axis.set_aspect(**aspect_opts1)

    # colorbar
    if do_colorbar:

        colorbar_opts1 = {
                 'label':None, 'orientation':'vertical', 'format':None,
                 'divider_opts': {'position':"right", 'size':"5%", 'pad':0.01}}
        colorbar_opts1.update(colorbar_opts)

        # position
        #if colorbar_opts1['divider_opts']['position'] in ['left', 'right']:
        #    colorbar_opts1['orientation'] = 'vertical'
        #elif colorbar_opts1['divider_opts']['position'] in ['top', 'bottom']:
        #    colorbar_opts1['orientation'] = 'horizontal'
            
        # format
        if (colorbar_opts1['format'] is None):
            if isinstance(plotargs1['norm'], mpl.colors.LogNorm):
                colorbar_opts1['format'] = mpl.ticker.FuncFormatter(
                                                  lambda y, _: '{:g}'.format(y))

        # make it locatable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axis)
        divider_opts = colorbar_opts1.pop('divider_opts')
        cax = divider.append_axes(**divider_opts)

        # Colorbar
        grid_dict['cbar'] = fig.colorbar(grid_dict['img'], cax=cax, **colorbar_opts1)

    return grid_dict

#def evalNone(x):
#    try:
#        return eval(x)
#    except:
#        return None


# ------------------- Interactive plot -----------------------
def interactive_plot(q3di, img, plotmap_opts={}, plot_opts={},
                      scale_figsize=1, plotmap_opts_dict=None):
    '''

    Creates an interactive plot using Tkinter and Matplotlib.

    Parameters
    ----------
    q3di: q3din object
        q3din object
    img : 2D array or dict
        A 2D array with the image to be plotted, or a dictionary with 2D arrays as values.
    plotmap_opts : dict, optional
        A dictionary of options for plotting the map. Default is an empty dictionary.
    plot_opts : dict, optional
        A dictionary of options for plotting the spectrum. Default is an empty dictionary.
    scale_figsize : float, optional
        A scaling factor for the figure size. Default is 1.
    plotmap_opts_dict : dict, optional
        A dictionary of plotmap options for each image key if img is a dictionary. Default is None.

    Notes
    -----
    This function creates an interactive plot with Tkinter and Matplotlib, allowing the user to 
    interact with the plot through various buttons and options. The user can update spectrum and 
    map options, follow the mouse to update the plot, and click to plot specific points.

    # Documentation initialized with Copilot
    '''

    class tkclass:

        def __init__(self, window, img):

            # Default options - spectrium
            self.plot_opts = {'clean_axis':True, 
                              'plot_error':False,
                              'plot_guess':False, 'plot_cont':True, 'plot_model':True, 
                              'return_pars':True, 'rest':False, 'xlim':None,
                              'ylim':None, 'plottype_data':'step', 'show':False,
                               'wvel_line':None}
            if 'redshift' not in self.plot_opts:
                self.plot_opts['redshift'] = q3di.zsys_gas

            # Default options - maps
            self.plotmap_opts = dict(
                                 origin='lower',
                                 vpercent=[5, 95],
                                 norm='linear',
                                 do_colorbar=True,
                                 do_contour=True,
                                 cmap='inferno',
                                 vmin=None,
                                 vmax=None,
                                 )

            self.plotmap_opts.update(plotmap_opts)
            self.plot_opts.update(plot_opts)

            self.plotmap_opts_default = {}
            self.plotmap_opts_default.update(plotmap_opts)

            # Data
            self.q3di = q3dutil.get_q3dio(q3di)

            # If a dictionary is informed, 
            if isinstance(img, dict):
                self.img_dict = img                

                self.img = list(self.img_dict.values())[0]

                if plotmap_opts_dict is not None:
                    self.plotmap_opts_dict = plotmap_opts_dict
                else:
                    for key in self.img_dict.keys:                  
                        self.plotmap_opts_dict[key] = self.plotmap_opts
            else:
                self.img = img

            # TK
            self.window = window
            self.buttonPlot = tk.Button(window, text='Image plot', command=self.choose_plot_img)
            self.buttonFollow = tk.Button(window, text='Follow mouse', command=self.follow)
            
            self.buttonSinglePlot = tk.Button(window, text='Plot on click', command=self.singleplot)

            if 1:
                self.text = tk.Text(window, bg='white', height=15, width=80,
                                    font=('Fixedsys', 12), wrap="none")

            #else:
                #Xscroll = tk.Scrollbar(self.text, orient=tk.HORIZONTAL)
                #xscrollbar = tk.Scrollbar(self.window, orient=tk.HORIZONTAL)
                #xscrollbar.grid(row=1, column=0)#, sticky=tk.N+tk.S+tk.E+tk.W)
                #xscrollbar.pack(side=tk.BOTTOM, fill='x')
                #self.text = tk.Text(self.window, bg='white', height=10, width=80,
                #                    font=('Fixedsys', 12),
                #                    xscrollcommand=xscrollbar.set,
                #                    wrap="none")
                #self.xscrollbar = xscrollbar
                #self.text.pack()

            #t.configure(yscrollcommand=Yscroll.set)
            #t.configure(xscrollcommand=Xscroll.set)

            #Yscroll.config(command=t.yview)
            #Xscroll.config(command=t.xview)

            #Xscroll.pack(side=tk.BOTTOM)

            # -----------------------
            self.fig_img = Figure(figsize=np.array([6, 6])*scale_figsize)
            self.fig_fit = Figure(figsize=np.array([12, 6])*scale_figsize)

            ax_img = self.fig_img.add_subplot(111)
            gs = gridspec.GridSpec(5, 1)
            ax_spec1 = self.fig_fit.add_subplot(gs[0:4])
            ax_spec2 = self.fig_fit.add_subplot(gs[4], sharex=ax_spec1)

            self.ax_img = ax_img
            self.ax_spec = [ax_spec1, ax_spec2] 

            # === Canvas ===
            # Canvas - Initialize
            self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self.window)
            self.canvas_spec = FigureCanvasTkAgg(self.fig_fit, master=self.window)

            # Canvas - Define the grid region to be used
            self.canvas_img.get_tk_widget().grid(row=0, column=0, rowspan=6, columnspan=6)
            self.canvas_spec.get_tk_widget().grid(row=0, column=7, rowspan=6, columnspan=12)


            tb_frame = tk.Frame(self.window)
            tb_frame.grid(row=7, column=0, columnspan=6, sticky='W')
            toolbar = NavigationToolbar2Tk(self.canvas_img, tb_frame)
            toolbar.update()

            other_tb_frame = tk.Frame(self.window)
            other_tb_frame.grid(row=7, column=7, columnspan=12, sticky='W')
            other_toolbar = NavigationToolbar2Tk(self.canvas_spec, other_tb_frame)
            other_toolbar.update()

            # Add buttons
            self.buttonPlot.grid(row=8, column=0, sticky='W')
            self.buttonFollow.grid(row=9, column=0, sticky='W')
            self.buttonSinglePlot.grid(row=10, column=0, sticky='W')

            # Add text grid
            self.text.grid(row=8, column=7, rowspan=4)

            # Input - Updade spectrum option
            self.buttonSpecProperties = tk.Button(window,
                         text='Spectrum options', command=self.input_spec_opts)
            self.buttonSpecProperties.grid(row=8, column=1)

            # Input - Update map options            
            self.buttonMapProperties = tk.Button(window,
                         text='Map options', command=self.input_map_opts)
            self.buttonMapProperties.grid(row=9, column=1)            

            # 
            # Creating a Listbox and 
            listbox = tk.Listbox(window)#, height=10, width=10) 
            listbox.grid(row=8, column=5, rowspan=4)

            # attaching it to root window 
            #scrollbar = tk.Scrollbar(window) 
              
            # Insert elements into the listbox 
            if hasattr(self, 'img_dict'):
                for values in list(self.img_dict.keys()): 
                    listbox.insert(tk.END, values) 

            self.listbox = listbox

            # update dicts with figs and axes objects
            #self.plotmap_opts.update(fig=self.fig_img, axis=self.ax_img)
            self.plot_opts.update(fig=self.fig_fit, axis=self.ax_spec)

            # Key input
            self.window.bind("<Key>", self.key_handler)

        def key_handler(self, event):

            if event.keysym == 'Right':

                self.x = self.x + 1
                self.xcol = self.xcol + 1

                self.plot_spec()

            elif event.keysym == 'Left':

                self.x = self.x - 1 
                self.xcol = self.xcol - 1 

                self.plot_spec()

            elif event.keysym == 'Down':

                self.y = self.y - 1
                self.yrow = self.yrow - 1

                self.plot_spec()

            elif event.keysym == 'Up':

                self.y = self.y + 1
                self.yrow = self.yrow + 1

                self.plot_spec()


        def input_spec_opts(self):
            entry_list = ['rest', 'redshift', 'xlim', 'ylim', 'plot_guess', 
                          'plot_cont', 'plot_model', 'plot_error', 
                          'wvel_line', 'plottype_data']
            output_dict = self.plot_opts

            self.create_input_box(entry_list, output_dict)

        def input_map_opts(self):
            entry_list = ['do_colorbar', 'do_contour', 'cmap',
                          'norm',  'vpercent', 'vmin', 'vmax']
            output_dict = self.plotmap_opts

            self.create_input_box(entry_list, output_dict)

        def create_input_box(self, entry_list, output_dict):
            window_spec = tk.Toplevel(self.window)
            self.tkentries = {}

            for row, entrytext in enumerate(entry_list):

                self.add_entry(window_spec, row, entrytext,
                               output_dict[entrytext])

            self.input_window = window_spec
            self.input_entry_list = entry_list
            self.input_output_dict = output_dict
            
            # OK Buttom, that updates the dictionary and replot the image/spec
            ButtomOK = tk.Button(window_spec, text="Update", command=self.get_values)
            ButtomOK.grid(row=row+1, column=0)

        def add_entry(self, window, row, text, default_value):
            #v = tk.StringVar(window, value='default text')
            E = tk.Entry(window)
            
            E.insert(tk.END, json.dumps(default_value))

            tk.Label(window, text=text+':').grid(row=row, column=0)
            E.grid(row=row, column=1)
            self.tkentries[text] = E

        def get_values(self):
            window = self.input_window
            entry_list = self.input_entry_list
            output_dict = self.input_output_dict 

            entry_values = {}
            for entry in entry_list:

                tkentry = getattr(self, 'tkentries')[entry]
                value = tkentry.get()

                if len(value):
                    try:
                        entry_values[entry] = json.loads(value)
                    except:
                        entry_values[entry] = json.loads(value.lower())
            output_dict.update(entry_values) 

            # update map plot
            if 'rest' in output_dict:

                self.plot_spec()
            # update spectrum plot
            elif 'cmap' in output_dict:

                self.plot_img()

        def choose_plot_img(self):

            if hasattr(self, 'img_dict'):
                img_key = self.listbox.get(tk.ACTIVE)

                self.img = self.img_dict[img_key]

                self.plotmap_opts = {}

                if img_key in self.plotmap_opts_dict:
                    self.plotmap_opts.update(self.plotmap_opts_dict[img_key])
                else:
                    key_dict = [key for key in self.plotmap_opts_dict.keys() if key in img_key]

                    self.plotmap_opts.update(self.plotmap_opts_dict[key_dict[0]])

            self.plot_img()

        def plot_img(self):
            
            # clean axis map
            self.ax_img.clear()
            if hasattr(self, 'grid_dict'):

                if 'cbar' in self.grid_dict:
                    # remove colorbar
                    self.grid_dict['cbar'].ax.remove()

            self.grid_dict = plot_quantity_map(self.img, 
                                               fig=self.fig_img,
                                               axis=self.ax_img,
                                               **self.plotmap_opts)

            self.canvas_img.draw()

        def plot_spec(self):

            self.text.delete('1.0', 'end')
            self.text.insert('insert', '(x, y)     = ({:4d}, {:4d})\n'.format(self.x, self.y))
            self.text.insert('insert', '(col, row) = ({:4d}, {:4d})\n'.format(self.xcol, self.yrow))

            self.highlight_pixel()#i, j)


            try:
                # Call plot spec funtionc
                pars = plot_single_fit(self.q3di, 
                                       col=self.xcol, row=self.yrow,
                                       **self.plot_opts)
            
                #plt.show()

                # Print parameters values to text box
                self.text.insert('insert', pars.to_string())

                #self.xscrollbar.config(command=self.text.xview)

            except IndexError:
                self.text.insert('insert', 'Outside FoV!')

            self.canvas_spec.draw()

        def singleplot(self):
            try:
                self.canvas_img.mpl_disconnect(self.connect_id)
            except:
                pass
            self.connect_id = self.canvas_img.mpl_connect('button_press_event', self.onclick)

        def follow(self):
            try:
                self.canvas_img.mpl_disconnect(self.connect_id)
            except AttributeError:
                pass
            self.connect_id = self.canvas_img.mpl_connect('motion_notify_event', self.onclick)

        def onclick(self, event):
            # Action to draw spectrum based on the (x,y) position of the pointer

            try:
                x, y = [int(xy + 0.5) for xy in (event.xdata, event.ydata)]
                if np.any(np.array([x, y]) < 0):
                    self.text.delete('1.0', 'end')
                    self.text.insert('insert', 'Index Error!')

                    return
            except AttributeError:
                self.text.delete('1.0', 'end')
                self.text.insert('insert', 'You clicked outside the plot!')
                return

            self.x = x
            self.y = y

            self.xcol = self.x  + 1
            self.yrow = self.y + 1

            self.plot_spec()

        def highlight_pixel(self):#, x, y):
                """ Outline the Pixel that was clicked on """
                from matplotlib.patches import Rectangle

                x = self.x
                y = self.y

                # remove current highlight
                for patch in self.ax_img.patches:
                    patch.remove()
                # new highlight
                rect = Rectangle((x-0.5, y-0.5), 1, 1, ec="cyan", fc="none")
                self.ax_img.add_patch(rect)
                self.canvas_img.draw()

    app_window = tk.Tk()
    start = tkclass(app_window, img)
    app_window.mainloop()


def call_plot_interactive(q3di=None, opts_load_maps={}, q3di_continuum=None,
                          interactive_plot_opts={}, opts_continuum={}):

    opts_continuum1 = dict(flux_density=True, use_same_width=True,
                         wavelenth_range=None, filename_noise=None,
                         SNR_cut=None)
    opts_continuum1.update(opts_continuum)

    # Load q3di data
    q3di = q3dutil.get_q3dio(q3di)

    # Get linelist and linenames
    linelist = q3dutil.get_linelist(q3di)
    linenames = linelist['name'].value

    # Load cube data
    cube = q3di.load_cube()
    opts_load_maps['cube'] = cube

    # Initialize dictionaries and lists
    linemaps = {}
    img_dict = {}
    plotmap_opts_dict = {}
    data_list = []
    ext_names = []

    # Add general data
    key_list_general = ['redchisq', 'maxncomp']
    line_ref = linenames[0]
    linemaps_ref = bq3dutils.load_linemaps(q3di, line_ref, **opts_load_maps)
    for key in key_list_general:

        img_dict[f'{key}'] = getattr(linemaps_ref, key)

    # =============
    # Add continuum
    if q3di_continuum is None:

        q3di_continuum = q3di

    img_cont = bq3dutils.get_continuum_img(q3di_continuum, **opts_continuum1)

    plotmap_opts_cont = dict(doLog=True, 
                            vpercent=[1, 99],
                            do_contour=True,
                            do_colorbar=True,
                            cmap='inferno',
                            contour_opts=(dict(nlevels=20, colors='white',
                                    linewidths=0.9, alpha=0.5)),
                            vmin=None, vmax=None,
                            norm='log',
                                    )
    img_dict['cont_optical'] =  img_cont
    plotmap_opts_dict.update({'cont':plotmap_opts_cont})
    ext_names += ['cont_optical'] 

    # =============
    # Get values from linemaps
    key_list = 'pkflux', 'vel', 'sig', 'flux'
    for line in linenames:
        linemaps[line] = bq3dutils.load_linemaps(q3di, line, **opts_load_maps) 

        for key in key_list:
            img_dict[f'{line}_{key}'] = getattr(linemaps[line], key)

        # Get total flux
        img_dict[f'{line}_flux_tot'] = np.sum(img_dict[f'{line}_flux'], axis=0)

        # Split components
        ncomp = len(img_dict[f'{line}_{key}'])
        for comp in range(ncomp):
            for key in key_list:
                img_dict[f'{line}_{key}_c{comp}'] = img_dict[f'{line}_{key}'][comp]

        # Remove 3D data
        for key in key_list:
            _ = img_dict.pop(f'{line}_{key}')

    # Add linemaps for last
    ext_names += list(img_dict.keys())

    for ext in ext_names:

        datai = img_dict[ext].data.value
        data_list += [datai] 

    # =================================
    # Define default options for plotting
    # Flux
    plotmap_opts_flux = dict(
                    origin='lower',
                    vpercent=[1, 99],
                    vmin=None, vmax=None,
                    norm='log',
                    do_colorbar=True,
                    do_contour=True,
                    cmap='magma')

    # Sigma
    plotmap_opts_sig = dict(
                    origin='lower',
                    vpercent=[5, 95],
                    vmin=None, vmax=None,
                    norm='log',
                    do_colorbar=True,
                    do_contour=True,
                    cmap='inferno')

    # Vel
    plotmap_opts_vel = dict(
                    origin='lower',
                    vmin=None, vmax=None,
                    vpercent=[1, 99],
                    norm='linear',
                    do_colorbar=True,
                    do_contour=True,
                    cmap='coolwarm')

    # redchisq
    plotmap_opts_redchisq = dict(
                    origin='lower',
                    vpercent=None,
                    vmin=0.5, vmax=1.5,
                    norm='linear',
                    do_colorbar=True,
                    do_contour=False,
                    cmap='RdGy_r')

    # maxncomp
    plotmap_opts_ncomp = dict(
                    origin='lower',
                    vpercent=None,
                    vmin=None, vmax=None,
                    norm='linear',
                    do_colorbar=True,
                    do_contour=False,
                    colorbar_opts=dict(ticks=np.arange(1, 8), 
                                      boundaries=np.arange(0.5, 8.5, 1)),
                    cmap='viridis')

    # ======
    # Update
    plotmap_opts_dict.update({'flux':plotmap_opts_flux,
                              'sig':plotmap_opts_sig,
                              'vel':plotmap_opts_vel,
                              'redchisq':plotmap_opts_redchisq,
                              'ncomp':plotmap_opts_ncomp})

    # =================================
    # Call interactive plot function
    interactive_plot_opts1 = dict(scale_figsize=1.1,
                                  plotmap_opts_dict=plotmap_opts_dict,
                                  plot_opts=dict(plot_guess=False,
                                                 plot_cont=False,
                                                 #wvel_line='[OIII]5007',
                                                 #rest=True
                                                 )
                                 )
    
    interactive_plot_opts1.update(interactive_plot_opts)

    interactive_plot(q3di, img_dict, **interactive_plot_opts1)










