import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import matplotlib as mpl

# astropy
import astropy
from astropy.io import fits
from astropy import constants
from astropy import units as u

# q3dfit
from q3dfit.q3din import q3din
from q3dfit.q3df import q3dfit
from q3dfit.q3dcollect import q3dcollect
from q3dfit.q3dout import load_q3dout
import q3dfit.q3dpro as q3dpro
from q3dfit import q3dutil

# iterative plot
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

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
            return x.data
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
def plot_single_fit(filename_q3di, x, y, wvel_line=None, rest=True,
                    axis=None, fig=None,  clean_axis=False, return_pars=False):
    '''
    Input
    -----
    filename_q3di: str
        Filename of the q3di configuration file
    x, y: int
        Column, row (first index=0) of the data cube to be fitted.
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
   
    '''

    # ====================================
    # Load
    #print(filename_q3di, x, y)
    q3di = q3dutil.get_q3dio(filename_q3di)
    q3do = load_q3dout(q3di, x, y)

    linelist = q3dutil.get_linelist(q3di)
    # only saved if checkcomp was used
    q3do.sepfitpars()

    # ====================================
    # Spectral
    xlabel = "Wavelength ($\mu m$)"

    spectral = q3do.wave * 1
    if rest:
        spectral = spectral / (1 + q3do.zstar) 
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
    cont_fit = q3do.cont_fit
    line_fit = q3do.line_fit
    model = cont_fit + line_fit
    ct_indx = q3do.ct_indx

    # FIXME: use q3dfit way of generating the gaussian profiles
    if 0:# TODO from fitspec.py        
        q3do.line_fit = emlmod.eval(lmout.params, x=gdlambda)

    # Initial guess - second fit (result first fit)
    # FIXME also convolve the guess models
    p1 = np.array([item.value for key, item in q3do.parinit.items()])
    p1[2::3] = p1[2::3] / c_kms.value  * p1[1::3]#linelist['lines']
    line_init1 = gaussian(q3do.wave, *p1)

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

    # ====================================
    # plots
    if fig is None:
        fig = plt.figure()
    if axis is None:
        axis = fig.subplots(2, 1)
    if clean_axis:
        axis[0].cla()
        axis[1].cla()
    ax = axis 

    # data used for the second iteration continuum fit
    ax[0].plot(spectral[ct_indx], spec[ct_indx], 'r.', label="cont. data", ms=10)
    # data
    ax[0].plot(spectral, spec, label="data")
    # full model (result from from first fit iteration)
    ax[0].plot(spectral, cont_fit + line_init1, label="guess1")
    # final full model
    ax[0].plot(spectral, model, label="model")
    # final fitted continuum
    ax[0].plot(spectral, cont_fit, color='purple', label="continuum")
    # individual components
    for imodel, imodel0 in zip(model_comps, model_comps0):
        ax[0].plot(spectral, cont_fit + imodel, ls="--", color='k', lw=1, alpha=.7)

    # residuals 
    ax[1].plot(spectral[ct_indx], (spec - model)[ct_indx], 'r.', ms=10)
    ax[1].plot(spectral, spec - model)
    #ax[1].plot(spectral, cont_fit*0, color='purple')
    ax[1].axhline(0, color='gray', ls='-', lw=.5, alpha=.5)

    # axis limits
    res_cont_max = np.nanmax(abs(spec - model)[q3do.ct_indx])
    res_cont_std = np.nanstd(abs(spec - model)[q3do.ct_indx])

    cont_min = np.nanmin(cont_fit)
    model_max = np.nanmax(model)
    #ax[1].set_ylim(-res_cont_max*1.1, +res_cont_max*1.1)
    #ax[0].set_ylim(cont_min-res_cont_max*1.1, model_max*1.2)
    ax[0].set_ylim(cont_min-res_cont_std*3, model_max+res_cont_std*3)
    ax[1].set_ylim(-res_cont_std*7, res_cont_std*7)

    # legend
    ax[0].legend(frameon=False, loc='best', fontsize="small")

    # axis labels 
    ax[0].xaxis.set_ticklabels([])
    ax[1].set_xlabel(xlabel)
    ax[0].set_xlabel('')
    ax[0].set_ylabel("Normalized flux density")
    ax[1].set_ylabel("Residuals")

    # tight layout
    plt.tight_layout()

    # print string with values
    Amp_fit = q3do.line_fitpars['fluxpk'].to_pandas()
    wobs_fit = q3do.line_fitpars['wave'].to_pandas()
    vel_fit = (wobs_fit/(1+q3do.zstar)/linelist['lines']-1) * c_kms
    sig_fit = q3do.line_fitpars['sigma'].to_pandas()

    Amp_fit.index =  [f'amp_{i}' for i in range(len(Amp_fit))]
    vel_fit.index =  [f'vel_{i}' for i in range(len(Amp_fit))]
    sig_fit.index =  [f'sig_{i}' for i in range(len(Amp_fit))]
    pars = pandas.concat([Amp_fit, vel_fit, sig_fit], ignore_index=False)

    print(pars)
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
                                          contour_opts1['nlevels'])

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
def interactive_plot(filename_q3di, img, plotmap_opts1={}):
             

    class tkclass:

        def __init__(self, window, img):

            # Data
            self.filename_q3di = filename_q3di
            self.img = img

            # Initial options
            #self.plot_opts = {'xlim':None, 'ylim':None,
            #                  'spectral_axis':self.spectral_axis}

            self.plot_opts = {'clean_axis':True}
            self.plotmap_opts = dict(
                                 origin='lower',
                                 vpercent=[15, 95],
                                 norm='log',
                                 do_colorbar=True,
                                 do_contour=True,
                                 cmap='inferno'
                                 )
            self.plotmap_opts.update(plotmap_opts1)

            print(1)

            # TK
            self.window = window
            self.buttonPlot = tk.Button(window, text='Image plot', command=self.plot)
            self.buttonFollow = tk.Button(window, text='Follow mouse', command=self.follow)
            
            self.buttonSinglePlot = tk.Button(window, text='Plot on click', command=self.singleplot)
            self.text = tk.Text(window, bg='white', height=10, width=80, font=('Fixedsys', 12))

            # -----------------------
            self.fig_img = Figure(figsize=(8, 8))
            self.fig_fit = Figure(figsize=(16, 8))

            self.ax1 = self.fig_img.add_subplot(111)
            self.ax2 = self.fig_fit.subplots(2, 1)

            #
            canvas = FigureCanvasTkAgg(self.fig_img, master=self.window)
            canvas2 = FigureCanvasTkAgg(self.fig_fit, master=self.window)

            canvas.get_tk_widget().grid(row=0, column=0, rowspan=6, columnspan=6)
            canvas2.get_tk_widget().grid(row=0, column=7, rowspan=6, columnspan=12)

            tb_frame = tk.Frame(window)
            tb_frame.grid(row=7, column=0, columnspan=6, sticky='W')
            toolbar = NavigationToolbar2Tk(canvas, tb_frame)
            toolbar.update()

            other_tb_frame = tk.Frame(window)
            other_tb_frame.grid(row=7, column=7, columnspan=12, sticky='W')
            other_toolbar = NavigationToolbar2Tk(canvas2, other_tb_frame)
            other_toolbar.update()

            self.buttonPlot.grid(row=8, column=0, sticky='W')
            self.buttonFollow.grid(row=9, column=0, sticky='W')
            self.buttonSinglePlot.grid(row=10, column=0, sticky='W')

            self.text.grid(row=8, column=7, rowspan=4)

            # Input - new
            #self.buttonSpecProperties = tk.Button(window,
            #             text='Spectrum options', command=self.input_spec_opts)
            #self.buttonSpecProperties.grid(row=8, column=1)
            #
            #self.buttonMapProperties = tk.Button(window,
            #             text='Map options', command=self.input_map_opts)
            #self.buttonMapProperties.grid(row=9, column=1)            

            self.canvas = canvas
            self.canvas2 = canvas2
            canvas.draw()

            # update dicts
            self.plotmap_opts.update(fig=self.fig_img, axis=self.ax1)
            #self.plot_opts.update(fig=self.fig_fit, axis=self.ax2)
            self.plot_opts.update(fig=self.fig_fit, axis=self.ax2)

        #def input_spec_opts(self):
        #    entry_list = ['xlim', 'ylim', 'radius']
        #    output_dict = self.plot_opts

        #    self.create_input_box(entry_list, output_dict)

        #def input_map_opts(self):
        #    entry_list = ['vmin', 'vmax', 'doLog', 'do_colorbar', 'do_contour']
        #    output_dict = self.plotmap_opts

        #    self.create_input_box(entry_list, output_dict)

        #def input_radius(self):
        #    entry_list = ['radius']
        #    output_dict = self.radius
        #    self.create_input_box(entry_list, output_dict)

        #def create_input_box(self, entry_list, output_dict):
        #    window_spec = tk.Toplevel(self.window)
        #    self.tkentries = {}

        #    for row, entrytext in enumerate(entry_list): 
        #        self.add_entry(window_spec, entrytext, row)

        #    self.input_window = window_spec
        #    self.input_entry_list = entry_list
        #    self.input_output_dict = output_dict
            
        #    ButtomOK = tk.Button(window_spec, text="OK", command=self.get_values)
        #    ButtomOK.grid(row=row+1, column=0)

        #def add_entry(self, window, text, row):
        #    E = tk.Entry(window)
        #    tk.Label(window, text=text+':').grid(row=row, column=0)
        #    E.grid(row=row, column=1)
        #    self.tkentries[text] = E

        #def get_values(self):
        #    window = self.input_window
        #    entry_list = self.input_entry_list
        #    output_dict = self.input_output_dict 

        #    entry_values = {}
        #    for entry in entry_list:

        #        tkentry = getattr(self, 'tkentries')[entry]
        #        value = tkentry.get()
        #        entry_values[entry] = evalNone(value)

        #    output_dict.update(entry_values) 


        def plot(self, contrast=1):

            grid_dict = plot_quantity_map(self.img, **self.plotmap_opts)
            print('*****************************')

            self.canvas.draw()


        def singleplot(self):
            try:
                self.canvas.mpl_disconnect(self.connect_id)
            except:
                pass
            self.connect_id = self.canvas.mpl_connect('button_press_event', self.onclick)

        def follow(self):
            try:
                self.canvas.mpl_disconnect(self.connect_id)
            except AttributeError:
                pass
            self.connect_id = self.canvas.mpl_connect('motion_notify_event', self.onclick)

        def onclick(self, event):

            try:
                i, j = [int(np.floor(x) + 0.5) for x in (event.xdata, event.ydata)]
                if np.any(np.array([i, j]) < 0):
                    self.text.delete('1.0', 'end')
                    self.text.insert('insert', 'Index Error!')
                    return
            except AttributeError:
                self.text.delete('1.0', 'end')
                self.text.insert('insert', 'You clicked outside the plot!')
                return

            self.text.delete('1.0', 'end')
            self.text.insert('insert', '({:6d}, {:6d})\n'.format(i, j))

            try:
                pars = plot_single_fit(self.filename_q3di, x=i, y=j,
                                            return_pars=True,
                                            **self.plot_opts)

                self.text.insert('insert', '({:6d}, {:6d})\n{}'.format(
                                                        i, j, pars.to_string()))


            except IndexError:
                self.text.insert('insert', 'Index Error!')

            self.canvas2.draw()

    app_window = tk.Tk()
    start = tkclass(app_window, img)
    app_window.mainloop()









