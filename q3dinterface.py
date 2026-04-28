import numpy as np
import matplotlib.pyplot as plt
import pandas
import os, sys
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
from matplotlib.patches import Rectangle


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
def plot_single_fit(q3do, q3di, x, y, wvel_line=None, rest=True, axis=None, fig=None,  clean_axis=False, return_pars=False, plot_initial_guess=False):
    '''
    Input
    -----
    # filename_q3di: str
    #     Filename of the q3di configuration file
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
    

    linelist = q3dutil.get_linelist(q3di)
    # only saved if checkcomp was used
    q3do.sepfitpars()

    # ====================================
    # Spectral
    xlabel = r"Wavelength ($\mu m$)"
    
    if not hasattr(q3do, "zstar"):
        q3do.zstar = q3di.zsys_gas
    if q3do.zstar is None:
        q3do.zstar = q3di.zsys_gas

    spectral = q3do.wave * 1
    if rest:
        spectral = spectral / (1 + q3do.zstar) 
        xlabel = r"Rest wavelength ($\mu m$)"

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

    # FIXME: use q3dfit way of generating the gaussian profiles
    # if 0:# TODO from fitspec.py        
    #     q3do.line_fit = emlmod.eval(lmout.params, x=gdlambda)

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

    # data
    ax[0].plot(spectral, spec, label="data", zorder=10)
    # print(spec.max())

    if q3do.docontfit:
        cont_fit = q3do.cont_fit
        ct_indx = q3do.ct_indx

        # data used for the second iteration continuum fit
        ax[0].plot(spectral[ct_indx], spec[ct_indx], 'r.', label="cont. data", ms=10)
        # final fitted continuum
        ax[0].plot(spectral, cont_fit, color='purple', label="continuum")
    else:
        cont_fit = np.zeros_like(spectral)
    
    if q3do.dolinefit:# and not q3do.discard:
        line_fit = q3do.line_fit
        model = cont_fit + line_fit
        # Initial guess - second fit (result first fit)
        # FIXME also convolve the guess models
        if plot_initial_guess:
            p1 = np.array([item.value for key, item in q3do.parinit.items()])
            print(np.array([(item.min, item.max) for key, item in q3do.parinit.items()]))
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
        model_comps = np.array([ispecres.spect_convolver(q3do.wave, imodel0, pi[1]) for pi, imodel0, ispecres in zip(pp, model_comps0, specres)])

        # final full model
        ax[0].plot(spectral, model, label="model")
        if plot_initial_guess:
            # full model (result from from first fit iteration)
            ax[0].plot(spectral, cont_fit + line_init1, label="guess1")
        # individual components
        for i, (imodel, imodel0) in enumerate(zip(model_comps, model_comps0)):
            ax[0].plot(spectral, cont_fit + imodel, ls="--", color=('k' if i >= len(model_comps)//3 else "r"), lw=1, alpha=.7)
        # residuals
        ax[1].plot(spectral, spec - model)
        #ax[1].plot(spectral, cont_fit*0, color='purple')
        ax[1].axhline(0, color='gray', ls='-', lw=.5, alpha=.5)

    if q3do.docontfit and q3do.dolinefit:# and not q3do.discard:
        # residuals 
        ax[1].plot(spectral[ct_indx], (spec - model)[ct_indx], 'r.', ms=10)
    
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

    # print(q3do.line_fitpars["fluxpk"]["[OIII]5007"].min())

    # print(pars)
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


class Q3DInterface:
    """ An interactive way to view and plot using q3dfit """

    def __init__(self, window: tk.Tk, label, onefile=False, path="", img=None):
        i = 1
        self.q3di = {}
        self.onefile = onefile
        if onefile:
            self.q3di[i] = q3dutil.get_q3dio(os.path.join(path, "input", f"q3di_{label}.npy"))
        else:
            while i < 10:
                try:
                    self.q3di[i] = q3dutil.get_q3dio(os.path.join(path, "input", f"q3di_{label}_{i}.npy"))
                except FileNotFoundError:
                    break
                i += 1

        self.ncomp = 4

        # TK
        self.window = window

        # select number of components
        self.ncomp_listbox = tk.Listbox(window, font=('Fixedsys', 24), selectmode="single")
        self.ncomp_listbox.grid(row=8, column=9, rowspan=3, sticky="W")
        self.ncomp_listbox.bind("<<ListboxSelect>>", self.select_comp)
        if not onefile:
            def fake_plain_arrow(event):
                # Remove Shift and inject plain arrow key
                keysym = event.keysym
                if keysym == 'Up':
                    self.ncomp_listbox.event_generate('<Up>')
                elif keysym == 'Down':
                    self.ncomp_listbox.event_generate('<Down>')
                self.window.after(1, self.select_active_item)
                return "break"
            self.ncomp_listbox.bind("<Shift-Up>", fake_plain_arrow)
            self.ncomp_listbox.bind("<Shift-Down>", fake_plain_arrow)
        
        # display aic values
        if not onefile:
            self.aic_text = tk.Text(self.window, bg="white", height=20, width=32)
            self.aic_text.grid(row=8, column=8, rowspan=6)

        self.snr_text = tk.Text(self.window, bg="white", height=20, width=32)
        self.snr_text.grid(row=8, column=10, rowspan=6)

        # plot button
        self.buttonPlot = tk.Button(self.window, text='Image plot', command=self.plot_img)
        self.buttonPlot.grid(row=8, column=0, sticky='W')
        # follow mouse button
        self.buttonFollow = tk.Button(self.window, text='Follow mouse', command=self.follow)
        self.buttonFollow.grid(row=9, column=0, sticky='W')
        # click on plot button
        self.buttonSinglePlot = tk.Button(self.window, text='Plot on click', command=self.singleplot)
        self.buttonSinglePlot.grid(row=10, column=0, sticky='W')
        # plot initial guess button
        self.show_initial_guess = False
        self.initial_guess_button = tk.Button(self.window, text="Toggle Inital Guess", command=self.toggle_initial_guess)
        self.initial_guess_button.grid(row=11, column=0, sticky="W")
        # text box
        self.text = tk.Text(self.window, bg='white', height=20, width=80, font=('Fixedsys', 16))
        self.text.grid(row=8, column=7, rowspan=6)

        # image figure
        self.fig_img = Figure(figsize=(8, 8))
        self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self.window)
        self.canvas_img_widget = self.canvas_img.get_tk_widget()
        self.canvas_img_widget.grid(row=0, column=0, rowspan=6, columnspan=6)
        self.window.after(100, self.canvas_img_widget.focus_set)

        tb_frame_img = tk.Frame(self.window)
        tb_frame_img.grid(row=7, column=0, columnspan=6, sticky='W')
        toolbar_img = NavigationToolbar2Tk(self.canvas_img, tb_frame_img)
        toolbar_img.update()

        # fit figure
        self.fig_fit = Figure(figsize=(16, 8))
        self.ax_fit = self.fig_fit.subplots(2, 1)
        self.canvas_fit = FigureCanvasTkAgg(self.fig_fit, master=self.window)
        self.canvas_fit.get_tk_widget().grid(row=0, column=7, rowspan=6, columnspan=12)

        tb_frame_fit = tk.Frame(self.window)
        tb_frame_fit.grid(row=7, column=7, columnspan=12, sticky='W')
        toolbar_fit = NavigationToolbar2Tk(self.canvas_fit, tb_frame_fit)
        toolbar_fit.update()

        # Generating a map that will be used as a base image to locate spaxels
        # linedat = q3dpro.LineData(self.q3di)
        # o3data = q3dpro.OneLineData(linedat, '[OIII]5007')
        # self.img = np.nansum(o3data.flux.T, axis=0)
        # # plot/fit options
        # self.plotmap_opts = dict(
        #                         origin='lower',
        #                         vpercent=[15, 95],
        #                         norm='log',
        #                         do_colorbar=True,
        #                         do_contour=True,
        #                         cmap='inferno'
        #                         )
        # plot/fit options
        self.plotmap_opts = dict(origin='lower', norm='lin', do_colorbar=True)
        if img is None:
            number_of_components = np.load(os.path.join(path, "noc.npy"))
            self.img = number_of_components.max(axis=0).T
            if onefile:
                q3di = q3dutil.get_q3dio(self.q3di[1])
                self.img = q3di.ncomp[list(q3di.ncomp.keys())[0]].T
                # self.img = q3di.ncomp["[OIII]5007"].T
        else:
            self.img = img

        self.img_size = self.img.T.shape

        self.plot_opts = {'clean_axis':True}

        self.plotmap_opts.update(fig=self.fig_img, axis=self.ax_img)
        self.plot_opts.update(fig=self.fig_fit, axis=self.ax_fit)

        self.canvas_img.mpl_connect("key_press_event", self.onkey)
        self.selected_pixel = [18-1, 33-1]

        if not onefile:
            self.shift_pressed = False
            self.window.bind_all("<KeyPress-Shift_L>", self.shift_press)
            self.window.bind_all("<KeyPress-Shift_R>", self.shift_press)
            self.window.bind_all("<KeyRelease-Shift_L>", self.shift_release)
            self.window.bind_all("<KeyRelease-Shift_R>", self.shift_release)

        self.plot_img()

        try:
            self.display_fit(*self.selected_pixel)
        except Exception:
            ...

        # activate click on plot on start
        self.connect_id = self.canvas_img.mpl_connect('button_press_event', self.onclick)

    def plot_img(self):
        """ Show the Image for selecting spaxels """
        plot_quantity_map(self.img, **self.plotmap_opts)
        self.canvas_img.draw()
        
    def singleplot(self):
        """ Mode: Show Fits when clicking on a spaxel """
        try:
            self.canvas_img.mpl_disconnect(self.connect_id)
        except AttributeError:
            pass
        self.connect_id = self.canvas_img.mpl_connect('button_press_event', self.onclick)

    def follow(self):
        """ Mode: Show Fits when hovering over image """
        try:
            self.canvas_img.mpl_disconnect(self.connect_id)
        except AttributeError:
            pass
        self.connect_id = self.canvas_img.mpl_connect('motion_notify_event', self.onclick)

    def onclick(self, event):
        """ Handle click event """
        try:
            i, j = [int(np.round(x)) for x in (event.xdata, event.ydata)]
            if np.any(np.array([i, j]) < 0):
                self.text.delete('1.0', 'end')
                self.text.insert('insert', 'Index Error!')
                return
        except AttributeError:
            self.text.delete('1.0', 'end')
            self.text.insert('insert', 'You clicked outside the plot!')
            return
        self.selected_pixel = [i, j]
        self.display_fit(i, j)

    def onkey(self, event):
        if event.key == 'up':
            self.selected_pixel[1] += 1
            self.selected_pixel[1] %= self.img_size[1]
        elif event.key == 'down':
            self.selected_pixel[1] -= 1
            self.selected_pixel[1] %= self.img_size[1]
        elif event.key == 'right':
            self.selected_pixel[0] += 1
            self.selected_pixel[0] %= self.img_size[0]
        elif event.key == 'left':
            self.selected_pixel[0] -= 1
            self.selected_pixel[0] %= self.img_size[0]

        self.display_fit(*self.selected_pixel)

    def shift_press(self, event):
        if event.keysym in ("Shift_L", "Shift_R"):
            self.shift_pressed = True
            # print("START SHIFT")
            self.window.after(10, self.ncomp_listbox.focus_set)
            self.window.after(11, self.select_active_item)

    def shift_release(self, event):
        if event.keysym in ("Shift_L", "Shift_R"):
            self.shift_pressed = False
            # print("END SHIFT")
            self.window.after(10, self.canvas_img_widget.focus_set)

    def refocus_canvas(self, event=None):
        if self.shift_pressed:  # Shift held
            self.ncomp_listbox.focus_set()
        else:
            self.window.after(10, self.canvas_img_widget.focus_set)

    def select_active_item(self):
        active = self.ncomp_listbox.index("active")
        self.ncomp_listbox.selection_clear(0, tk.END)
        self.ncomp_listbox.selection_set(active)
        self.ncomp_listbox.activate(active)
        self.select_comp(None)

    def select_comp(self, event):
        self.ncomp = self.ncomp_listbox.curselection()[0] + 1
        self.display_fit(*self.selected_pixel)
        self.refocus_canvas(event)

    def toggle_initial_guess(self):
        self.show_initial_guess = not self.show_initial_guess
        self.display_fit(*self.selected_pixel)

    def display_fit(self, i, j):
        self.text.delete('1.0', 'end')
        self.text.insert('insert', '({:6d}, {:6d})\n'.format(i+1, j+1))
        self.highlight_pixel(i, j)

        items = self.ncomp_listbox.get(0, "end")
        
        q3dos = []
        for indx, q3dii in self.q3di.items():
            try:
                q3do = load_q3dout(q3dii, i+1, j+1)
                q3dos.append(q3do)
            except FileNotFoundError:
                self.ncomp_listbox.delete(indx-1, "end")
                continue
            if (label := f"{indx} Components") not in items:
                self.ncomp_listbox.insert(indx-1, label)
        if len(q3dos) >= self.ncomp:
            q3do = q3dos[self.ncomp-1]
            n = self.ncomp
        else:
            q3do = q3dos[-1]
            n = len(q3dos)

        try:
            pars = plot_single_fit(q3do, self.q3di[n], x=i, y=j, return_pars=True, 
                                   plot_initial_guess=self.show_initial_guess,
                                   **self.plot_opts)

            self.text.insert('insert', pars.to_string())
        except IndexError:
            self.text.insert('insert', 'Index Error!')

        # display SNR values
        self.snr_text.delete('1.0', 'end')
        self.snr_text.insert("insert", "===  SNR: ===\n")
        for line in q3do.linelabel:
            self.snr_text.insert("insert", f"{line}:\n")
            for i in range(q3do.maxncomp):
                snr = q3do.line_fitpars['flux'][line].data[i] / q3do.line_fitpars['fluxerr'][line].data[i]
                self.snr_text.insert("insert", f"Comp {i+1}: {snr:.2f}\n")

        # display AIC values
        if not self.onefile:
            self.aic_text.delete('1.0', 'end')
            self.aic_text.insert("insert", "===  AIC: ===\n")
            for i, q3do in enumerate(q3dos):
                self.aic_text.insert('insert', f"{i+1} Components: {q3do.aic: 7.2f} \n")
                try:
                    self.aic_text.insert('insert', f"        diff: {q3do.aic-q3dos[i+1].aic: 7.2f} \n")
                except IndexError:
                    ...
            self.aic_text.insert("insert", "===  BIC: ===\n")
            for i, q3do in enumerate(q3dos):
                self.aic_text.insert('insert', f"{i+1} Components: {q3do.bic: 7.2f} \n")
                try:
                    self.aic_text.insert('insert', f"        diff: {q3do.bic-q3dos[i+1].bic: 7.2f} \n")
                except IndexError:
                    ...

        


        self.canvas_fit.draw()

    def highlight_pixel(self, x, y):
        """ Outline the Pixel that was clicked on """
        # remove current highlight
        for patch in self.ax_img.patches:
            patch.remove()
        # new highlight
        rect = Rectangle((x-0.5, y-0.5), 1, 1, ec="cyan", fc="none", zorder=10)
        self.ax_img.add_patch(rect)
        self.canvas_img.draw()

    

if __name__ == "__main__":
    window = tk.Tk()
    label = "4C1971_conv"
    label = "4C1971_from_map"
    label = "4C1971_neighbour_method"
    # label = "4C1971_other_lines_Halpha"
    # label = "4C1971_other_lines_OII"
    # label = "4C1971_other_lines_SII"
    # label = "4C1971_other_lines_NeIII"
    Q3DInterface(window, label, onefile=True)
    def on_closing():
        window.quit()
        window.destroy()
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()