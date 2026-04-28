import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas

# astropy
from astropy.io import fits
from astropy import constants

# q3dfit
from q3dfit.q3din import q3din
from q3dfit.q3df import q3dfit
from q3dfit.q3dcollect import q3dcollect
from q3dfit.q3dout import load_q3dout
import q3dfit.q3dpro as q3dpro
from q3dfit import q3dutil

# Bruno
from bcodes import bfit

# constants
c_kms = constants.c.to('km/s')

def plot_single_fit(filename_q3di, x, y, wvel_line=None, rest=True,
                    axis=None, fig=None, clean_axis=False, 
                    simplified=True):

    # ====================================
    # Load
    print(filename_q3di, x, y)
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


    # flux densities
    spec = q3do.spec
    cont_fit = q3do.cont_fit
    line_fit = q3do.line_fit
    model = cont_fit + line_fit
    ct_indx = q3do.ct_indx

    if 0:# TODO from fitspec.py        
        q3do.line_fit = emlmod.eval(lmout.params, x=gdlambda)


    # Initial guess - second fit (result first fit)
    if not simplified:
        ct_indx0 = q3do.ct_indx0
        p0 = np.array([item.value for key, item in q3do.parinit0.items()])
        p0[2::3] = p0[2::3] / c_kms.value  * p0[1::3]#linelist['lines']
        line_init0 = bfit.gaussian(q3do.wave, *p0)

    # Initial guess
    p1 = np.array([item.value for key, item in q3do.parinit.items()])
    p1[2::3] = p1[2::3] / c_kms.value  * p1[1::3]#linelist['lines']
    line_init1 = bfit.gaussian(q3do.wave, *p1)


    # model components
    p = np.array([item for key, item in q3do.param.items()\
                  if 'SPECRES' not in key.upper()])
    specres = np.array([item for key, item in q3do.param.items()\
                        if 'SPECRES' in key.upper()])
    p[2::3] = p[2::3] / c_kms.value * p[1::3]# * linelist['lines']
    pp = p.reshape(len(p)//3, 3)

    model_comps0 = np.array([bfit.gaussian(q3do.wave, *pi) for pi in pp])
    model_comps = np.array([ispecres.spect_convolver(q3do.wave, imodel0, pi[1])\
                   for pi, imodel0, ispecres in zip(pp, model_comps0, specres)])

    # ====================================
    # plots
    #plt.close('all')
    if fig is None:
        fig = plt.figure()
    if axis is None:
        axis = fig.subplots(2, 1)
    if clean_axis:
        print(111111111111111111111)
        axis[0].cla()
        axis[1].cla()
    ax = axis 

    if not simplified:
        ax[0].plot(spectral[ct_indx0], spec[ct_indx0], 'yx', label="cont. data0", ms=10)
    ax[0].plot(spectral[ct_indx], spec[ct_indx], 'r.', label="cont. data", ms=10)
    ax[0].plot(spectral, spec, label="data")
    ax[0].plot(spectral, model, label="model")
    if not simplified:
        ax[0].plot(spectral, cont_fit + line_init0, label="guess0", ls='--')
    ax[0].plot(spectral, cont_fit + line_init1, label="guess1")
    #ax[0].plot(spectral, cont_fit0, color='purple', ls='--', label="continuum0")
    ax[0].plot(spectral, cont_fit, color='purple', label="continuum")
    for imodel, imodel0 in zip(model_comps, model_comps0):
        ax[0].plot(spectral, cont_fit + imodel, ls="--", color='k', lw=1, alpha=.7)

    if not simplified:
        ax[1].plot(spectral[ct_indx0], (spec - model)[ct_indx0], 'yx', ms=10)
    ax[1].plot(spectral[ct_indx], (spec - model)[ct_indx], 'r.', ms=10)
    ax[1].plot(spectral, spec - model)
    #ax[1].plot(spectral, cont_fit0-cont_fit, color='purple', ls='--')
    ax[1].plot(spectral, cont_fit*0, color='purple')
    ax[1].axhline(0, color='gray', ls='-', lw=.5, alpha=.5)

    # limits
    res_cont_max = np.nanmax(abs(spec - model)[q3do.ct_indx])
    res_cont_std = np.nanstd(abs(spec - model)[q3do.ct_indx])

    cont_min = np.nanmin(cont_fit)
    model_max = np.nanmax(model)
    ax[1].set_ylim(-res_cont_max*1.1, +res_cont_max*1.1)
    ax[0].set_ylim(cont_min-res_cont_max*1.1, model_max*1.2)

    ax[0].set_ylim(cont_min-res_cont_std*3, model_max+res_cont_std*3)
    ax[1].set_ylim(-res_cont_std*7, res_cont_std*7)

    # legend
    ax[0].legend(frameon=False, loc='best', fontsize="small")


    ax[0].xaxis.set_ticklabels([])
    ax[1].set_xlabel(xlabel)
    ax[0].set_xlabel('')
    ax[0].set_ylabel("Normalized flux density")
    ax[1].set_ylabel("Residuals")


    plt.tight_layout()
    plt.show()

    # string
    Amp_fit = q3do.line_fitpars['fluxpk'].to_pandas()
    wobs_fit = q3do.line_fitpars['wave'].to_pandas()
    vel_fit = (wobs_fit/(1+q3do.zstar)/linelist['lines']-1) * c_kms
    sig_fit = q3do.line_fitpars['sigma'].to_pandas()

    Amp_fit.index =  [f'amp_{i}' for i in range(len(Amp_fit))]
    vel_fit.index =  [f'vel_{i}' for i in range(len(Amp_fit))]
    sig_fit.index =  [f'sig_{i}' for i in range(len(Amp_fit))]
    pars = pandas.concat([Amp_fit, vel_fit, sig_fit], ignore_index=False)

    #print(pars.to_string(float_format='%g'))
    print(pars)




# ==========================================
from bcube import plots, utils, ctools, cplots

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure


def evalNone(x):
    try:
        return eval(x)
    except:
        return None

def iter(filename_q3di, img, plotmap_opts1={}, small_screen=False,
             cube=None, cube2=None, cube_comp=None, cube3=None, cube4=None, cube5=None, wl=None):
             

    class tkclass:

        def __init__(self, window, img, cube, cube2, cube_comp):

            # Data
            if 1:# delete
                self.spectral_axis = wl
                self.img = img
                self.cube = cube
                self.cube2 = cube2
                self.cube_comp = cube_comp
                self.cube3 = cube3
                self.cube4 = cube4
                self.cube5 = cube5
            
            self.filename_q3di = filename_q3di

            # Initial options
            #self.plot_opts = {'xlim':None, 'ylim':None,
            #                  'spectral_axis':self.spectral_axis}

            self.plot_opts = {'clean_axis':True}
            self.plotmap_opts = {'vmin':None, 'vmax':None, 'cmap':'inferno',
                                 'doLog':False, 'do_colorbar':False,
                                 'do_contour':True}
            self.plotmap_opts.update(plotmap_opts1)

            # TK
            self.window = window
            self.buttonPlot = tk.Button(window, text='Image plot', command=self.plot)
            self.buttonFollow = tk.Button(window, text='Follow mouse', command=self.follow)
            
            self.buttonSinglePlot = tk.Button(window, text='Plot on click', command=self.singleplot)
            self.text = tk.Text(window, bg='white', height=10, width=80, font=('Fixedsys', 12))

            # -----------------------
            if small_screen:
                self.fig_img = Figure(figsize=(3, 3))
                self.fig_fit = Figure(figsize=(6, 3))
            else:
                self.fig_img = Figure(figsize=(6, 6))
                self.fig_fit = Figure(figsize=(12, 6))


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
            #
            self.buttonSpecProperties = tk.Button(window,
                         text='Spectrum options', command=self.input_spec_opts)
            self.buttonSpecProperties.grid(row=8, column=1)
            #
            self.buttonMapProperties = tk.Button(window,
                         text='Map options', command=self.input_map_opts)
            self.buttonMapProperties.grid(row=9, column=1)            

            self.canvas = canvas
            self.canvas2 = canvas2
            canvas.draw()

            # update dicts
            self.plotmap_opts.update(fig=self.fig_img, axis=self.ax1)
            #self.plot_opts.update(fig=self.fig_fit, axis=self.ax2)
            self.plot_opts.update(fig=self.fig_fit, axis=self.ax2)

        def input_spec_opts(self):
            entry_list = ['xlim', 'ylim', 'radius']
            output_dict = self.plot_opts

            self.create_input_box(entry_list, output_dict)

        def input_map_opts(self):
            entry_list = ['vmin', 'vmax', 'doLog', 'do_colorbar', 'do_contour']
            output_dict = self.plotmap_opts

            self.create_input_box(entry_list, output_dict)

        def input_radius(self):
            entry_list = ['radius']
            output_dict = self.radius
            self.create_input_box(entry_list, output_dict)
            #print()

        def create_input_box(self, entry_list, output_dict):
            window_spec = tk.Toplevel(self.window)
            self.tkentries = {}

            for row, entrytext in enumerate(entry_list): 
                self.add_entry(window_spec, entrytext, row)

            self.input_window = window_spec
            self.input_entry_list = entry_list
            self.input_output_dict = output_dict
            
            ButtomOK = tk.Button(window_spec, text="OK", command=self.get_values)
            ButtomOK.grid(row=row+1, column=0)

        def add_entry(self, window, text, row):
            E = tk.Entry(window)
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
                entry_values[entry] = evalNone(value)

            output_dict.update(entry_values) 


        def plot(self, contrast=1):

            #self.plot_opts['axis'].cla()
            grid_dict = plots.plot_quantity_map(self.img,# axis=axis, fig=fig,
                                                **self.plotmap_opts)

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

                #self.plot_opts['axis'][0].cla()
                #self.plot_opts['axis'][1].cla()


                '''
                cplots.plot_spectrum_cube(self.cube, x=i, y=j,
                                          **self.plot_opts)

                if self.cube2 is not None:
                    cplots.plot_spectrum_cube(self.cube2, x=i, y=j,
                                          **self.plot_opts)

                if self.cube3 is not None:
                    cplots.plot_spectrum_cube(self.cube3, x=i, y=j,
                                          **self.plot_opts)

                if self.cube4 is not None:
                    cplots.plot_spectrum_cube(self.cube4, x=i, y=j,
                                          **self.plot_opts)

                if self.cube5 is not None:
                    cplots.plot_spectrum_cube(self.cube5, x=i, y=j,
                                          **self.plot_opts)
                                                                                    
                if self.cube_comp is not None:
                    for cube_i in self.cube_comp:
                        cplots.plot_spectrum_cube(cube_i, x=i, y=j, 
                                                  alpha=.6, ls='--', plottype='plot',
                                                  **self.plot_opts)
                '''          

                plot_single_fit(self.filename_q3di, x=i, y=j, **self.plot_opts)
            
                plt.show()
                #self.text.insert('insert', s[j, i])
            except IndexError:
                self.text.insert('insert', 'Index Error!')

            self.canvas2.draw()

    app_window = tk.Tk()
    start = tkclass(app_window, img, cube, cube2, cube_comp)
    app_window.mainloop()









