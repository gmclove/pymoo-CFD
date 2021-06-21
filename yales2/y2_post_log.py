#!/usr/bin/env python3

import warnings
warnings.simplefilter("ignore")

import sys,os
import time as timer
import datetime
import glob
import os.path
import numpy as np
import matplotlib
import matplotlib.ticker as mticker
#matplotlib.use('pdf')
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import matplotlib.pyplot as plt
from tabulate import tabulate

# parameters
adapt_line_color = '#E6E6E6'

# definitions
plot_nite = []
plot_skewness = []
plot_metric_error = []
plot_hgrad = []
plot_target_skewness = []
plot_target_metric_error = []
plot_target_hgrad = []
count_skew_inter = []      # Total number of bad cells at the elgrp interface - all runs
count_skew_inner = []      # Total number of bad cells inside the elgrp - all runs
count_skew_total = []      # Total number of bad cells - all runs
interp_time = []           # Time spent for interpolation in the adaptation - all runs
mmg_time = []              # Time spent for mmg in the adaptation - all runs
coloring_time = []         # Time spent for the coloration in the adapatation - all runs
transfer_time = []         # Time spent for the transfer in the adaptation - all runs
t_global_ada = []          # Total time spent in the adaptation loop - all runs
total_nelem = []           # Total number of elements in mesh - all runs
min_nelem_full = []        # Min number of elements per core including adaptation substeps - all runs
max_nelem_full = []        # Max number of elements per core including adaptation substeps - all runs
mean_nelem_full = []       # Mean number of elements per core including adaptation substeps - all runs
min_nelem = []             # Min number of elements per core - all runs
max_nelem = []             # Max number of elements per core - all runs
mean_nelem = []            # Mean number of elements per core - all runs
adapt_iter = []            # Iteration index at adaptation
adapt_iter2 = []           # Number of adaptation step
umax = []                  # Max velocity - all runs
L2mean = []                # Mean L2 norm of velocity - all runs
drhodt_max = []            # Max DRHODT - all runs (VDS only)
mu_t_max = []              # Max MU_T - all runs (VDS only)
mu_artif_max = []          # Max MU_ARTIF - all runs (VDS only)
Tmax = []                  # Max temperature - all runs (VDS only)
volphi = []                # Volume LS_PHI - all runs
dxmin_pairbased = []       # Minimum cell size (pair length based) - all runs
dxmin_nodevol = []         # Minimum cell size (node volume based) - all runs
ites = []                  # Index of the iteration - all run
iadapt = []                # Index of the adaptation loop - all runs
adaptsteps = []            # Number of adaptation step in the adaptation loop i - all runs
dt = []                    # Time step - all runs
dt_cfl = []                # Convective time step - all runs
dt_visc = []               # Viscous time step - all runs
dt_st = []                 # Surface tension time step - all runs
time = []                  # Total time - all runs
local_time = []            # Time - all runs
time_adapt = []            # Total time at adaptation - all runs
total_mem = []             # Total memory use - all runs
adapt_step = []            # Index of the adaptation step - all runs
adapt_step.append(0)
tot_ites = []
tot_ites.append(0)
tot_adapt_iter = []
tot_adapt_iter.append(0)
tot_adapt_step = []
tot_adapt_step.append(0)
metric_interface_value = []
nbcores = []
final_time = []
final_local_time = []
final_solver_time = []
reset_stats = []
cases = []
wait3 = 0
n_adapt = 0
NPROPAG = 0
CFL     = 0
ncores  = 0
METRIC_INTERFACE_VALUE = float('nan')
new_adapt = False
solver_type = 'NONE'
git_branch = ''
git_commit = ''
git_date = ''
grid_adaptation = False
iplot = 0                     # plot number iterator
single_width = 16/3           # width of a single plot
single_height = 12/4          # height of a single plot
ncols = 3                     # number of columns of plots
nrows = 4                     # number of rows of plots
fsize=(16,12)                 # figsize (width, height)
accumulated_time = []         # accumulated stats time - all runs
reinit_stats_time_ifile = 0   # catch time when stats are reinit during a run
perf_time_dict = dict()       # Solver time - all runs
perf_rct_dict = dict()        # RCT (global) of all items - all runs
perf_itrct_dict = dict()      # RCT (local) of all items - all runs
loaded = False

#####################
# Function plot log
#####################
def plot_log(arg_cases,output=".",use_tex=False,save_pdf=True,show_figs=False,send_by_mail=False,fig_size=(16,12),fontsize=10,markersize=5):
   # Use global variables
   global plot_nite,plot_metric_error,plot_skewness,plot_hgrad,count_skew_inter,count_skew_inner,count_skew_total,mmg_time,interp_time,coloring_time,transfer_time,t_global_ada
   global total_nelem,min_nelem,max_nelem,mean_nelem,adapt_iter,adapt_iter2,plot_target_metric_error,plot_target_skewness,plot_target_hgrad
   global umax,volphi,dxmin_pairbased,dxmin_nodevol,ites,iadapt,adaptsteps,dt,dt_cfl,dt_visc,dt_st,time,local_time,time_adapt,adapt_step,adapt_step
   global tot_ites,tot_adapt_iter,tot_adapt_step,nbcores,final_time,final_local_time,final_solver_time
   global n_adapt,NPROPAG,CFL,ncores,new_adapt
   global fsize,cases,solver_type,grid_adaptation
   global iplot, ncols, nrows, loaded
   global perf_time_dict, perf_rct_dict, perf_itrct_dict
   global accumulated_time, reset_stats, reinit_stats_time_ifile, METRIC_INTERFACE_VALUE, metric_interface_value
   global git_branch, git_commit, git_date
   global min_nelem_full,max_nelem_full,mean_nelem_full

   # Timer
   tic = timer.perf_counter()

   if loaded:
       iplot = 0
   else:
      # Plot options
      rc('text', usetex=use_tex)
      rc('font', family='serif')
      rc('lines', markersize=markersize)
      rc('font', size=fontsize)
      plt.rc('legend',**{'fontsize':fontsize})

      # Cases filename 
      for arg_case in arg_cases:
         if os.path.isdir(arg_case):
            filelist = sorted(glob.glob(arg_case+'/solver01_rank*.log'))
            if len(filelist)>0:
               cases.append(filelist[0])
            #filelist = sorted(glob.glob(case+'/logfile_*.o'))
            #if len(filelist)>0:
            #   my_proc0_files = my_proc0_files+filelist[::3]
         elif os.path.isfile(arg_case):
            cases.append(arg_case)
         else:
            print("Couldn't find folder or file "+arg_case)
            sys.exit()

      if not cases:
         print("Couldn't find cases")
         sys.exit()

      # Check if the cases ran at least one ite
      sorted_cases = []
      for icase,case in enumerate(cases):
         with open(case) as f:
            mylines = f.readlines()
         started = False
         niter = 0 # to avoid error when there is only one iteration
         for aline in mylines:
            if "Begin iter" in aline:
               niter += 1
            if niter >= 2:
               started = True
               break
         
         if started:
            sorted_cases.append(case)
         else:
            print("> Skipping "+case+" (did not run)")
      cases = sorted_cases
      
      print(" ")
      for icase,case in enumerate(cases):

         print("> Loading "+case)
         with open(case) as f:
            mylines = f.readlines()

         wait3 = -1
         n_adapt = 0

         NPROPAG = 5
         CFL     = 0.5
         ncores  = -1
         METRIC_INTERFACE_VALUE = float('nan')
         new_adapt = False
         perf_adapt = False
         reset_stats_ifile = False
         accumulated_time_ifile = float('nan')

         for aline in mylines:

            splitted = [word for word in aline.replace('\n','').split(" ") if not word=='']

            # Solver selection
            if "SOLVER_TYPE = " in aline:
               if "SOLVER_TYPE = INCOMPRESSIBLE" in aline:
                  solver_type = 'ICS'
                  nrows = 4
               elif "SOLVER_TYPE = VARIABLEDENSITY" in aline:
                  solver_type = 'VDS'
                  nrows = 5
                  figsize_height = nrows*single_height
                  fig_size = (fig_size[0],figsize_height)
               elif "SOLVER_TYPE = SPRAY" in aline:
                  solver_type = 'SPS'
                  nrows = 5
                  figsize_height = nrows*single_height
                  fig_size = (fig_size[0],figsize_height)
               elif "SOLVER_TYPE = LEVELSET" in aline:
                  solver_type = 'LSS'
                  nrows = 4

            # get reset stats status
            if "RESET_ALL_STATS = TRUE" in aline:
               reset_stats_ifile = True
               accumulated_time_ifile = 0

            # Grid adaptation detection
            #if "ADAPTATION" in aline:
            #   grid_adaptation = True

            if "CFL =" in aline and not "ACOUSTIC" in aline:
               CFL = float(splitted[-1])

            # get GIT INFORMATION
            if "> GIT BRANCH" in aline:
               git_branch = splitted[-1]
            if "> GIT COMMIT" in aline:
               git_commit = splitted[-1]
            if "> GIT DATE" in aline:
               splitted_date = [word for word in aline.replace('\n','').split("DATE") if not word=='']
               git_date = splitted_date[-1]

            # get accumulated time of the run
            if "Accumulated time for stats of " in aline:
               splitted1 = [word.strip() for word in aline.replace('\n','').split("=") if not word=='']
               if np.isnan(accumulated_time_ifile):
                  if reset_stats_ifile:
                     accumulated_time_ifile = 0
                  else:
                     accumulated_time_ifile = float(splitted1[-1])
      
            # get reset stats during run (it can occur during adaptation)
            if "> Reset data statistics" in aline:
               accumulated_time_ifile = 0
               if local_time:
                  reinit_stats_time_ifile = local_time[-1]
               
            if "SCALAR METRIC_INTERFACE VALUE =" in aline:
               METRIC_INTERFACE_VALUE = float(splitted[-1])

            if "Begin iter" in aline:
               ite = int(splitted[-1])
               ites.append(ite+tot_ites[icase])
               umax.append(float('nan'))
               L2mean.append(float('nan'))
               Tmax.append(float('nan'))
               drhodt_max.append(float('nan'))
               mu_t_max.append(float('nan'))
               mu_artif_max.append(float('nan'))
               volphi.append(float('nan'))
               dxmin_pairbased.append(float('nan'))
               dxmin_nodevol.append(float('nan'))
               dt_cfl.append(float('nan'))
               dt_visc.append(float('nan'))
               dt_st.append(float('nan'))
               dt.append(float('nan'))
               time.append(float('nan'))
               local_time.append(float('nan'))
               metric_interface_value.append(METRIC_INTERFACE_VALUE)
               perf_adapt = False
      
            if "  Initialize adaptation" in aline:
               new_adapt = True
               perf_adapt = True
               n_adapt = n_adapt+1
               adapt_iter.append(ite+tot_ites[icase]) 
               plot_nite.append([])
               plot_skewness.append([])
               plot_metric_error.append([])
               plot_hgrad.append([])
               plot_target_skewness.append([])
               plot_target_metric_error.append([])
               plot_target_hgrad.append([])
         
            if len(splitted)>9:
               if "> TEMPORAL LOOP            |" in aline:
                  add_value_to_dict(perf_time_dict, ite+tot_ites[icase], 'TEMPORAL LOOP', float(splitted[-10]))
                  add_value_to_dict(perf_rct_dict, ite+tot_ites[icase], 'TEMPORAL LOOP', float(splitted[-5]))
                  if perf_adapt == True:
                     add_value_to_dict(perf_itrct_dict, ite+tot_ites[icase], 'TEMPORAL LOOP', float(splitted[-3]))
               if "----> |" in aline:
                  perf_name = aline.split('-')[0].split('> ')[1].strip()
                  add_value_to_dict(perf_rct_dict, ite+tot_ites[icase], perf_name, float(splitted[-5]))
                  if perf_adapt == True:
                     add_value_to_dict(perf_itrct_dict, ite+tot_ites[icase], perf_name, float(splitted[-3]))
               if "> ADAPTATION STEP " in aline:
                  adaptation_step = aline.split(' | ')[0].split('> ')[1].strip()
                  add_value_to_dict(perf_rct_dict, ite+tot_ites[icase], adaptation_step, float(splitted[-5]))
                  if  perf_adapt == True:
                     add_value_to_dict(perf_itrct_dict, ite+tot_ites[icase], adaptation_step, float(splitted[-3]))
               # Grid adaptation detection
               if "> GRID ADAPTATION" in aline:
                  grid_adaptation = True
                     
            if "Maximum norm of U =" in aline:
               if len(umax)>0 and splitted[-1].replace('.','',1).isdigit(): umax[-1] = float(splitted[-1])
            if "L2 norm mean of U =" in aline:
               if len(L2mean)>0: L2mean[-1] = float(splitted[-1])
            if "Maximum norm of T =" in aline:
               if len(Tmax)>0: Tmax[-1] = float(splitted[-1])
            if "Maximum norm of DRHODT =" in aline:
               if len(drhodt_max)>0: drhodt_max[-1] = float(splitted[-1])
            if "Maximum norm of MU_T =" in aline:
               if len(mu_t_max)>0: mu_t_max[-1] = float(splitted[-1])
            if "Maximum norm of MU_ARTIF =" in aline:
               if len(mu_artif_max)>0: mu_artif_max[-1] = float(splitted[-1])
            if "Minimum cell size (nodevol)" in aline and "mm" in aline:
               if len(dxmin_nodevol)>0: dxmin_nodevol[-1] = float(splitted[-2])
            if "Minimum cell size (pair-based)" in aline and "mm" in aline:
               if len(dxmin_pairbased)>0: dxmin_pairbased[-1] = float(splitted[-2])
            if "Volume integral of LS_PHI" in aline:
               if len(volphi)>0: volphi[-1] = float(splitted[-1])
            if "> Running YALES2 in" in aline:
               ncores = int(splitted[-2])
      
            if "Adaptation iteration =" in aline:
               nite = int(aline.replace('\n','').split(' ')[-1])
               plot_nite[-1].append(nite)
               adapt_step.append(adapt_step[-1]+1)
               if nite==1:
                  adapt_iter2.append(len(count_skew_inter))
               count_skew_inter.append(0)
               count_skew_inner.append(0)
               count_skew_total.append(0)
               min_nelem_full.append(last_min_nelem)
               max_nelem_full.append(last_max_nelem)
               mean_nelem_full.append(last_mean_nelem)
               interp_time.append(0.0)
               mmg_time.append(0.0)
               coloring_time.append(0.0)
               transfer_time.append(0.0)
               plot_hgrad[-1].append(float('nan'))
               plot_target_hgrad[-1].append(float('nan'))
               plot_metric_error[-1].append(float('nan'))
               plot_target_metric_error[-1].append(float('nan'))

            if ">  " in aline and "  dt_conv" in aline:
               dt_cfl[-1]=float(splitted[-1])
               if ite==1: dt_cfl[-1]=float('nan')
            elif ">  " in aline and "  dt_visc" in aline:
               dt_visc[-1]=float(splitted[-1])
            elif ">  dt_surf_tens" in aline:
               dt_st[-1]=float(splitted[-1])
            elif ">         dt" in aline or ">            dt" in aline:
               dt[-1]=float(splitted[-1])
            elif "> " in aline and " total time" in aline and not "[" in aline:
               time[-1]=float(splitted[-1])
            elif ">  " in aline and " time" in aline and not "[" in aline and not "total" in aline:
               local_time[-1]=float(splitted[-1])

            if "Current max skewness     =" in aline and new_adapt:
               skew = float(aline.replace('\n','').split(' ')[-1])
               plot_skewness[-1].append(skew)

            if "Current max metric error =" in aline and new_adapt:
               metric_error = float(aline.replace('\n','').split(' ')[-2])
               if len(plot_metric_error[-1])>0: plot_metric_error[-1][-1] = metric_error

            if "Current max hgrad        =" in aline and new_adapt:
               max_hgrad = float(aline.replace('\n','').split(' ')[-1])
               if len(plot_hgrad[-1])>0: plot_hgrad[-1][-1] = max_hgrad
      
            if "Desired max skewness in all adapted regions =" in aline and new_adapt:
               target_skew = float(aline.replace('\n','').split(' ')[-1])
               plot_target_skewness[-1].append(target_skew)

            if "Desired max metric error =" in aline and new_adapt:
               target_metric_error = float(aline.replace('\n','').split(' ')[-2])
               if len(plot_target_metric_error[-1])>0: plot_target_metric_error[-1][-1] = target_metric_error

            if "Desired max hgrad        =" in aline and new_adapt:
               target_max_hgrad = float(aline.replace('\n','').split(' ')[-1])
               if len(plot_target_hgrad[-1])>0: plot_target_hgrad[-1][-1] = target_max_hgrad

            if ": interface/el_grp " in aline:
               count_skew_total[-1]=count_skew_total[-1]+1
               if "=" in aline:
                  count_skew_inter[-1]=count_skew_inter[-1]+1
               else:
                  count_skew_inner[-1]=count_skew_inner[-1]+1
         
            splitted_line = splitted

            if "nelem        :" in aline:
               last_total_nelem = int(splitted_line[-1])
               last_min_nelem   = int(splitted_line[-4])
               last_max_nelem   = int(splitted_line[-3])
               last_mean_nelem  = int(splitted_line[-2])
      
            if "   > Interpolation OK" in aline and len(interp_time)>0:
               interp_time[-1] = interp_time[-1] + float(splitted_line[-2])
            if "   > Coloring OK" in aline and len(coloring_time)>0:
               coloring_time[-1] = coloring_time[-1] + float(splitted_line[-2])
            if "   > Transfering el_grps OK" in aline and len(transfer_time)>0:
               transfer_time[-1] = transfer_time[-1] + float(splitted_line[-2])
            if "   > MMG3D mesh adaptation OK" in aline and len(mmg_time)>0:
               mmg_time[-1] = mmg_time[-1] + float(splitted_line[-2])
        
            if "memory (MB):" in aline and n_adapt>0:
               total_mem.append(float(splitted_line[-1]))
 
            if "> Timer ADAPT_GRID" in aline:
               new_adapt = False
               t_global_ada.append(float(aline.replace('\n','').split(' ')[-1]))
               time_adapt.append(time[-1])
               total_nelem.append(last_total_nelem)
               min_nelem.append(last_min_nelem)
               max_nelem.append(last_max_nelem)
               mean_nelem.append(last_mean_nelem)
               iadapt.append(int(n_adapt)+tot_adapt_iter[icase])
               adaptsteps.append(len(plot_nite[-1]))
         
         tot_ites.append(ite+tot_ites[icase])
         tot_adapt_iter.append(int(n_adapt)+tot_adapt_iter[icase])
         if tot_adapt_step and adaptsteps:
            tot_adapt_step.append(adapt_iter2[-1]+adaptsteps[-1]-1)
         else:
            tot_adapt_step.append(0)
         nbcores.append(ncores)
         reset_stats.append(reset_stats_ifile)
         accumulated_time.append(accumulated_time_ifile + local_time[-1] - reinit_stats_time_ifile)
         final_time.append(time[-1])
         final_local_time.append(local_time[-1])
         try:
            final_solver_time.append(perf_time_dict['TEMPORAL LOOP']['value'][-1])
         except:
            final_solver_time.append("nan")

         if solver_type == 'NONE':
            print("    Solver: Unknown --> ICS used")
            solver_type = 'ICS'
         else:
            print("    Solver: ",solver_type)
            print("    Grid Adaptation: ",grid_adaptation)

      loaded = True

   # Timer 
   toc = timer.perf_counter()
   print(f"  Loaded the logfile(s) in {toc - tic:0.4f} seconds")

   # Plot the main global figure
   fsize = fig_size
   figs = []
   if len(cases)>1:
      print("> Plotting global figure")
      fig1 = plot_one_fig(None,use_tex,save_pdf,show_figs,send_by_mail,fig_size,fontsize,markersize)
      figs.append(fig1)

   # Loop over file to get local plots
   for ifile, file in enumerate(cases):
      print("> Plotting local figure ",ifile,file)
      iplot = 0 # reset the iplot counter for each file
      fig1 = plot_one_fig(ifile,use_tex,save_pdf,show_figs,send_by_mail,fig_size,fontsize,markersize)
      figs.append(fig1)

   # Timer
   tic = timer.perf_counter()
   print(f"  Plotted the logfile(s) in {tic - toc:0.4f} seconds")

   # Save the PDF file and show the figures
   output_pdf = None
   if save_pdf and len(cases)>0:
      #print("> Saving figures")
      if not os.path.exists(output):
         os.makedirs(output)
      output_pdf = output+"/post.pdf"

      with PdfPages('%s'% output_pdf) as pdf:
         if isinstance(figs, list):
            for fig in figs:
               pdf.savefig(fig, bbox_inches="tight")

         d = pdf.infodict()
         d['Title'] = 'post-processing'
         d['Author'] = os.environ['USER']
         d['Subject'] = 'Multipage pdf file containing automated post-processing'
         d['ModDate'] = datetime.datetime.today()

      # Timer
      toc = timer.perf_counter()
      print(f"  Saved the logfile(s) in {toc - tic:0.4f} seconds")

   if show_figs: plt.show()

   # Sending results by mail
   if send_by_mail:
      import y2_hosttype as host
      if host.can_mail: host.mail("Resultats run","",[output_pdf])

   return figs

#############################################################
def add_value_to_dict(performance_dict, iteration, name, value):
    performance_dict = eval('performance_dict')
    if not name in performance_dict:
        performance_dict[name] = dict(iteration=list(), value=list())
        
    performance_dict[name]['iteration'].append(iteration)
    performance_dict[name]['value'].append(value)

#############################################################
# Plot main figure only if more than one file
def plot_one_fig(ifile,use_tex,save_pdf,show_figs,send_by_mail,fig_size,fontsize=10,markersize=5):
   global iplot,nrows,ncols

   # Clear previous figures
   plt.close()

   # Plot figures
   fig = plt.figure(1,figsize=fsize,dpi=200)
   fig.tight_layout()
   fig.set_tight_layout(True)
   fig.canvas.draw()

   # Compute subplot matrix size
   #if fig_size[0]>fig_size[1]:
   #    if nrows > ncols:
   #        ncols_old = ncols
   #        ncols = nrows
   #        nrows = ncols_old
   #else:
   #    if nrows < ncols:
   ##        ncols_old = ncols
   #        ncols = nrows
   #        nrows = ncols_old

   # plot baseline 
   plot_baseline_info(fig,ifile,fontsize)

   # plot solver-specific info
   if solver_type == 'ICS':
      plot_ICS_info(fig,ifile)
   elif solver_type == 'SPS':
      plot_SPS_info(fig,ifile)
   elif solver_type == 'LSS':
      plot_LSS_info(fig,ifile)
   elif solver_type == 'VDS':
      plot_VDS_info(fig,ifile)

   if grid_adaptation:
      plot_adaptation(fig,ifile,fontsize)

   if ifile == None: # global plot
      # Run information
      if iplot <= (ncols*nrows)-2:
         plot_run_information = 'right'
         iplot += 1
      elif iplot == ncols*nrows:
         x_txt = 0.5
         plot_run_information = 'bottom'
         horizontalalignment = 'right'
      else:
         x_txt = -0.2
         plot_run_information = 'bottom'
         horizontalalignment = 'left' #default value
      ax12 = fig.add_subplot(nrows,ncols,iplot)
      fig.set_tight_layout(False)
      try:
         header = ['', 'ncores', 'total time (ms)', 'time (ms)', 'stats time (ms)', 'solver time (h)', 'Cost (kCPUh)']
         header_format = (None, 'd', '.2f', '.2f','.2f','.2f','.2f')
         data = []
         for ifile,file in enumerate(cases):
            data.append(['{:s}'.format('RUN'+str(ifile+1)),
                         '{:d}'.format(nbcores[ifile]),
                         '{:.2f}'.format(final_time[ifile]*1000),
                         '{:.2f}'.format(final_local_time[ifile]*1000),
                         '{:.2f}'.format(accumulated_time[ifile]*1000),
                         '{:.2f}'.format(final_solver_time[ifile]/3600.0),
                         '{:.2f}'.format(nbcores[ifile] * final_solver_time[ifile]/3600.0 /1000.0),
            ])
         str_table = tabulate(data, headers=header, tablefmt='psql', floatfmt=header_format)

         fontdict_mono = {'family' : 'monospace'}
         if plot_run_information == 'right':
            ax12.text(-0.2, 0.95, str_table, transform=ax12.transAxes, fontsize=fontsize,
                  verticalalignment='top', fontdict=fontdict_mono)
            plt.axis('off')         
         else:
            ax12.text(x_txt, -0.30, str_table, transform=ax12.transAxes, fontsize=fontsize,
                     verticalalignment='top', horizontalalignment=horizontalalignment, fontdict=fontdict_mono)
      except:
         for ifile,file in enumerate(cases):
            textstr = ' '.join((
                r'$\mathrm{RUN} %01d$' % (ifile+1, ),
                r'$\mathrm{NCORES}=%d$' % (nbcores[ifile], )))
            if plot_run_information == 'right':
               ax12.text(0.0, 0.95-ifile*0.10, textstr, transform=ax12.transAxes, fontsize=fontsize,
                     verticalalignment='top')
               plt.axis('off')
            else:
               ax12.text(0.0, -0.30-ifile*0.10, textstr, transform=ax12.transAxes, fontsize=fontsize,
                     verticalalignment='top')
            # plt.axis('off')
            # suptitle = '\n'.join([suptitle, textstr])
      fig.tight_layout()
      fig.subplots_adjust(top=0.92)
      fig.subplots_adjust(wspace=0.4)
      
      #if save_pdf: pdf.savefig(bbox_inches="tight")
      if show_figs: plt.show()

   else: # local plot
      date = str(datetime.datetime.now().replace(microsecond=0))
      full_path = os.path.abspath(cases[ifile])
      item_path = full_path.split('_')
      full_path_latex=""
      fig.set_tight_layout(False)
      for i,item in enumerate(item_path):
         if i==0:
            full_path_latex=full_path_latex+item
         else:
            full_path_latex=full_path_latex+"\_"+item
      if use_tex:
         try:
            plt.suptitle(full_path_latex+"\n"+date+" | NCORES="+str(nbcores[ifile])+ \
                                                   " | total time={:.3f}".format(final_time[ifile]*1000)+" ms"+ \
                                                   " | stats time={:.3f}".format(accumulated_time[ifile]*1000)+" ms"+ \
                                                   " | solver time={:.2f}".format(final_solver_time[ifile]/3600.0)+" h"+ \
                                                   " | cost={:.2f}".format(nbcores[ifile] * final_solver_time[ifile]/3600.0/1000.0)+" kCPUh"
                                                   +"\n"+"Git branch "+str(git_branch) + \
                                                   " | Git Commit "+str(git_commit[:9]) + \
                                                   " | Git Date" +str(git_date))
         except:
            plt.suptitle(full_path_latex+"\n"+date+" | NCORES="+str(nbcores[ifile]))
      else:
         try:
            plt.suptitle(full_path+"\n"+date+" | NCORES="+str(nbcores[ifile])+ \
                                             " | total time={:.3f}".format(final_time[ifile]*1000)+" ms"+ \
                                             " | stats time={:.3f}".format(accumulated_time[ifile]*1000)+" ms"+ \
                                             " | solver time={:.2f}".format(final_solver_time[ifile]/3600.0)+" h"+ \
                                             " | cost={:.2f}".format(nbcores[ifile] * final_solver_time[ifile]/3600.0/1000.0)+" kCPUh"
                                            +"\n"+"Git Branch "+str(git_branch) + \
                                             " | Git Commit "+str(git_commit[:9]) + \
                                             " | Git Date" +str(git_date))
         except:
            plt.suptitle(full_path+"\n"+date+" | NCORES="+str(nbcores[ifile]))
      
      fig.tight_layout() 
      fig.subplots_adjust(top=0.92)
      #if save_pdf: pdf.savefig(bbox_inches="tight")
      if show_figs: plt.show()

   return fig

################################################################################
def plot_baseline_info(fig,ifile=None,fontsize=10):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # ---------------
   # Time steps
   iplot += 1
   #print(nrows,ncols,iplot)
   ax11 = fig.add_subplot(nrows,ncols,iplot)
   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1
   
   ax11.plot(ites[imin:imax],np.array(dt[imin:imax])*1e06,label="$\Delta$t")
   ax11.plot(ites[imin:imax],np.array(dt_cfl[imin:imax])*1e06,label="$\Delta$t cfl")
   ax11.plot(ites[imin:imax],np.array(dt_visc[imin:imax])*1e06,label="$\Delta$t visc")
   ax11.plot(ites[imin:imax],np.array(dt_st[imin:imax])*1e06,label="$\Delta$t st")
   ax11.set_xlabel('Iter nb. [-]')
   ax11.set_ylabel('$\Delta$t [$\mu$s]')
   ax11.set_yscale('log')
   ax11.grid(axis='y', which='both', linewidth=0.5, linestyle='--', color=adapt_line_color)
   ax11.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10, 2), numticks=100))
   ax11.yaxis.set_minor_formatter(MathTextSciFormatter("%1.1e"))
   ax11.tick_params(axis = 'y', which = 'minor', labelsize = 0.8*fontsize)

   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

   # ---------------
   # RCT 
   iplot += 1   
   ax6 = fig.add_subplot(nrows,ncols,iplot)

   # get the top 3 of rct perf.
   top_val_rct = list()
   for i in range(0,3):
      val = dict()
      for key, value in perf_rct_dict.items():
         if key != 'TEMPORAL LOOP' and not 'ADAPTATION STEP' in key and not key in top_val_rct:
               val[key] = np.mean(value['value'][imin:imax])
      key_max = max(val.keys(), key=(lambda k: val[k]))
      top_val_rct.append(key_max)

   for adapt_it in adapt_iter[ibeg:iend]:
      ax6.plot([adapt_it,adapt_it],[0,10000],linewidth=0.5,linestyle='--',color=adapt_line_color)
   try:
      maxmax = max(np.nanmax(np.array(perf_rct_dict['TEMPORAL LOOP']['value'][imin:imax])),
                   np.nanmax(np.array(perf_rct_dict[top_val_rct[0]]['value'][imin:imax])),
                   np.nanmax(np.array(perf_rct_dict[top_val_rct[1]]['value'][imin:imax])),
                   np.nanmax(np.array(perf_rct_dict[top_val_rct[2]]['value'][imin:imax])))
      maxmax = maxmax * 1.1
   except:
      maxmax = 3000
   if np.isnan(maxmax):
      maxmax = 3000

   temp_loop_label  = "Temporal"
   top1_label    = top_val_rct[0].capitalize()
   top2_label    = top_val_rct[1].capitalize()
   top3_label    = top_val_rct[2].capitalize()

   if maxmax>3000:
      ymax = np.mean(np.array(perf_rct_dict['TEMPORAL LOOP']['value'][imin:imax])) * 2
      if np.isnan(ymax): 
         ymax = maxmax 

      if ymax<=maxmax:
         temp_loop_label = temp_loop_label + " (max="+str(np.nanmax(np.array(perf_rct_dict['TEMPORAL LOOP']['value'][imin:imax])))+")"
         top1_label   = top1_label   + " (max="+str(np.nanmax(np.array(perf_rct_dict[top_val_rct[0]]['value'][imin:imax])))+")"
         top2_label= top2_label  + " (max="+str(np.nanmax(np.array(perf_rct_dict[top_val_rct[1]]['value'][imin:imax])))+")"
         top3_label= top3_label  + " (max="+str(np.nanmax(np.array(perf_rct_dict[top_val_rct[2]]['value'][imin:imax])))+")"
   else:
      ymax = maxmax
   
   x_temporal = perf_rct_dict['TEMPORAL LOOP']['iteration'][imin:imax]
   y_temporal = perf_rct_dict['TEMPORAL LOOP']['value'][imin:imax]
   if x_temporal: ax6.plot(x_temporal, y_temporal, label=temp_loop_label)
   if x_temporal: ax6.scatter([max(x_temporal)], [y_temporal[-1]], s=40, clip_on=False, linewidth=0)
   if x_temporal: ax6.annotate(y_temporal[-1], xy=[max(x_temporal), y_temporal[-1]], xytext=[20, 2], textcoords='offset points')

   x_top1 = perf_rct_dict[top_val_rct[0]]['iteration'][imin:imax]
   y_top1 = perf_rct_dict[top_val_rct[0]]['value'][imin:imax]   
   if x_top1: ax6.plot(x_top1, y_top1,label=top1_label)
   if x_top1: ax6.scatter([max(x_top1)], [y_top1[-1]], s=40, clip_on=False, linewidth=0)
   if x_top1: ax6.annotate(y_top1[-1], xy=[max(x_top1), y_top1[-1]], xytext=[20, -2], textcoords='offset points')
   
   x_top2 = perf_rct_dict[top_val_rct[1]]['iteration'][imin:imax]
   y_top2 = perf_rct_dict[top_val_rct[1]]['value'][imin:imax]   
   if x_top2: ax6.plot(x_top2, y_top2,label=top2_label)
   if x_top2: ax6.scatter([max(x_top2)], [y_top2[-1]], s=40, clip_on=False, linewidth=0)
   if x_top2: ax6.annotate(y_top2[-1], xy=[max(x_top2), y_top2[-1]], xytext=[20, -2], textcoords='offset points')

   x_top3 = perf_rct_dict[top_val_rct[2]]['iteration'][imin:imax]
   y_top3 = perf_rct_dict[top_val_rct[2]]['value'][imin:imax] 
   if x_top3: ax6.plot(x_top3, y_top3,label=top3_label)
   if x_top3: ax6.scatter([max(x_top3)], [y_top3[-1]], s=40, clip_on=False, linewidth=0)
   if x_top3: ax6.annotate(y_top3[-1], xy=[max(x_top3), y_top3[-1]], xytext=[20, -2], textcoords='offset points')

   ax6.yaxis.set_ticks_position('both')
   ax6.minorticks_on()

   ax6.set_xlabel("Iter nb. [-]")
   ax6.set_ylabel('RCT [$\mu$s/(ite.node)]')
   ax6.set_ylim([0,ymax])
   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

################################################################################
def plot_ICS_info(fig,ifile=None):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # ---------------
   # Max Norm U and L2 norm mean of U
   iplot += 1   
   ax7 = fig.add_subplot(nrows,ncols,iplot)
   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1
   
   for adapt_it in adapt_iter[ibeg:iend]:
      ax7.plot([adapt_it,adapt_it],[0,10000],linewidth=0.5,linestyle='--',color=adapt_line_color)
   ax7.plot(ites[imin:imax],umax[imin:imax],label="Max Norm U")

   np_umax = np.array(umax[imin:imax])
   np_umax_isnan = np.isnan(np_umax).all()
   if not np_umax_isnan:
      maxumax = np.nanmax(np_umax)
      maxumax = min(maxumax*1.1,1000)
   else:
      maxumax = None
   ax7.set_xlabel('Iter nb. [-]')
   ax7.set_ylabel('Max Norm U [m/s]')
   ax7.set_ylim([0,maxumax])

   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1

   ax72 = ax7.twinx()
   ax72.plot(ites[imin:imax],L2mean[imin:imax],'r',label="L2 norm mean of U")
   ax72.set_ylabel('L2 norm mean of U', color='r')
   ax72.tick_params('y', colors='r')

   return ax7

################################################################################
def plot_VDS_info(fig,ifile=None):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # ---------------
   # Max Norm U & L2 Norm mean
   ax7 = plot_ICS_info(fig,ifile)

   # ---------------
   # DRHODT and Tmax
   iplot += 1   
   ax10 = fig.add_subplot(nrows,ncols,iplot)
   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax -1

   # define scale for DRHODT
   minmin = min((drhodt_max[imin:imax]))
   maxmax = max((drhodt_max[imin:imax]))
   minmin = minmin * 0.9
   maxmax = maxmax * 1.1
   ymax = np.mean(np.array(drhodt_max[imin:imax])) * 5
   drhodt_max_label = "Max Norm DRHO/DT"
   if maxmax>ymax:
      drhodt_max_label = drhodt_max_label + " (max={:.3e})".format(maxmax)
   else:
      ymax = maxmax
   
   ax10.plot(ites[imin:imax],drhodt_max[imin:imax],label=drhodt_max_label)
   ax10.set_xlabel('Iter nb. [-]')
   ax10.set_ylabel(r'Max norm DRHO/DT')
   ax10.set_ylim([minmin, ymax])
   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

   ax102 = ax10.twinx()
   minmin = min(Tmax[imin:imax]) * 0.98
   maxmax = max(Tmax[imin:imax]) * 1.02
   ax102.plot(ites[imin:imax],Tmax[imin:imax],'r',label="Max norm Temperature")
   ax102.set_ylabel('Max norm T [K]', color='r')
   ax102.tick_params('y', colors='r')
   ax102.set_ylim([minmin,maxmax])   

   # ---------------
   # plot MU_T and MU_ARTIF
   iplot += 1
   ax11 = fig.add_subplot(nrows,ncols,iplot)

   # define scaling
   minmin = min(min(mu_t_max[imin:imax],mu_artif_max[imin:imax]))
   maxmax = max(max(mu_t_max[imin:imax],mu_artif_max[imin:imax]))
   if maxmax == 0:
      fig.delaxes(ax11)
      iplot -= 1
   else:
      minmin = minmin * 0.9
      maxmax = maxmax * 1.1
      ymax = np.mean(np.array(max(mu_t_max[imin:imax],mu_artif_max[imin:imax]))) * 2
      mu_t_max_label = "Max Norm MU T"
      mu_artif_max_label = "Max Norm MU ARTIF"
      if maxmax>ymax:
         mu_t_max_label = mu_t_max_label + " (max={:.3e})".format(max(mu_t_max[imin:imax]))
         mu_artif_max_label = mu_artif_max_label + " (max={:.3e})".format(max(mu_artif_max[imin:imax]))
      else:
         ymax = maxmax

      plot1 = ax11.plot(ites[imin:imax],mu_t_max[imin:imax],label=mu_t_max_label)
      plot2 = ax11.plot(ites[imin:imax],mu_artif_max[imin:imax],label=mu_artif_max_label)
      ax11.set_xlabel('Iter nb. [-]')
      ax11.set_ylabel('Max Norm value')
      ax11.set_yscale('log')
      ax11.grid(axis='y', which='both', linewidth=0.5, linestyle='--', color=adapt_line_color)
      ax11.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10, 2), numticks=100))
      ax11.yaxis.set_minor_formatter(MathTextSciFormatter("%1.1e"))
      ax11.tick_params(axis = 'y', which = 'minor', labelsize = 7)

      try:
         mu_artif_mu_t_ratio = [num / den for num, den in zip(mu_artif_max[imin:imax], mu_t_max[imin:imax])]
         ax112 = ax11.twinx()
         minmin = min(mu_artif_mu_t_ratio) * 0.98
         maxmax = max(mu_artif_mu_t_ratio) * 1.02
         mu_artif_mu_t_ratio_label = "Max Norm MU ARTIF/MU T"
         plot3 = ax112.plot(ites[imin:imax],mu_artif_mu_t_ratio,'r',label=mu_artif_mu_t_ratio_label)
         ax112.set_ylabel('Max norm MU ARTIF/MU T', color='r')
         ax112.tick_params('y', colors='r')
         ax112.set_ylim([minmin,maxmax])
      except ZeroDivisionError:
         mu_artif_mu_t_ratio = 0
         plot3 = []

      # add legend
      lns = plot1 + plot2 + plot3
      labels = [l.get_label() for l in lns]
      leg = ax11.legend(lns, labels)
      leg.get_frame().set_alpha(0.5)

################################################################################
def plot_SPS_info(fig,ifile=None):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # ---------------
   # Max Norm U from ICS plot
   ax7 = plot_ICS_info(fig,ifile)

   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1

   # ICS plot generates a shared axis, then a check if a shared axis exists is performed
   shared_axis = [a for a in ax7.figure.axes if a is not ax7 and a.bbox.bounds == ax7.bbox.bounds]
   if not shared_axis:
      ax72 = ax7.twinx()
   else:
      ax72 = shared_axis[0]
   ax72.plot(ites[imin:imax],volphi[imin:imax],'r',label="Vol PHI")
   ax72.set_ylabel('Vol LS PHI (m3)', color='r')
   ax72.tick_params('y', colors='r')
   ax72.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

################################################################################
def plot_LSS_info(fig,ifile=None):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # ---------------
   # Max Norm U from ICS plot
   ax7 = plot_ICS_info(fig,ifile)

   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1

   # ICS plot generates a shared axis, then a check if a shared axis exists is performed
   shared_axis = [a for a in ax7.figure.axes if a is not ax7 and a.bbox.bounds == ax7.bbox.bounds]
   if not shared_axis:
      ax72 = ax7.twinx()
   else:
      ax72 = shared_axis[0]
   ax72 = ax7.twinx()
   ax72.plot(ites[imin:imax],volphi[imin:imax],'r',label="Vol PHI")
   ax72.set_ylabel('Vol LS PHI (m3)', color='r')
   ax72.tick_params('y', colors='r')
   ax72.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

################################################################################
def plot_adaptation(fig,ifile=None,fontsize=10):
   global iplot

   ibeg = 0
   iend = -1
   plot_type = 'global'
   if ifile != None: # local plot
      ibeg = tot_adapt_iter[ifile]
      iend = tot_adapt_iter[ifile+1]
      plot_type = 'local'

   # Max skewness
   iplot += 1   
   ax1 = fig.add_subplot(nrows,ncols,iplot)

   # plot target first
   maxlen = max([len(ta) for ta in plot_nite[ibeg:iend]])
   maxloc = [len(ta) for ta in plot_nite[ibeg:iend]].index(maxlen)
   pltx = [0]+plot_nite[ibeg+maxloc]
   plty = plot_target_skewness[ibeg+maxloc]
   if not len(pltx)==len(plty):
      len1 = min(len(pltx),len(plty))
      pltx = pltx[:len1]
      plty = plty[:len1]
   ax1.plot(pltx,plty,"--")

   # plot current skewness
   for ix,x in enumerate(plot_nite[ibeg:iend]):
      ixx = ix
      if plot_type == 'local': ixx = ix+tot_adapt_iter[ifile]
      pltx = [0]+x # skewness is given n+1 times in log file
      plty = [x for x in plot_skewness[ixx] if str(x) != 'nan']
      if not len(pltx)==len(plty):
         # Cas d'un run en cours
         len1 = min(len(pltx),len(plty))
         pltx = pltx[:len1]
         plty = plty[:len1]
      ax1.plot(pltx,plty)
      if len(pltx)>0:
         ax1.annotate(""+str(ix+1),xy=(pltx[-1],plty[-1]), xycoords='data',
               xytext=(-50, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle="->"),fontsize=fontsize)
   ax1.set_xlabel("Adaptation substep [-]")
   ax1.set_ylabel("Max skewness [-]")

   # Max metric error
   iplot += 1   
   ax1 = fig.add_subplot(nrows,ncols,iplot)

   # plot target first
   maxlen = max([len(ta) for ta in plot_nite[ibeg:iend]])
   maxloc = [len(ta) for ta in plot_nite[ibeg:iend]].index(maxlen)
   pltx = plot_nite[ibeg+maxloc]
   plty = plot_target_metric_error[ibeg+maxloc]
   if not len(pltx)==len(plty):
      len1 = min(len(pltx),len(plty))
      pltx = pltx[:len1]
      plty = plty[:len1]
   ax1.plot(pltx,plty,"--")

   # plot current max metric error
   for ix,x in enumerate(plot_nite[ibeg:iend]):
      ixx = ix
      if plot_type == 'local': ixx = ix+tot_adapt_iter[ifile]
      plty = [x for x in plot_metric_error[ixx] if str(x) != 'nan']
      if len(x)==len(plty):
         pltx = x
      else:
         # Cas d'un run en cours
         len1 = min(len(x),len(plty))
         pltx = x[:len1]
         plty = plty[:len1]
      ax1.plot(pltx,plty)
      if len(pltx)>0:
         ax1.annotate(""+str(ix+1),xy=(pltx[-1],plty[-1]), xycoords='data',
               xytext=(-50, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle="->"),fontsize=fontsize)
   ax1.set_xlabel("Adaptation substep [-]")
   ax1.set_ylabel("Max metric error [%]")

   # Max hgrad
   iplot += 1   
   ax1 = fig.add_subplot(nrows,ncols,iplot)

   # plot target first
   maxlen = max([len(ta) for ta in plot_nite[ibeg:iend]])
   maxloc = [len(ta) for ta in plot_nite[ibeg:iend]].index(maxlen)
   pltx = plot_nite[ibeg+maxloc]
   plty = plot_target_hgrad[ibeg+maxloc]
   if not len(pltx)==len(plty):
      len1 = min(len(pltx),len(plty))
      pltx = pltx[:len1]
      plty = plty[:len1]
   ax1.plot(pltx,plty,"--")

   # plot current max hgrad
   for ix,x in enumerate(plot_nite[ibeg:iend]):
      ixx = ix
      if plot_type == 'local': ixx = ix+tot_adapt_iter[ifile]
      plty = [x for x in plot_hgrad[ixx] if str(x) != 'nan']
      if len(x)==len(plty):
         pltx = x
      else:
         # Cas d'un run en cours
         len1 = min(len(x),len(plty))
         pltx = x[:len1]
         plty = plty[:len1]
      ax1.plot(pltx,plty)
      if len(pltx)>0:
         ax1.annotate(""+str(ix+1),xy=(pltx[-1],plty[-1]), xycoords='data',
               xytext=(-50, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle="->"),fontsize=fontsize)
   ax1.set_xlabel("Adaptation substep [-]")
   ax1.set_ylabel("Max hgrad [-]")

   # Element numbers
   iplot += 1   
   ax2 = fig.add_subplot(nrows,ncols,iplot)
   if ibeg<iend: iend = iend -1
   ax2.plot(np.array(time_adapt[ibeg:iend])*1e03,np.array(total_nelem[ibeg:iend])/1e06)
   ax2.set_xlabel("Time [ms]")
   ax2.set_ylabel("Nb. elements [x 10$^6$]")
   ax2.get_yaxis().get_major_formatter().set_useOffset(False)

   # Total memory
   ax22 = ax2.twinx()
   ax22.plot(np.array(time_adapt[ibeg:iend])*1e03,np.array(total_mem[ibeg:iend]),'r')
   ax22.set_ylabel('Total memory (MB)', color='r')
   ax22.tick_params('y', colors='r')

   # Nb of bad quality cells 
   iplot += 1   
   ax3 = fig.add_subplot(nrows,ncols,iplot)
   try:
      maxmax = max(np.nanmax(np.array(count_skew_inter)),
                   np.nanmax(np.array(count_skew_total)))
   except:
      maxmax = 100
   for adapt_it in adapt_iter2[ibeg:iend]:
      plt.plot([adapt_it,adapt_it],[0,10000],linewidth=0.5,linestyle='--',color=adapt_line_color)
   imin = 0
   imax = -1
   if plot_type == 'local':
      if ifile == 0:
         imin = tot_adapt_step[ifile]
      else:
         imin = tot_adapt_step[ifile]+1
      imax = tot_adapt_step[ifile+1]
      if imin<imax: imax = imax-1
      imax = imin+min(len(adapt_step[imin:imax]),len(count_skew_inter[imin:imax]),\
                 len(count_skew_total[imin:imax]))
      ax3.plot(adapt_step[imin:imax],count_skew_inter[imin:imax],label="At interface")
      ax3.plot(adapt_step[imin:imax],count_skew_total[imin:imax],linestyle='--',marker='o',label="Total")
   elif plot_type == 'global':
      ax3.plot(count_skew_inter,label="At interface")
      ax3.plot(count_skew_total,marker='o',label="Total")
   ax3.set_xlabel('Cumulated adaptation substep [-]')
   ax3.set_ylabel('Nb. of bad quality cells [-]')
   ax3.set_ylim([0,maxmax])
   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

   # Mean adaptation loop time
   iplot += 1   
   ax4 = fig.add_subplot(nrows,ncols,iplot)
   imax = iend

   if ibeg<iend: imax = iend-1
   if len(total_nelem)>0:
      ax4.plot(iadapt[ibeg:imax],np.array(t_global_ada[ibeg:imax])/np.array(adaptsteps[ibeg:imax]))
   ax4.set_xlabel('Adapt nb. [-]')
   ax4.set_ylabel('Mean adaptation loop time [s/step]')

   # Min, Max and Mean nelem per core during adaptation substeps 
   iplot += 1
   ax3 = fig.add_subplot(nrows,ncols,iplot)
   try:
      maxmax = max(np.nanmax(np.array(min_nelem_full)),
                   np.nanmax(np.array(max_nelem_full)),
                   np.nanmax(np.array(mean_nelem_full)))
   except:
      maxmax = 100
   for adapt_it in adapt_iter2[ibeg:iend]:
      plt.plot([adapt_it,adapt_it],[0,1000000],'0.5',linestyle='--',color=adapt_line_color)
   ax3.axhspan(0,50000,color='grey',alpha=0.5)
   ax3.axhspan(50000,60000,color='#E6E6E6',alpha=0.5)
   ax3.axhspan(200000,250000,color='#E6E6E6',alpha=0.5)
   ax3.axhspan(250000,1000000,color='grey',alpha=0.5)
   imin = 0
   imax = -1
   if plot_type == 'local':
      if ifile == 0:
         imin = tot_adapt_step[ifile]
      else:
         imin = tot_adapt_step[ifile]+1
      imax = tot_adapt_step[ifile+1]
      if imin<imax: imax = imax-1
      imax = imin+min(len(adapt_step[imin:imax]),
                      len(min_nelem_full[imin:imax]),
                      len(max_nelem_full[imin:imax]),
                      len(mean_nelem_full[imin:imax]))
      ax3.plot(adapt_step[imin:imax], min_nelem_full[imin:imax])
      ax3.plot(adapt_step[imin:imax], max_nelem_full[imin:imax])
      ax3.plot(adapt_step[imin:imax],mean_nelem_full[imin:imax])
   elif plot_type == 'global':
      ax3.plot( min_nelem_full)
      ax3.plot( max_nelem_full)
      ax3.plot(mean_nelem_full)
   ax3.set_xlabel('Cumulated adaptation substep [-]')
   ax3.set_ylabel('Min Max Mean nelem per core [-]')
   ax3.set_ylim([0,maxmax])
   leg.get_frame().set_alpha(0.5)
   
   # Min, Max and Mean nelem per core
   iplot += 1   
   ax5 = fig.add_subplot(nrows,ncols,iplot)
   imin = ibeg
   imax = iend
   if imin<imax: imax = imax-1
   #if plot_type == 'global':
   ax5.axhspan(0,50000,color='grey',alpha=0.5)
   ax5.axhspan(50000,60000,color='#E6E6E6',alpha=0.5)
   ax5.axhspan(200000,250000,color='#E6E6E6',alpha=0.5)
   ax5.axhspan(250000,1000000,color='grey',alpha=0.5)
   if len(total_nelem)>0:
      ax5.plot(iadapt[ibeg:imax], min_nelem[ibeg:imax])
      ax5.plot(iadapt[ibeg:imax], max_nelem[ibeg:imax])
      ax5.plot(iadapt[ibeg:imax],mean_nelem[ibeg:imax])
      if len(max_nelem[ibeg:iend])>0: ax5.set_ylim([0,1.1*np.nanmax(max_nelem[ibeg:iend])])
   else:
      ax5.set_ylim([0,None])
   ax5.set_xlabel('Adapt nb. [-]')
   ax5.set_ylabel('Min Max Mean nelem per core [-]')
   
   # Time in each step of adaptation 
   iplot += 1   
   ax9 = fig.add_subplot(nrows,ncols,iplot)
   for adapt_it in adapt_iter2[ibeg:iend]:
      ax9.plot([adapt_it,adapt_it],[0,10000], linewidth=0.5, linestyle='--',color=adapt_line_color)
   imin = ibeg
   imax = iend
   if plot_type == 'local':
      if ifile == 0:
         imin = tot_adapt_step[ifile]
      else:
         imin = tot_adapt_step[ifile]+1
      imax = tot_adapt_step[ifile+1]
      if imin<imax: imax = imax-1
   try:
      maxmax = max(np.nanmax(np.array(  interp_time[imin:imax])),
                   np.nanmax(np.array(coloring_time[imin:imax])),
                   np.nanmax(np.array(     mmg_time[imin:imax])),
                   np.nanmax(np.array(transfer_time[imin:imax])))
   except:
      maxmax = 100
   if plot_type == 'local':
      imax = imin+min(  len(adapt_step[imin:imax]),
                        len(interp_time[imin:imax]),
                        len(mmg_time[imin:imax]),
                        len(coloring_time[imin:imax]),
                        len(transfer_time[imin:imax]))
      ax9.plot(adapt_step[imin:imax],  interp_time[imin:imax],marker='o',linestyle=':',label='Interpolation')
      ax9.plot(adapt_step[imin:imax],     mmg_time[imin:imax],marker='o',linestyle=':',label='Adaptation')
      ax9.plot(adapt_step[imin:imax],coloring_time[imin:imax],marker='o',linestyle=':',label='Coloring')
      ax9.plot(adapt_step[imin:imax],transfer_time[imin:imax],marker='o',linestyle=':',label='Transfer')
   elif plot_type == 'global':
      ax9.plot(interp_time  ,marker='o',linestyle=':',label='Interpolation')
      ax9.plot(mmg_time     ,marker='o',linestyle=':',label='Adaptation')
      ax9.plot(coloring_time,marker='o',linestyle=':',label='Coloring')
      ax9.plot(transfer_time,marker='o',linestyle=':',label='Transfer')
   ax9.set_xlabel('Cumulated adaptation substep [-]')
   ax9.set_ylabel('Time [s]')
   ax9.set_ylim([0,maxmax])
   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

   # Min cell size and metric interface
   iplot += 1
   ax10 = fig.add_subplot(nrows,ncols,iplot)
   imin = ibeg
   imax = iend
   if plot_type == 'local':
      try:
         imin = ites.index(tot_ites[ifile])+1
      except:
         imin = 0
      imax = ites.index(tot_ites[ifile+1])+1
      if imin<imax: imax = imax-1
   for adapt_it in adapt_iter[ibeg:iend]:
      ax10.plot([adapt_it,adapt_it],[0,10000],linewidth=0.5,linestyle='--',color=adapt_line_color)

   np_dxmin = np.array(dxmin_nodevol)
   np_dxmin = np_dxmin*1000.0
   buf_ites = np.array(ites[imin:imax])
   buf_dxmin= np.array(np_dxmin[imin:imax])
   mask = ~np.isnan(buf_dxmin)
   ax10.plot(buf_ites[mask],buf_dxmin[mask],label="$\Delta_x$ min (nodevol)")

   np_dxmin = np.array(dxmin_pairbased)
   np_dxmin = np_dxmin*1000.0
   buf_ites = np.array(ites[imin:imax])
   buf_dxmin= np.array(np_dxmin[imin:imax])
   mask = ~np.isnan(buf_dxmin)
   ax10.plot(buf_ites[mask],buf_dxmin[mask],label="$\Delta_x$ min (pair-based)")

   np_dxmin_isnan = np.isnan(np_dxmin).all()
   if np_dxmin_isnan: np_dxmin = np.nan_to_num(np_dxmin)
  
   np_metricval = np.array(metric_interface_value)
   np_metricval = np_metricval*1.0e6
   buf_ites = np.array(ites[imin:imax])
   buf_dxmin= np.array(np_metricval[imin:imax])
   mask = ~np.isnan(buf_dxmin)
   ax10.plot(buf_ites[mask],buf_dxmin[mask],label="Metric interface") 
   np_metricval_isnan = np.isnan(np_metricval).all()
   if np_metricval_isnan: np_metricval = np.nan_to_num(np_metricval)

   maxdxmin = max(np.nanmax(np_dxmin), np.nanmax(np_metricval))
   if np.isnan(maxdxmin):
      maxdxmin = 100
   else:
      maxdxmin = min(maxdxmin*1.1,1000)

   ax10.set_xlabel('Iter nb. [-]')
   ax10.set_ylabel('$\Delta_x$ min [$\mu$m]')
   ax10.set_ylim([0,maxdxmin])
   leg = plt.legend()
   leg.get_frame().set_alpha(0.5)

class MathTextSciFormatter(mticker.Formatter):
   '''
      Class to set scientific notation on plot ticks label 
      Solution found here : https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
   '''
   def __init__(self, fmt="%1.1e"):
      self.fmt = fmt
   def __call__(self, x, pos=None):
      s = self.fmt % x
      decimal_point = '.'
      positive_sign = '+'
      tup = s.split('e')
      significand = tup[0].rstrip(decimal_point)
      sign = tup[1][0].replace(positive_sign, '')
      exponent = tup[1][1:].lstrip('0')
      if exponent:
         exponent = '10^{%s%s}' % (sign, exponent)
      if significand and exponent:
         s =  r'%s{\times}%s' % (significand, exponent)
      else:
         s =  r'%s%s' % (significand, exponent)
      return "${}$".format(s)

#####################
# Main
#####################
if __name__ == '__main__':

   from docopt import docopt
   help = """
         Plot values from one or several YALES2 logfiles

         Note:
           If you have a fixed and long list of input files you can use
           the template y2_post_log_template.py to use y2_post_log.py as a module

         Usage:
           y2_post_log.py (-h | --help)
           y2_post_log.py [-o <path> | --output <path>] [--nomail] [--tex]
           y2_post_log.py <file> ... [-o <path> | --output <path>] [--nomail] [--tex]

         Options:
           -h --help                    Show this screen
           -o <path>, --output <path>   PDF output folder (local folder by default)
           --tex                        Use LaTeX style
           --nomail                     Do not send PDF plot by mail

         Examples:
           y2_post_log.py                                         Find the solver01_rank0??0.log in the current directory and plot it
           y2_post_log.py log/logfile_23322.o                     Plot the logfile_23322.o
           y2_post_log.py log/logfile_23322.o log/logfile_23323.o Plot the sequence of logfile_23322.o and logfile_23323.o
           y2_post_log.py log/logfile_23322.o -o my_output_folder Plot the logfile_23322.o and save pdf plot in my_output_folder
           y2_post_log.py log/logfile_23322.o --nomail            Plot the logfile_23322.o without sending pdf by mail
         """
   # Get arguments and load the workflow
   arguments = docopt(help)

   if arguments['<file>']==[]:
      my_cases = ["."]
   elif len(arguments['<file>'])>0:
      my_cases = arguments['<file>']

   if arguments['--output'] is None:
      output_path = "."
   else:
      output_path = arguments['--output']
   
   use_tex   = arguments['--tex']
   send_mail = not arguments['--nomail']

   plot_log(my_cases,output_path,use_tex,save_pdf=True,show_figs=False,send_by_mail=send_mail,fig_size=(16,12))

