#!/usr/local/bin/python

from yales2 import *

import os
import numpy
import ctypes
import traceback

# load the shared library
prefix=os.getenv('YALES2_HOME')
mylib = ctypes.CDLL(prefix+"/src/python/liby2callback.so", mode = ctypes.RTLD_GLOBAL)

#------------------------------------------------------------------------------
# UDF

#===================================================
def initialize_data():
    try:

        if (comm_defs_m.myworker==comm_defs_m.master):
            print("Initialize data")

        solver = yales2_m.get_first_solver()
        grid = solver.first_grid
        first_data = grid.first_data

        u_ptr, found = data_defs_m.find_data(first_data,"U")
        if (not found):
            print('Data U not found')
            raise(Exception())

        z_ptr, found = data_defs_m.find_data(first_data,"Z")
        if (not found):
            print('Data Z not found')
            raise(Exception())

        for n in range(grid.nel_grps):
            el_grp = grid.el_grps[n].ptr
            u_val = u_ptr.r2_ptrs[n].ptr.val
            z_val = z_ptr.r1_ptrs[n].ptr.val
            for ino in range(el_grp.nnode):
                u_val[0][ino] = 0.15
                u_val[1][ino] = 0.00
                z_val[ino] = 1.0

    except:
        print(traceback.format_exc())
        message_m.error_and_exit("Error in initialize_data")

    return

#===================================================
def temporal_loop_preproc():
    try:
        solver = yales2_m.get_first_solver()
        grid = solver.first_grid
        first_data = grid.first_data

        u_ptr, found = data_defs_m.find_data(first_data,"U")
        if (not found):
            print('Data U not found')
            raise(Exception())

        z_ptr, found = data_defs_m.find_data(first_data,"Z")
        if (not found):
            print('Data Z not found')
            raise(Exception())

        u_max = data_checking_m.compute_maxnorm_data(u_ptr,grid)
        z_min = data_checking_m.compute_minnorm_data(z_ptr,grid)
        z_max = data_checking_m.compute_maxnorm_data(z_ptr,grid)

        z_pt1 = probe_operators_m.get_probe_r0_value("pt1",z_ptr,grid)

        if (comm_defs_m.myworker==comm_defs_m.master):
            time_array.append(solver.time)
            u_max_array.append(u_max)
            z_min_array.append(z_min)
            z_max_array.append(z_max)
            z_pt1_array.append(z_pt1)
            if (solver.niter%100==0):
                drawnow(update_plot,show_once=True)

    except:
        print(traceback.format_exc())
        message_m.error_and_exit("Error in temporal_loop_preproc")

    return

#===================================================
def temporal_loop_postproc():
    try:
        solver = yales2_m.get_first_solver()
        grid = solver.first_grid
        first_data = grid.first_data

    except:
        print(traceback.format_exc())
        message_m.error_and_exit("Error in temporal_loop_postproc")

    return
    
#------------------------------------------------------------------------------
# wrap and set the UDF
py2f_wrapper = ctypes.CFUNCTYPE(None)
py2f_initialize_data = py2f_wrapper(initialize_data)
mylib.set_udf_initialize_data(py2f_initialize_data)
py2f_temporal_loop_preproc = py2f_wrapper(temporal_loop_preproc)
mylib.set_udf_temporal_loop_preproc(py2f_temporal_loop_preproc)
py2f_temporal_loop_postproc = py2f_wrapper(temporal_loop_postproc)
mylib.set_udf_temporal_loop_postproc(py2f_temporal_loop_postproc)

#------------------------------------------------------------------------------

try:
    # ------------------------
    # inputfile
    inputfile = "2D_cylinder.in"

    # ------------------------
    # init
    yales2_m.init_yales2(inputfile,initmpi=True)

    # ------------------------
    # init plot
    time_array = []
    u_max_array = []
    z_min_array = []
    z_max_array = []
    z_pt1_array = []
   
    if (comm_defs_m.myworker==comm_defs_m.master):
        import matplotlib.pyplot as plt
        from drawnow import drawnow,figure
       
        plt.ion()
        plt.figure(figsize=(12,7))
   
        def update_plot(): # update the data
            sub1 = plt.subplot(221)
            sub1.set_ylabel('max(u)')
            sub1.plot(time_array,u_max_array,'-',color='black')
            sub2 = plt.subplot(222)
            sub2.set_ylabel('min(z)')
            sub2.plot(time_array,z_min_array,'-',color='red')
            sub3 = plt.subplot(223)
            sub3.set_ylabel('max(z)')
            sub3.set_xlabel('Time (s)')
            sub3.plot(time_array,z_max_array,'-',color='blue')
            sub4 = plt.subplot(224)
            sub4.set_ylabel('z_pt1')
            sub4.set_xlabel('Time (s)')
            sub4.plot(time_array,z_pt1_array,'-',color='green')

    # ------------------------
    # modify niter_max
    solver = yales2_m.get_first_solver()
    #solver.niter_max = 10

    # ------------------------
    # temporal loop
    yales2_m.temporal_loop()

    # ------------------------
    # destroy
    yales2_m.destroy_yales2(destroympi=True)

except:
    print(traceback.format_exc())
    message_m.error_and_exit("Problem in 2D_cylinder")
