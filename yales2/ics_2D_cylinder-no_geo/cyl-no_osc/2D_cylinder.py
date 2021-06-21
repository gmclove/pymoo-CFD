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

        if (comm_defs_m.myworker==comm_defs_m.master):
           print("From python: u_max, z_min, z_max: ",u_max,z_min,z_max)

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
    # get solver
    solver = yales2_m.get_first_solver()

    # ------------------------
    # temporal loop
    yales2_m.temporal_loop()

    # ------------------------
    # destroy
    yales2_m.destroy_yales2(destroympi=True)

except:
    print(traceback.format_exc())
    message_m.error_and_exit("Problem in 2D_cylinder")
