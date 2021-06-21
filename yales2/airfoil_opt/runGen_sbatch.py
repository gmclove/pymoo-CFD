import subprocess
from distutils.dir_util import copy_tree
import time
import h5py
import os
import numpy as np
from scipy.integrate import simpson

from pymooIN import *


def runGen(x, gen):
    global genDir, n_ind
    n_ind = len(x)
    genDir = './gen%i' % gen
    ############################
    ####### PRE PROCESS ########
    ############################
    for ind in range(n_ind):
        indDir = f'{genDir}/ind{ind}'
        copy_tree('base_case', indDir)

        editJobslurm(gen, ind, indDir)

        # Extract parameters for each individual
        para = x[ind, :]
        amp = para[0]
        freq = para[1]
        ####### Simulation Input Parameters ###########
        inputDir = indDir + '/2D_cylinder.in'
        # open and read YALES2 input file to array of strings for each line
        with open(inputDir, 'r') as f_orig:
            in_lines = f_orig.readlines()

        # find line that must change using a keyword
        keyword = 'CYL_ROTATION_PROP'
        keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
        # create new string to replace line
        newLine = keyword_line[:keyword_line.find('=')+2] + str(amp) + ' ' + str(freq)+ '\n'
        in_lines[keyword_line_i] = newLine

        # REPEAT FOR EACH LINE THAT MUST BE CHANGED

        with open(inputDir, 'w') as f_new:
            f_new.writelines(in_lines)

    ####################################
    ####### EXECUTE SIMULATIONS ########
    ####################################
    execSims(gen)

    #############################
    ####### POST PROCESS ########
    #############################
    obj = np.ones((n_ind, n_obj))

    for ind in range(n_ind):
        indDir = f'{genDir}/ind{ind}'
        # Extract parameters for each individual
        para = x[ind, :]
        amp = para[0]
        freq = para[1]
        ######## Compute Objectives ##########
        ######## Objective 1: Drag on Cylinder #########
        # create string for directory of individual's data file
        dataDir = f'{indDir}/ics_temporals.txt'
        data = np.genfromtxt(dataDir, skip_header=1)
        # collect data after 8 seconds
        mask = np.where(data[:,1] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[mask, 4]
        tau_intgrl_1 = data[mask, 6]
        F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        C_drag = F_drag/((1/2)*rho*U**2*D**2)
        # LIFT: y-direction integrals
        # extract P_OVER_RHO_INTGRL_(2) and TAU_INTGRL_(2)
        p_over_rho_intgrl_2 = data[mask, 5]
        tau_intgrl_2 = data[mask, 7]
        F_lift = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        C_lift = F_lift/((1/2)*rho*U**2*D**2)



        ######## Objective 2: Position of vortex #########
        # Objective 2: Calculate area inside airfoil where fuel is stored
        # use composite simpson's rule to find absolute value of area
        area = abs(simpson(y, x))

        # drag minimized, lift and area maximized
        obj[ind] = [-lift, drag, -area]

    # Normalize objectives if their values are on different orders of magnitude
    # normalize(obj)
    return obj


################################################################################
################################################################################
################################################################################
################################################################################
#####################
##### Functions #####
#####################
def editJobslurm(gen, ind, indDir):
    # change jobslurm.sh to correct directory and change job name
    with open(indDir + '/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find('base_case')] + 'gen%i/ind%i' % (gen, ind) + '\n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'g%i.i%i' % (gen, ind) + '\n'
    job_lines[keyword_line_i] = newLine
    with open(indDir + '/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)



def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


def execSims(gen):
    # check if generation is already complete
    completeCount = 0
    for ind in range(n_ind):
        indDir = genDir + '/ind%i/' % ind
        if os.path.exists(indDir+'solver01_rank00.log'):
            completeCount += 1
    if completeCount == n_ind:
        return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for ind in range(n_ind):
        # create string for directory of individuals job slurm shell file
        indDir = genDir + '/ind%i/jobslurm.sh' % ind
        out = subprocess.check_output(['sbatch', indDir])
        # Extract number from following: 'Submitted batch job 1847433'
        # print(int(out[20:]))
        batchIDs.append(int(out[20:]))
    # print(batchIDs)

    waiting = True
    count = np.ones(n_ind)
    processes = []
    while waiting:
        for bID_i in range(len(batchIDs)):
            # grep for batch ID of each individual
            out = subprocess.check_output('squeue | grep --count %i || :' % batchIDs[bID_i], shell=True)  # '|| :' ignores non-zero exit status error
            count[bID_i] = int(out)
            # if job batch number can not be found then start post-processing
            # if count[bID_i] == 0:
            #     postProc(bID_i)
            #     # Run post processing once simulation finishes
            #     # proc = Process(target=postProc(bID_i))
            #     # proc.start()
            #     # processes.append(proc)

        # print(count)
        # check if all batch jobs are done
        if sum(count) == 0:
            # wait for post processing to complete
            # for proc in processes:
            #     proc.join()
            # end while loop
            waiting = False
        # print(count)
        # print('SUM OF COUNT = %i' % sum(count))
        time.sleep(1)


    # print('GEN%i: EXECUTING SIMULATION COMPLETE' % gen)



def normalize(obj):
    # for loop through each individual
    for obj_ind in obj:
        # objective 1 normalization
        obj_norm = (obj_ind - obj_o)/(obj_max - obj_o)
