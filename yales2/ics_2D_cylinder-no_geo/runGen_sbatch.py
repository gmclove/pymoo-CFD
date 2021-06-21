import subprocess
from distutils.dir_util import copy_tree
import time
import h5py
import os
import numpy as np
from scipy.integrate import quad

from pymooIN import *
# from exectBatch import execSims


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

        # since we are using slurm to exectute our simulations we must edit
        # the file used to lauch our job
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
    execSims(gen, n_ind)

    #############################
    ####### POST PROCESS ########
    #############################
    obj = np.ones((n_ind, n_obj))

    for ind in range(n_ind):
        indDir = f'{genDir}/ind{ind}'
        # Extract parameters for each individual
        para = x[ind, :]
        omega, freq = getVar(para)
        ######## Compute Objectives ##########
        ######## Objective 1: Drag on Cylinder #########
        U = 1
        rho = 1
        D = 1
        # create string for directory of individual's data file
        dataDir = f'{indDir}/ics_temporals.txt'
        try:
            data = np.genfromtxt(dataDir, skip_header=1)
        except IOError:
            print('no ics_temporals.txt created')
            print(IOError)
            obj[ind] = [0, 0]
            continue
        # data = np.genfromtxt(dataDir, skip_header=1)

        # collect data after 100 seconds of simulation time
        mask = np.where(data[:,1] > 100)
        # Surface integrals of Cp and Cf
        # DRAG: x-direction integrals
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[mask, 4]
        tau_intgrl_1 = data[mask, 6]
        F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        C_drag = F_drag/((1/2)*rho*U**2*D**2)

        ######## Objective 2 #########
        # Objective 2: Power consumed by rotating cylinder
        D = 1  # [m] cylinder diameter
        t = 0.1  # [m] thickness of cylinder wall
        r_o = D/2  # [m] outer radius
        r_i = r_o-t  # [m] inner radius
        d = 2700  # [kg/m^3] density of aluminum
        L = 1  # [m] length of cylindrical tube
        V = L*np.pi*(r_o**2-r_i**2) # [m^3] volume of cylinder
        m = d*V # [kg] mass of cylinder
        I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
        P_cyc = 0.5*I*quad(lambda t : (omega*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle

        obj[ind] = [C_drag, P_cyc]

    # Normalize objectives if their values are on different orders of magnitude
    normalize(obj)
    return obj





################################################################################
#####################
##### Functions #####
#####################

def getObj(simDir):
    ######## Compute Objectives ##########
    ######## Objective 1: Drag on Cylinder #########
    U = 1
    rho = 1
    D = 1
    # create string for directory of individual's data file
    dataDir = f'{simDir}/ics_temporals.txt'
    # try:
    #     data = np.genfromtxt(dataDir, skip_header=1)
    # except IOError:
    #     print('no ics_temporals.txt created')
    #     print(IOError)
    #     obj[ind] = [0, 0]
    #     continue
    data = np.genfromtxt(dataDir, skip_header=1)

    # collect data after 100 seconds of simulation time
    mask = np.where(data[:,1] > 100)
    # Surface integrals of Cp and Cf
    # DRAG: x-direction integrals
    # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
    p_over_rho_intgrl_1 = data[mask, 4]
    tau_intgrl_1 = data[mask, 6]
    F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
    C_drag = F_drag/((1/2)*rho*U**2*D**2)

    ######## Objective 2 #########
    # Objective 2: Power consumed by rotating cylinder
    D = 1  # [m] cylinder diameter
    t = 0.1  # [m] thickness of cylinder wall
    r_o = D/2  # [m] outer radius
    r_i = r_o-t  # [m] inner radius
    d = 2700  # [kg/m^3] density of aluminum
    L = 1  # [m] length of cylindrical tube
    V = L*np.pi*(r_o**2-r_i**2) # [m^3] volume of cylinder
    m = d*V # [kg] mass of cylinder
    I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
    P_cyc = 0.5*I*quad(lambda t : (omega*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle
    # print(P_cyc)

    np.savetxt('objectives.txt', [C_drag, P_cyc])
    return C_drag, P_cyc

def getVar(para):
    amp = para[0]
    freq = para[1]
    return amp, freq

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


def execSims(gen, n_ind):
    # check if meshStudy is already complete
    # completeCount = 0
    # for ind in range(n_ind):
    #     indDir = f'{genDir}/ind{ind}'
    #     if os.path.exists(f'{indDir}/{output_file}'):
    #         completeCount += 1
    # if completeCount == n_ind:
    #     return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for ind in range(n_ind):
        indDir = f'{genDir}/ind{ind}'
        if os.path.exists(f'{indDir}/{output_file}'):
            continue
        else:
            # create string for directory of individuals job slurm shell file
            jobDir = f'{indDir}/jobslurm.sh'
            out = subprocess.check_output(['sbatch', jobDir])
            # Extract number from following: 'Submitted batch job 1847433'
            # print(int(out[20:]))
            batchIDs.append(int(out[20:]))
    # print(batchIDs)

    waiting = True
    count = np.ones(len(batchIDs))
    processes = []
    while waiting:
        for bID_i in range(len(batchIDs)):
            # grep for batch ID of each individual
            out = subprocess.check_output('squeue | grep --count %i || :' % batchIDs[bID_i], shell=True)  # '|| :' ignores non-zero exit status error
            count[bID_i] = int(out)
        # print(count)
        # check if all batch jobs are done
        if sum(count) == 0:
            waiting = False
        # print(count)
        # print('SUM OF COUNT = %i' % sum(count))
        time.sleep(1)

    print('BATCH OF SLURM SIMULATIONS COMPLETE')
