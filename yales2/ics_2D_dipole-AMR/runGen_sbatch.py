import subprocess
from distutils.dir_util import copy_tree
import time
import h5py
import os

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
        thres = para[0]
        prop = para[1]
        ####### Simulation Input Parameters ###########
        inputDir = indDir + '/DynamicAdaptation.in'
        # open and read YALES2 input file to array of strings for each line
        with open(inputDir, 'r') as f_orig:
            in_lines = f_orig.readlines()

        # find line that must change using a keyword
        keyword = 'THRESHOLD'
        keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
        # create new string to replace line
        newLine = keyword_line[:keyword_line.find('=')+2] + str(thres) + '\n'
        in_lines[keyword_line_i] = newLine

        # find line that must change using a keyword
        keyword = 'PROPAGATION_STEPS'
        keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
        # create new string to replace line
        newLine = keyword_line[:keyword_line.find('=')+2] + str(prop) + '\n'
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

        ######## Objective 1: Simulation Time #########
        # sim_time should be on the 11th to last line (grep last 20 to be safe)
        cmd = 'tail -n 11 '+indDir+'/solver01_rank0.log | grep "WALL CLOCK TIME"'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        time = process.communicate()[0][-9:]  # extract time value in last 9 characters of line
        # print(time)
        sim_time = float(time)

        ######## Objective 2: Position of vortex #########
        # sort dump files to find latest mesh and solution files
        folder = indDir + '/dump'
        litems = os.listdir(folder)
        litems.sort()
        for i in range(len(litems)):
            if litems[i].endswith('.mesh.h5'):
                latestMesh = litems[i]
            if litems[i].endswith('.sol.h5'):
                latestSoln = litems[i]

        # extract location (x_coor, y_coor) of max and min vorticity
        with h5py.File(indDir + '/dump/' + latestSoln, 'r') as f:
            vort = f['Data']['VORTICITY'][:]
        with h5py.File(indDir + '/dump/' + latestMesh, 'r') as f:
            coor = f['Coordinates']['XYZ'][:][:, :2]
        imax = np.argmax(vort)
        imin = np.argmin(vort)
        amr_vMaxCoor = coor[imax, :]
        amr_vMinCoor = coor[imin, :]
        # coordinates of max/min vorticity from dense cartesian mesh simulation
        cart_vMinCoor = np.array([20.9, -1.5])
        cart_vMaxCoor = np.array([20.9, 1.5])
        vMax_perErr = abs((cart_vMaxCoor - amr_vMaxCoor) / cart_vMaxCoor) * 100
        vMin_perErr = abs((cart_vMinCoor - amr_vMinCoor) / cart_vMinCoor) * 100
        sum_perErr = sum(vMin_perErr + vMax_perErr)

        obj[ind] = [sim_time, sum_perErr]

    # Normalize objectives if their values are on different orders of magnitude
    normalize(obj)
    return obj





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
    for obj_i in obj:
        # objective 1 normalization
        obj_norm[0] = (obj_i[0] - obj_o[0])/(obj_max[0] - obj_o[0])
