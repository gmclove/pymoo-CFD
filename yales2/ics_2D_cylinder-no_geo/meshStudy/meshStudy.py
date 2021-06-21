import subprocess
from distutils.dir_util import copy_tree
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from genMesh_rect import genMesh

inputDir = '../base_case'
studyDir = '.'
# meshSF = [0.5, 1, 1.5, 1.75, 2, 2.25, 2.5]
# meshSF = np.linspace(0.5,2,8)
# meshSF = [0.2, 0.3, 0.4, 0.5, 0.71428571, 0.92857143, 1.14285714, 1.35714286, 1.57142857, 1.78571429, 2]
meshSF = [0.1] #, 0.2, 0.3, 0.4, 0.5] #, 0.7142857142857143, 0.9285714285714286, 1.1428571428571428, 1.3571428571428572, 1.5714285714285714, 1.7857142857142856, 2.0]

# try:
#     os.mkdir(studyDir)
# except OSError:
#     print(studyDir + ' directory already exists')

def meshStudy():
    ###################################
    ######### PRE-PROCESS #############
    ###################################
    for sf in meshSF:
        sfDir = f'{studyDir}/{sf}'
        copy_tree(inputDir, f'{sfDir}')

        genMesh(sfDir, 1, sf)

        # edit jobslurm files so directories are correct
        editJobslurm(sf)

    ###################################
    ########### EXECUTION #############
    ###################################
    execSims()

    ###################################
    ######### POST-PROCESS ############
    ###################################
    C_drag = np.zeros(len(meshSF))
    i=0

    for sf in meshSF:
        sfDir = f'{studyDir}/{sf}'
        # Extract Wall Clock Time
        cmd = 'tail -n 11 '+sfDir+'/solver01_rank0.log | grep "WALL CLOCK TIME"'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        time = process.communicate()[0][-9:]  # extract time value in last 9 characters of line
        # print(time)
        sim_time = float(time)

        ######## Objective 1: Drag on Cylinder #########
        rho = 1
        U = 1
        D = 1
        # create string for directory of individual's data file
        dataDir = f'{sfDir}/ics_temporals.txt'
        data = np.genfromtxt(dataDir, skip_header=1)
        # collect data after 8 seconds
        mask = np.where(data[:,1] > 100)
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[mask, 4]
        tau_intgrl_1 = data[mask, 6]
        F_drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
        coeff_drag = F_drag/((1/2)*rho*U**2*D**2)
        C_drag[i] = coeff_drag
        i += 1

    np.save('data', [meshSF, C_drag])
    plot(meshSF, C_drag)


def plot(sf, C_drag):
    plt.plot(sf, C_drag)
    plt.suptitle('Mesh Sensitivity Study')
    plt.title(r'$Re=150,f=0.4,\omega=2.0$')
    plt.xlabel('Mesh Max Size')
    plt.ylabel('Coefficient of Drag')
    plt.savefig(f'{studyDir}/drag-meshSize.png')
    # plt.show()

def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i

def execSims():
    # # check if meshStudy is already complete
    # completeCount = 0
    # for sf in meshSF:
    #     sfDir = f'{studyDir}/{sf}'
    #     if os.path.exists(f'{sfDir}/solver01_rank00.log'):
    #         completeCount += 1
    # if completeCount == len(meshSF):
    #     return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for sf in meshSF:
        sfDir = f'{studyDir}/{sf}'
        if os.path.exists(f'{sfDir}/solver01_rank00.log'):
            pass
        else:
            # create string for directory of individuals job slurm shell file
            jobDir = f'{sfDir}/jobslurm.sh'
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

    print('MESH STUDY EXECUTING SIMULATIONS COMPLETE')


def editJobslurm(sf):
    # change jobslurm.sh to correct directory and change job name
    with open('./' + str(sf) + '/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = 'cd ' + os.getcwd() + '/' + str(sf) + '\n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'meshStudy' + '\n'
    job_lines[keyword_line_i] = newLine
    with open('./' + str(sf)+ '/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)


meshStudy()
