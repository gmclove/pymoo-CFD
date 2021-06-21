makeDir(np)import subprocess
from distutils.dir_util import copy_tree
import time
import os
import numpy as np
import matplotlib.pyplot as plt

inputDir = '../base_case'

studyDir = '.'
# try:
#     os.mkdir(studyDir)
# except OSError:
#     print(studyDir + ' directory already exists')

nProc = [1, 5, 10, 15, 20, 25, 30]

def nProcStudy():
    ###################################
    ######### PRE-PROCESS #############
    ###################################
    for np in nProc:
        npDir = makeDir(np)
        copy_tree(inputDir, f'{npDir}')
        # edit jobslurm file so directory, job-name and num. nodes is correct
        editJobslurm(np)

    ###################################
    ########### EXECUTION #############
    ###################################
    execSims()

    ###################################
    ######### POST-PROCESS ############
    ###################################
    C_drag = np.zeros(len(nProc))
    sim_time = np.zeros(len(nProc))
    i=0

    for np in nProc:
        npDir = makeDir(np)
        # Extract Wall Clock Time
        # sim_time should be on the 11th to last line (grep last 20 to be safe)
        cmd = f'tail -n 11 {npDir}/solver01_rank00.log | grep "WALL CLOCK TIME"'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        time = process.communicate()[0][-9:]  # extract time value in last 9 characters of line
        # print(time)
        sim_time[i] = float(time)
        i += 1

    np.save('data', [nProc, C_drag])
    plot(nProc, C_drag)



################################################################################
################################################################################
################################################################################
################################
######## FUNCTIONS #############
################################
def plot(x, y):
    plt.plot(x, y)
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
    # for np in nProc:
    #     npDir = makeDir(np)
    #     if os.path.exists(f'{npDir}/solver01_rank00.log'):
    #         completeCount += 1
    # if completeCount == len(nProc):
    #     return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for np in nProc:
        npDir = makeDir(np)
        if os.path.exists(f'{npDir}/solver01_*'):
            pass
        else:
            # create string for directory of individuals job slurm shell file
            jobDir = f'{npDir}/jobslurm.sh'
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


def editJobslurm(np):
    npDir = makeDir(np)
    # change jobslurm.sh to correct directory and change job name
    with open(f'{npDir}/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = f'cd {os.getcwd()}/np-{np} \n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'np-' + str(np) + '\n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'nodes='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + str(np) + '\n'
    job_lines[keyword_line_i] = newLine
    with open(f'{npDir}/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)

def makeDir(np):
    return f'{studyDir}/np-{np}'

nProcStudy()
