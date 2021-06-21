import subprocess
from distutils.dir_util import copy_tree
import time
import h5py
import os
import numpy as np
import matplotlib as plt



inputFile = '2D_vortex.in'
inputDir = './base_case/' + inputFile
studyDir = './meshStudy'
# meshSF = [0.5, 1, 1.5, 1.75, 2, 2.25, 2.5]
meshSF = np.linspace(0.5,4,8)

# try:
#     os.mkdir(studyDir)
# except OSError:
#     print(studyDir + ' directory already exists')

def meshStudy():
    ###################################
    ######### PRE-PROCESS #############
    ###################################
    with open(inputDir, 'r') as f:
        in_lines = f.readlines()

    # extract nx value
    keyword = 'CART_NX'
    nx_line, nx_line_i = findKeywordLine(keyword, in_lines)
    nx = int(nx_line[nx_line.find('=')+2:])
    # extract ny value
    keyword = 'CART_NY'
    ny_line, ny_line_i = findKeywordLine(keyword, in_lines)
    ny = int(ny_line[ny_line.find('=')+2:])


    for sf in meshSF:
        nx_new = round(nx*sf)
        nx_newLine = nx_line[:nx_line.find('=')+2] + str(nx_new) + '\n'
        in_lines[nx_line_i] = nx_newLine

        ny_new = round(ny*sf)
        ny_newLine = ny_line[:ny_line.find('=')+2] + str(ny_new) + '\n'
        in_lines[ny_line_i] = ny_newLine

        sfDir = f'{studyDir}/{sf}'
        copy_tree('base_case', f'{sfDir}')
        with open(f'{sfDir}/{inputFile}', 'w') as f_new:
            f_new.writelines(in_lines)

        # edit jobslurm files so directories are correct
        editJobslurm(sfDir)

    ###################################
    ########### EXECUTION #############
    ###################################
    execSims()

    ###################################
    ######### POST-PROCESS ############
    ###################################
    for sf in meshSF:
        sfDir = f'{studyDir}/{sf}'

        ######## Objective 2: Position of vortex #########
        # sort dump files to find latest mesh and solution files
        folder = sfDir + '/dump'
        litems = os.listdir(folder)
        litems.sort()
        for i in range(len(litems)):
            if litems[i].endswith('.mesh.h5'):
                latestMesh = litems[i]
            if litems[i].endswith('.sol.h5'):
                latestSoln = litems[i]

        # extract location (x_coor, y_coor) of max and min vorticity
        with h5py.File(sfDir + '/dump/' + latestSoln, 'r') as f:
            vort = f['Data']['VORTICITY'][:]
        with h5py.File(sfDir + '/dump/' + latestMesh, 'r') as f:
            coor = f['Coordinates']['XYZ'][:][:, :2]
        imax = np.argmax(vort)
        imin = np.argmin(vort)
        cart_vMaxCoor = coor[imax, :]
        cart_vMinCoor = coor[imin, :]

        print(f'mesh size factor: {sf}')
        print(cart_vMaxCoor, cart_vMinCoor)
        plot(sf, cart_vMinCoor, cart_vMaxCoor)

def plot(sf, vMin, vMax):
    plt.plot(sf, vMin[0], 'ro', vMax[0], 'b^')
    plt.xlabel('Mesh Size Factor')
    plt.ylabel('X Coordinates')
    plt.save(f'{studyDir}/vort-x.png')
    # plt.show()

    plt.plot(sf, vMin[1], 'ro', vMax[1], 'b^')
    plt.xlabel('Mesh Size Factor')
    plt.xlabel('Y Coordinates')
    plt.save(f'{studyDir}/vort-y.png')
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
    #     if os.path.exists(f'{sfDir}/solver01_rank0.log'):
    #         completeCount += 1
    # if completeCount == len(meshSF):
    #     return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for sf in meshSF:
        sfDir = f'{studyDir}/{sf}'
        if os.path.exists(f'{sfDir}/solver01_rank0.log'):
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


def editJobslurm(sfDir):
    # change jobslurm.sh to correct directory and change job name
    with open(sfDir + '/jobslurm.sh', 'r') as f_orig:
        job_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find('base_case')] + sfDir + '\n'
    job_lines[keyword_line_i] = newLine

    # find job-name line
    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'meshStudy' + '\n'
    job_lines[keyword_line_i] = newLine
    with open(sfDir + '/jobslurm.sh', 'w') as f_new:
        f_new.writelines(job_lines)


meshStudy()
'''
RESTART_TYPE = CART
CART_NX = 20 # number of cells in the x direction
CART_NY = 20 # number of cells in the y direction
CART_NZ = 20 # number of cells in the z direction
CART_X_MIN = -0.5 # minimum x of the nodes
CART_X_MAX = 0.5 # maximum x of the nodes
CART_Y_MIN = -0.5 # minimum y of the nodes
CART_Y_MAX = 0.5 # maximum y of the nodes
CART_Z_MIN = -0.5 # minimum z of the nodes
CART_Z_MAX = 0.5 # maximum z of the nodes
CART_X_RAND = 0.00 # random displacement of the x coordinate
CART_Y_RAND = 0.00 # random displacement of the y coordinate
CART_Z_RAND = 0.00 # random displacement of the z coordinate
CART_DX_RAND = 0.00 # random spacing of nodes in the x direction
CART_DY_RAND = 0.00 # random spacing of nodes in the y direction
CART_DZ_RAND = 0.00 # random spacing of nodes in the z direction
CART_Y_SKEWNESS = 00.0 # angle in degrees for the first skewness (2D and 3D)
CART_Z_SKEWNESS = 00.0 # angle in degrees for the second skewness (3D only)
CART_AXI_X = 0 # axi-symmetry in the x direction (=0 or 1)
CART_AXI_Y = 0 # axi-symmetry in the y direction (=0 or 1)
CART_AXI_X_ANGLE = 50 # angle of the axi-symmetry
CART_AXI_Y_ANGLE = 50 # angle of the axi-symmetry
CART_X_SINE_STRETCH = 0 # sine mesh stretching in the x direction
CART_Y_SINE_STRETCH = 0 # sine mesh stretching in the y direction
CART_Z_SINE_STRETCH = 0 # sine mesh stretching in the z direction
CART_X_TANH_STRETCH = 0 # tanh mesh stretching in the x direction
CART_Y_TANH_STRETCH = 0 # tanh mesh stretching in the y direction
CART_Z_TANH_STRETCH = 0 # tanh mesh stretching in the z direction
CART_X_NORMALIZED_COOR = 'x_norm_coor.txt' # normalized x coordinates of the nodes (between 0 and 1)
CART_Y_NORMALIZED_COOR = 'y_norm_coor.txt' # normalized y coordinates of the nodes (between 0 and 1)
CART_Z_NORMALIZED_COOR = 'z_norm_coor.txt' # normalized z coordinates of the nodes (between 0 and 1)
'''
