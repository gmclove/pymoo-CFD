import numpy as np
import os
# import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree
import subprocess
import glob

from pymooCFD.util.sysTools import makeDir
from pymooCFD.setupOpt import baseCaseDir, var_labels, inputFile


def preProc(caseDir, var):  # , jobName=jobName, jobFile=jobFile):
    """
    | preProc(caseDir, var, jobName=jobName, jobFile=jobFile)
    |
    |   CFD pre-process function. Edits CFD case input variables.
    |
    |   Parameters
    |   ----------
    |   caseDir : string
    |       CFD case directory containing input files used by CFD solver.
    |   var: list (or numpy array)
    |       List of variables typically generated my pyMOO and passed to CFD
    |       pre-process function which edits the case inputs.
    |
    |   Returns
    |   -------
    |   None
    |       Typically this function does not need to return anything because
    |       it's purpose is edit the CFD case input files.
    """
    # print(f'PRE-PROCESSING CFD CASE: {caseDir}')
    makeDir(caseDir)
    
    #### EXTERNAL CFD SOLVER ####
    # copy contents of base case into 
    copy_tree(baseCaseDir, caseDir)
  

    #### SLURM PRE-PROCESS #####
    # since we are using slurm to exectute our simulations we must edit
    # the file used to lauch our job
    # editJobslurm(gen, ind, caseDir)
    # editSlurmJob(caseDir, jobName=jobName, jobFile=jobFile)

    ####### EXTRACT VAR ########
    # Extract parameters for each individual
    thres = var[var_labels.index('Threshold')]
    prop = var[var_labels.index('Propagation Steps')]
    max_steps = var[var_labels.index('Max. Number of Steps')]

    ####### Simulation Input Parameters ###########
    inputPath = os.path.join(caseDir, inputFile)
    # open and read YALES2 input file to array of strings for each line
    with open(inputPath, 'r') as f_orig:
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
    
    # find line that must change using a keyword
    keyword = 'ADAPTATION_NSTEPS_MAX'
    keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find('=')+2] + str(max_steps) + '\n'
    in_lines[keyword_line_i] = newLine
    # REPEAT FOR EACH LINE THAT MUST BE CHANGED

    with open(inputPath, 'w') as f_new:
        f_new.writelines(in_lines)


def postProc(caseDir, var):
    '''
    | postProc(caseDir, var)
    |
    |   CFD pre-process function. Edits CFD case input variables.
    |
    |   Parameters
    |   ----------
    |   caseDir : string
    |       CFD case directory containing output files used by CFD solver.
    |   var : list (or numpy array)
    |       List of variables typically generated my pyMOO and passed to CFD
    |       pre-process function which edits the case inputs. Now in
    |       post-process fuctiom this values are sometime needed to compute
    |       objectives but not always.
    |
    |   Returns
    |   -------
    |   obj : list of objectives
    |       Objectives extracted from the CFD case in post-processing
    '''
    # print(f'POST-PROCESSING CFD CASE: {caseDir}')
    
    ####### EXTRACT VAR ########
    # OPTIONAL: Extract parameters for each individual
    # sometimes variables are used in the computation of the objectives
    
    ######## Compute Objectives ##########
    ######## Objective 1: Simulation Time #########
    # sim_time should be on the 11th to last line (grep last 20 to be safe)
    log, = glob.glob(os.path.join(caseDir, 'solver01_rank*.log'))
    cmd = f'tail -n 11 {log} | grep "WALL CLOCK TIME"'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    time = process.communicate()[0][-9:]  # extract time value in last 9 characters of line
    print(time)
    sim_time = float(time)

    ######## Objective 2: Match Shape and Location of Droplet #########
    # import instance of GridInterp class create during in preProcOpt module
    # an object is creared in preProcOpt to ensure interpolation parameters 
    # are conistent
    from pymooCFD.preProcOpt import gridInterp
    amr_grids = gridInterp.getGrids(caseDir)
    # Load high quality mesh grid
    hq_grids = np.load('hq_grids')
    
    meanDiff = gridInterp.meanDiff(hq_grids, amr_grids)
    
    obj = [sim_time, meanDiff]
    ###### SAVE VARIABLES AND OBJECTIVES TO TEXT FILES #######
    # save variables in case directory as text file after completing pre-processing
    saveTxt(caseDir, 'var.txt', var)
    # save objectives in text file
    saveTxt(caseDir, 'obj.txt', obj)
    return obj


###############################################################################
###### FUNCTIONS ######
#######################
def runCase(caseDir, x):
    print(f'Running Case {caseDir}: {x}')
    if completed(caseDir, x):
        return 
    else:
        preProc(caseDir, x)
        from pymooCFD.setupOpt import solverExec  # , nProc
        # cmd = f'cd {caseDir} && mpirun -np {nProc} {solverExec} > output.dat'
        cmd = f'cd {caseDir} && mpirun {solverExec} > output.dat'
        os.system(cmd)
        obj = postProc(caseDir, x)
        return obj
    
def runGen(genDir, X):
    """
    Run an entire generation at a time. Used when working with singleNode or 
    slurmBatch modules found in the sub-package execSimsBatch. 
    """
    preProcGen(genDir, X)
    from pymooCFD.execSimsBatch.singleNode import execSims
    execSims(genDir, 'ind', len(X))
    obj = postProcGen(genDir, X)
    return obj

def preProcGen(genDir, X):
    for ind in range(len(X)):
        indDir = os.path.join(genDir, f'ind{ind + 1}')
        # check if individual has already completed exection and postProc
        checkCompletion(X[ind], indDir)
        # Create directory is it does not exist
        makeDir(indDir)
        preProc(indDir, X[ind])
 
def postProcGen(genDir, X):
    obj = [] #np.zeros((len(X), n_obj))
    for ind in range(len(X)):
        indDir = os.path.join(genDir, f'ind{ind + 1}')
        obj_ind = postProc(indDir, X)
        obj.append(obj_ind)
    return obj

def checkCompletion(var, caseDir):
    ###### Check Completion ######
    global varFile, objFile
    # load in previous variable file if it exist and
    # check if it is equal to current variables
    varFile = os.path.join(caseDir, 'var.txt')
    objFile = os.path.join(caseDir, 'obj.txt')
    if os.path.exists(varFile) and os.path.exists(objFile):
        try:
            prev_var = np.loadtxt(varFile)
            if np.array_equal(prev_var, var):
                print(f'{caseDir} already complete')
                return
        except OSError as err:
            print(err)

def completed(caseDir, var):
    ###### Check Completion ######
    # global varFile, objFile
    # load in previous variable file if it exist and
    # check if it is equal to current variables
    varFile = os.path.join(caseDir, 'var.txt')
    objFile = os.path.join(caseDir, 'obj.txt')
    if os.path.exists(varFile) and os.path.exists(objFile):
        try:
            prev_var = np.loadtxt(varFile)
            if np.array_equal(prev_var, var):
                print(f'{caseDir} already complete')
                return True
        except OSError as err:
            print(err)
            return False
    else:
        return False

# def dataMatches(fname, dat):
#     if 
    

def saveTxt(path, fname, data):
    datFile = os.path.join(path, fname)
    # save data as text file in directory  
    np.savetxt(datFile, data)


def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


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

