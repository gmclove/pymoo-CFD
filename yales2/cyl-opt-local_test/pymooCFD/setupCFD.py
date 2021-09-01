from pymooCFD.setupOpt import *
from scipy.integrate import quad
# import os

def preProc(caseDir, var): #, jobName=jobName, jobFile=jobFile):
    '''
    | preProc(caseDir, var, jobName=jobName, jobFile=jobFile)
    |
    |   CFD pre-process function. Edits CFD case input variables.
    |
    |   Parameters
    |   ----------
    |   caseDir : string
    |       CFD case directory containing input files used by CFD solver.
    |   var : list (or numpy array)
    |       List of variables typically generated my pyMOO and passed to CFD
    |       pre-process function which edits the case inputs.
    |
    |   Returns
    |   -------
    |   None
    |       Typically this function does not need to return anything because
    |       it's purpose is edit the CFD case input files.
    '''
    print(f'PRE-PROCESSING CFD CASE: {caseDir}')
    # load in previous variable file if it exist and
    # check if it is equal to current variables being sent into pre-process
    varFile = f'{caseDir}/var.txt'
    if os.path.exists(varFile):
        prev_var = np.loadtxt(varFile)
        if np.array_equal(prev_var, var):
            return
    # since we are using slurm to exectute our simulations we must edit
    # the file used to lauch our job
    # editJobslurm(gen, ind, caseDir)
    # editSlurmJob(caseDir, jobName=jobName, jobFile=jobFile)

    # Extract parameters for each individual
    amp = var[var_labels.index('Amplitude')]
    freq = var[var_labels.index('Frequency')]

    ####### Simulation Input Parameters ###########
    inputDir = f'{caseDir}/2D_cylinder.in'
    # open and read YALES2 input file to array of strings for each line
    with open(inputDir, 'r') as f_orig:
        in_lines = f_orig.readlines()

    # find line that must change using a keyword
    keyword = 'CYL_ROTATION_PROP'
    keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
    # create new string to replace line
    newLine = f'{keyword_line[:keyword_line.index("=")]} {amp} {freq} \n'
    in_lines[keyword_line_i] = newLine
    # REPEAT FOR EACH LINE THAT MUST BE CHANGED

    with open(inputDir, 'w') as f_new:
        f_new.writelines(in_lines)

    # save variables in case directory as text file after completing pre-processing
    np.savetxt(f'{caseDir}/var.txt', var)


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
    print(f'POST-PROCESSING CFD CASE: {caseDir}')
    # Extract parameters for each individual
    amp = var[var_labels.index('Amplitude')]
    freq = var[var_labels.index('Frequency')]
    ######## Compute Objectives ##########
    ######## Objective 1: Drag on Cylinder #########
    U = 1
    rho = 1
    D = 1
    # create string for directory of individual's data file
    dataDir = f'{caseDir}/ics_temporals.txt'
    # data = np.genfromtxt(dataDir, skip_header=1)
    try:
        data = np.genfromtxt(dataDir, skip_header=1)
    except IOError as err:
        print(err)
        print('ics_temporals.txt does not exist')
        obj = [None] * n_obj
        return obj

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
    P_cyc = 0.5*I*quad(lambda t : (amp*np.sin(t))**2, 0, 2*np.pi)[0]*freq  # [Watt]=[J/s] average power over 1 cycle

    obj = [C_drag, P_cyc]
    normalize(obj)
    np.savetxt(f'{caseDir}/obj.txt', obj)
    return obj


# def editSlurmJob(caseDir, jobFile=jobFile, jobName=jobName):
#     # fJob = kwargs.get('jobFile', jobFile) # if no jobFile give take from imported setupOpt module
#     file = f'{caseDir}/{jobFile}'
#     # change jobslurm.sh to correct directory and change job name
#     with open(file, 'r') as f_orig:
#         job_lines = f_orig.readlines()

#     # use keyword 'cd' to find correct line
#     keyword = 'cd'
#     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
#     # create new string to replace line
#     # newLine = keyword_line[:keyword_line.find('base_case')] + 'gen%i/ind%i' % (gen, ind) + '\n'
#     newLine = f"cd {os.getcwd()}/{caseDir.lstrip('./')} \n"
#     job_lines[keyword_line_i] = newLine

#     # if kwargs.has_key('jobName'):
#     # find job-name line
#     keyword = 'job-name='
#     keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
#     # create new string to replace line
#     # nJob = kwargs.get('jobName', jobName)
#     newLine = f'{ keyword_line[:keyword_line.find(keyword)] }{ keyword }{ jobName }\n'
#     job_lines[keyword_line_i] = newLine

#     with open(file, 'w') as f_new:
#         f_new.writelines(job_lines)

def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i

# def getVar():
#     para = x[ind, :]
#     amp = para[0]
#     freq = para[1]
#     return amp, freq
