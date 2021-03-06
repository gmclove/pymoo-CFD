from distutils.dir_util import copy_tree
from subprocess import check_output
from multiprocessing import Process
from os import getcwd

import numpy as np


class RunYALES2:
    def __init__(self, x, gen):  #, procLim, nProc):
        self.x = x #np.array([[0, 1]])
        self.gen = gen
        # self.procLim = procLim
        # self.nProc = nProc

        self.exFile = '2D_cylinder'
        self.dataFile = 'ics_temporals.txt'

        self.genDir = './gen%i' % gen

        # create array for objectives
        numObj = 2  # number of objectives
        self.obj = np.zeros((len(self.x), numObj))

        self.preProc()
        self.executeSims()
        self.postProc()

    ####################################################################################################################
    def preProc(self):
        def findKeywordLine(kw, file_lines):
            kw_line = -1
            kw_line_i = -1

            for line_i in range(len(file_lines)):
                line = file_lines[line_i]
                if line.find(kw) >= 0:
                    kw_line = line
                    kw_line_i = line_i

            return kw_line, kw_line_i

        def testBaseCase():
            if self.gen == 0:
                out = check_output(['sbatch', './base_case/jobslurm.sh'])
                batchID = int(out[20:])
                # print(batchID)
                waiting = True
                while waiting:
                    out = check_output('squeue | grep --count %i || :' % batchID, shell=True)
                    # print(int(out))
                    if int(out) == 0:
                        waiting = False

        # MAIN BODY

        # test base case
        # if self.gen == 0:
        #     out = check_output(['sbatch', './base_case/jobslurm.sh'])
        #     batchID = int(out[20:])
        #     # print(batchID)
        #     waiting = True
        #     while waiting:
        #         out = check_output('squeue | grep --count %i || :' % batchID, shell=True)
        #         # print(int(out))
        #         if int(out) == 0:
        #             waiting = False

        # create folder for individuals from base case:
        # change parameters and adjust path of jobslurm.sh file
        for ind in range(len(self.x)):
            # Extract parameters for each individual
            para = self.x[ind, :]
            omega = para[0]
            freq = para[1]
            # copy base case files to new directory for each individual
            indDir = self.genDir + '/ind%i' % ind
            copy_tree('base_case', indDir)

            # see if simulation has already run
            # Best way:
            # check if individual already has correct values in .in file?
            #       requires addition check of output

            # change jobslurm.sh to correct directory and change job name
            with open(indDir + '/jobslurm.sh', 'r') as f_orig:
                job_lines = f_orig.readlines()
            # use keyword 'cd' to find correct line
            keyword = 'cd'
            keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
            # create new string to replace line
            newLine = keyword_line[:keyword_line.find('base_case')] + 'gen%i/ind%i' % (self.gen, ind) + '\n'
            job_lines[keyword_line_i] = newLine

            # find job-name line
            keyword = 'job-name='
            keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
            # create new string to replace line
            newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'g%i.i%i' % (self.gen, ind) + '\n'
            job_lines[keyword_line_i] = newLine
            with open(indDir + '/jobslurm.sh', 'w') as f_new:
                f_new.writelines(job_lines)

            ####### Simulation Boundary Condition Parameters ###########
            exDir = indDir + '/' + self.exFile + '.in'
            # open and read YALES2 input file to array of strings for each line
            with open(exDir, 'r') as f_orig:
                in_lines = f_orig.readlines()
            # find line that must change using a keyword
            keyword = 'CYL_ROTATION_PROP'
            keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
            # create new string to replace line
            newLine = keyword + ' = ' + str(omega) + ' ' + str(freq) + '\n'
            in_lines[keyword_line_i] = newLine
            with open(exDir, 'w') as f_new:
                f_new.writelines(in_lines)
            # REPEAT FOR EACH LINE THAT MUST BE CHANGED

            ######### Simulation Geometric Parameters ############

    ####################################################################################################################
    def executeSims(self):
        # Queue all the individuals in the generation using SLURM
        batchIDs = []  # collect batch IDs
        for ind in range(len(self.x)):
            # create string for directory of individuals job slurm shell file
            indDir = self.genDir + '/ind%i/jobslurm.sh' % ind
            out = check_output(['sbatch', indDir])
            # Extract number from following: 'Submitted batch job 1847433'
            # print(int(out[20:]))
            batchIDs.append(int(out[20:]))

        waiting = True
        count = np.ones(len(self.x))
        processes = []
        while waiting:
            for bID_i in range(len(batchIDs)):
                # grep for batch ID of each individual
                out = check_output('squeue | grep --count %i || :' % batchIDs[bID_i], shell=True)  # '|| :' ignores non-zero exit status error
                count[bID_i] = int(out)
                # if job batch number can not be found then start post-processing
                # if count[bID_i] == 0:
                #     self.postProc(bID_i)
                #     # Run post processing once simulation finishes
                #     # proc = Process(target=self.postProc(bID_i))
                #     # proc.start()
                #     # processes.append(proc)
            # check if all batch jobs are done
            if sum(count) == 0:
                # wait for post processing to complete
                # for proc in processes:
                #     proc.join()
                # end while loop
                waiting = False
            # print(count)
            # print('SUM OF COUNT = %i' % sum(count))

        # print('GEN%i: EXECUTING SIMULATION COMPLETE' % self.gen)

    ####################################################################################################################
    def postProc(self):
        for ind in range(len(self.x)):
            # print("POST-PROCESSING: gen%i/ind%i" % (self.gen, ind))
            # Extract parameters for each individual
            para = self.x[ind, :]
            omega = para[0]
            freq = para[1]
            ####### Extract data from case file ########
            # create string for directory of individual's data file
            indDir = self.genDir + '/ind%i/' % ind + self.dataFile
            data = np.genfromtxt(indDir, skip_header=1)
            # collect data after 8 seconds
            noffset = 8 * data.shape[0] // 10
            # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
            p_over_rho_intgrl_1 = data[noffset:, 4]
            tau_intgrl_1 = data[noffset:, 6]

            ######## Compute Objectives ##########
            # Objective 1: Drag on cylinder
            drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)

            # Objective 2: Power consumed by rotating cylinder
            D = 1  # [m] cylinder diameter
            t = 0.1  # [m] thickness of cylinder wall
            r_o = D/2  # [m] outer radius
            r_i = r_o-t  # [m] inner radius
            d = 2700  # [kg/m^3] density of aluminum
            L = 1  # [m] length of cylindrical tube
            V = L*np.pi*(r_o**2-r_i**2)
            m = d*V
            I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
            E = 0.5*I*omega**2  # [J] or [(kg m^2)/s^2] energy consumption at peak rotational velocity (omega)
            P_avg = E*4*freq  # [J/s] average power over 1/4 cycle

            # obj_i = [drag]
            # self.obj.append(obj_i)
            self.obj[ind] = [drag, P_avg]
            # print('GEN%i OBJECTIVE:' % ind)
            # print(self.obj)
    ####################################################################################################################
    # def postProc(self, ind):
    #     print("POST-PROCESSING: gen%i/ind%i" % (self.gen, ind))
    #     # for ind in range(len(self.x)):
    #     ####### Extract data from case file ########
    #     # create string for directory of individual's data file
    #     indDir = self.genDir + '/ind%i/' % ind + self.dataFile
    #     data = np.genfromtxt(indDir, skip_header=1)
    #     # collect data after 8 seconds
    #     noffset = 8 * data.shape[0] // 10
    #     # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
    #     p_over_rho_intgrl_1 = data[noffset:, 4]
    #     tau_intgrl_1 = data[noffset:, 6]
    #
    #     ######## Compute Objectives ##########
    #     drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)
    #
    #     # obj_i = [drag]
    #     # self.obj.append(obj_i)
    #     self.obj[ind] = drag
    #     print('GEN%i OBJECTIVE:' % ind)
    #     print(self.obj)

        # ADD clean up of generation file (i.e. remove unnecessary data)
