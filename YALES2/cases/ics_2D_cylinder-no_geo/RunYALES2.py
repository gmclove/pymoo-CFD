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

        self.caseDir = './YALES2/cases/ics_2D_cylinder'
        self.genDir = self.caseDir + '/gen%i' % gen

        # create array for objectives
        self.obj = []

        self.preProc()
        self.executeSims()

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

        # MAIN BODY
        for ind in range(len(self.x)):
            # Extract parameters for each individual
            para = self.x[ind, :]
            omega = para[0]
            freq = para[1]
            # copy base case files to new directory for each individual
            indDir = self.genDir + '/ind%i' % ind
            copy_tree(self.caseDir + '/base-case', indDir)

            # change jobslurm.sh to correct directory
            with open(indDir + '/jobslurm.sh', 'r') as f_orig:
                job_lines = f_orig.readlines()
            # find cd line
            keyword = 'cd'
            keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
            # create new string to replace line
            newLine = keyword_line[:keyword_line.find('base-case')] + 'gen%i/ind%i' % (self.gen, ind) + '\n'
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
        batchID = []  # collect batch IDs
        for ind in range(len(self.x)):
            # create string for directory of individuals job slurm shell file
            indDir = self.genDir + '/ind%i/jobslurm.sh' % ind
            # cmd = 'sbatch ' + indDir
            out = check_output(['sbatch', indDir])#cmd)
            # Extract number from following: 'Submitted batch job 1847433'
            print(int(out[20:]))
            batchID.append(int(out[20:]))

        waiting = True
        count = []
        processes = []
        while waiting:
            for bID_i in range(len(batchID)):
                # grep for batch ID of each individual
                out = check_output('squeue | grep --count %i' % int(batchID[bID_i]))
                count.append(int(out))
                if int(count[bID_i]) == 0:
                    # Run post processing once simulation finishes
                    proc = Process(target=self.postProc(bID_i))
                    proc.start()
                    processes.append(proc)
            # check if all batch jobs are done
            if sum(count) == 0:
                # wait for post processing to complete
                for proc in processes:
                    proc.join()
                # end while loop
                waiting = False

    ####################################################################################################################
    def postProc(self, ind):
        # for ind in range(len(self.x)):
        ####### Extract data from case file ########
        # create string for directory of individual's data file
        indDir = self.genDir + '/ind%i' + self.dataFile % ind
        data = np.genfromtxt(indDir, skip_header=1)
        # collect data after 8 seconds
        noffset = 8 * data.shape[0] // 10
        # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
        p_over_rho_intgrl_1 = data[noffset:, 4]
        tau_intgrl_1 = data[noffset:, 6]

        ######## Compute Objectives ##########
        drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)

        obj_i = [drag]
        self.obj.append(obj_i)

        # ADD clean up of generation file (i.e. remove unnecessary data)
