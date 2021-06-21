import os
from distutils.dir_util import copy_tree
import subprocess
import numpy as np

from pymooIN import *

class RunYALES2:
    def __init__(self, x, gen, procLim, nProc):
        self.procLim = procLim  # Maximum processors to be used
        self.nProc = nProc  # Number of processors for each individual (EQUAL or SMALLER than procLim)

        self.x = x #np.array([[0, 1]])
        self.gen = gen

        self.exFile = '2D_cylinder'
        self.dataFile = 'ics_temporals.txt'

        self.caseDir = os.getcwd()
        self.genDir = self.caseDir + '/gen%i' % gen

        # create array for objectives
        numObj = 2  # number of objectives
        self.obj = np.zeros((len(self.x), numObj))

        self.preProc()
        self.executeSims()
        self.postProc()

    ####################################################################################################################
    def preProc(self):

        def testBaseCase():
            if self.gen == 0:
                print('TESTING BASE CASE')
                # recompile
                os.system('cd ' + indDir + '\n rm -fv' + self.dataFile + '\n make clean \n make')
                # run
                pid = subprocess.Popen(
                    ['cd', indDir, '\n', 'mpirun', '-np', str(self.nProc), self.exFile])
                pid.wait()

                out = subprocess.check_output(['sbatch', './base_case/jobslurm.sh'])
                batchID = int(out[20:])
                # print(batchID)
                # wait for slurm job batch to complete
                waiting = True
                while waiting:
                    out = subprocess.check_output('squeue | grep --count %i || :' % batchID, shell=True)
                    # print(int(out))
                    if int(out) == 0:
                        waiting = False
                if os.path.exists('base_case/ics_temporals.txt') is False:
                    print('ics_temporals.txt not created')
                    print('EXITING')
                    exit()
                print('BASE CASE SUCCESSFUL')

        def gen0MeshConv(ind):
            for ms in meshSizes:
                genMesh(indDir, self.x[ind, self.geoVars], ms)




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
        # testBaseCase()

        # first generation Pre-Processing to base_case
        if self.gen == 0:
            # recompile base_case
            os.system('cd base_case/; rm -vrf dump/ output.dat; make clean; make')

        # create folder for individuals from base case:
        # change parameters and adjust path of jobslurm.sh file
        for ind in range(len(self.x)):
            # if self.gen == 0:
            #     gen0MeshConv(ind)
            # Extract parameters for each individual
            para = self.x[ind, :]
            C_D = para[0]
            omega = para[1]
            freq = para[2]
            # copy base case files to new directory for each individual
            indDir = self.genDir + '/ind%i' % ind
            copy_tree('base_case', indDir)

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
            # if geoVar is not None:
            from genMesh import genMesh
            indMeshDir = indDir + '/meshes'
            genMesh(indMeshDir, C_D)  # re-mesh domain with cylinder diameter parameter

    ####################################################################################################################
    def executeSims(self):
        # print('EXECUTING SIMULATIONS: gen%i' % self.gen)
        def wait(pids):
            print('WAITING')
            print(pids)
            for pid in pids:
                # print(pid)
                pid.wait()
                # os.kill(pid, 0)
                # os.waitpid(pid, 0)
            print('done waiting')

        # All processors will be queued until all are used
        ind = 0
        currentP = 0
        pids = []

        if self.nProc > 1:
            ###########################################################################################
            ########################## MAIN BODY #################################################
            # print('PARALLEL RUN')
            while ind < len(self.x):
                if currentP != self.procLim:  # currentP < procLim:
                    # print('## Sending ind%i to simulation...' % ind)
                    indDir = self.genDir + '/ind%i' % ind
                    # print(indDir)
                    # Send another simulation
                    cmd = 'cd ' + indDir + '; mpirun -np ' + str(self.nProc) + ' ' + self.exFile
                    pid = subprocess.Popen(cmd, shell=True)
                    # Store the PID of the above process
                    pids.append(pid)
                    # counters
                    ind += 1
                    currentP = currentP + self.nProc
                # Then, wait until completion and fill processors again
                else:
                    # Wait until all PID in the list has been completed
                    wait(pids)
                    # Delete the current array with all PID
                    pids.clear()
                    # Restart the number of processors currently in use
                    currentP = 0

            # Wait until all PID in the list has been completed
            wait(pids)

        else:
            pass
            ##### Serial individual computing #####
            # print('SERIAL RUN')
            while ind < len(self.x):
                # If all processors are not in use yet
                if currentP != self.procLim:  # currentP < procLim:
                    indDir = self.genDir + '/ind%i' % ind
                    # Send another simulation
                    # pid = subprocess.Popen([os.getcwd(), self.solver, '-case', 'ind%i' % ind], shell=True).pid
                    cmd = 'cd ' + indDir + '; mpirun -np ' + str(self.nProc) + ' ' + self.exFile
                    pid = subprocess.Popen(cmd, shell=True)
                    # Store the PID of the above process
                    pids.append(pid)
                    # counters
                    currentP = currentP + self.nProc
                    ind += 1
                # Then, wait until completion and fill processors again
                else:
                    # Wait until all PID in the list has been completed
                    wait(pids)
                    # Delete the current array with all PID
                    pids.clear()
                    # Restart the number of processors currently in use
                    currentP = 0

            # Wait until all PID in the list has been completed
            wait(pids)

    ####################################################################################################################
    def postProc(self):
        # improve by vectorizing instead of for loop
        for ind in range(len(self.x)):
            # print("POST-PROCESSING: gen%i/ind%i" % (self.gen, ind))
            # Extract parameters for each individual
            para = self.x[ind, :]
            C_D = para[0] # [m] cylinder diameter
            omega = para[1]
            freq = para[2]
            ####### Extract data from case file ########
            # create string for directory of individual's data file
            indDir = self.genDir + '/ind%i/' % ind + self.dataFile
            data = np.genfromtxt(indDir, skip_header=1)
            # collect data after 100 seconds
            noffset = 100 * data.shape[0] // 10
            # extract P_OVER_RHO_INTGRL_(1) and TAU_INTGRL_(1)
            p_over_rho_intgrl_1 = data[noffset:, 4]  # \int_{0}^{2\pi}{P/\rho}
            tau_intgrl_1 = data[noffset:, 6]  # \int_{0}^{2\pi}{\tau}

            ######## Compute Objectives ##########
            # Objective 1: Drag on cylinder
            drag = np.mean(p_over_rho_intgrl_1 - tau_intgrl_1)  # coefficient unitless? [kN] drag

            # Objective 2: Power consumed by rotating cylinder
            t = 0.1*C_D  # [m] thickness of cylinder wall
            r_o = C_D/2  # [m] outer radius
            r_i = r_o-t  # [m] inner radius
            d = 2700  # [kg/m^3] density of aluminum
            L = 1  # [m] length of cylindrical tube
            V = L*np.pi*(r_o**2-r_i**2)  # [m^3] Volume of the cylindrical tube
            m = d*V  # [kg] Mass of the cylindrical tube
            I = 0.5*m*(r_i**2+r_o**2)  # [kg m^2] moment of inertia of a hollow cylinder
            E = 0.5*I*omega**2  # https://idp.uvm.edu/idp/profile/SAML2/Redirect/SSO?execution=e1s1&_eventId_proceed=1[J] or [(kg m^2)/s^2] energy consumption at peak rotational velocity (omega)
            P_avg = E*4*freq*1e6  # [micro-J/s] average power over 1/4 cycle

            # obj_i = [drag]
            # self.obj.append(obj_i)
            # print('drag: ' + str(drag))
            # print('P_avg: ' + str(P_avg))
            self.obj[ind] = [drag, P_avg]
            # print('GEN%i OBJECTIVE:' % ind)
            # print(self.obj)
