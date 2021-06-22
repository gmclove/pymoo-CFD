# =============================================================================
# import os
# import subprocess
# 
# 
# class ExecuteSimulations:
# 
#     def __init__(self, x, gen, solver, procLim, nProc):
#         self.x = x
#         self.gen = gen
#         self.solver = solver
#         self.nProc = nProc
#         self.procLim = procLim
# 
#     def run(self):
#         if self.nProc > 1:
#             self.parallelRun()
#         else:
#             self.serialRun()
# 
#     def wait(self, pids):
#         print('WAITING')
#         for pid in pids:
#             #pid.wait()
#             print(pid)
#             #os.kill(pid, 0)
#             os.waitpid(pid, 0)
# 
#     ###### Run individuals in parallel ######
#     def parallelRun(self):
#         ###################### FUNCTIONS ##################################
#         ##### Decompose mesh #####
#         def decompMesh():
#             # If parallel computing is desired, the mesh must be first decomposed
#             for ind in range(len(self.x)):
#                 # print('.## decomposePar for individual %i ...' % ind)
#                 # os.system('cd cases/gen%i/ind%i \n decomposePar', %(self.gen, self.ind))
#                 os.system('decomposePar -case ind%i' % ind)
#                 # os.system('decomposePar -case ind%i > DPg%ii%i 2>&1' %(ind, gen, ind))
#                 # os.system('mv DPg%ii%i ind%i/DPg%ii%i' %(gen, ind, gen, ind))
# 
#         # Recompose mesh for each individual
#         def recompMesh():
#             # If parallel computing is desired, the mesh must be reconstructed after
#             for ind in range(len(self.x)):
#                 # print('.## reconstructPar for individual %i ...' %ind)
#                 os.system('reconstructPar -case ind%i' % ind)
#                 # os.system('reconstructPar -case ind%i > RPg%ii%i 2>&1 &' %(ind, self.gen, ind))
#                 # mv RPg"$gen"i"$ind" ind$ind/RPg"$gen"i"$ind"
# 
#         ###########################################################################################
#         ########################## MAIN BODY #################################################
#         # print('PARALLEL RUN')
#         # Move to the generation folder
#         ogDir = os.getcwd()  # original directory
#         os.chdir('./cases/gen%i' % self.gen)  # os.mkdir()
# 
#         # If parallel computing is desired, the mesh must be first decomposed
#         decompMesh()
# 
#         # All processors will be queued until all are used
#         ind = 0
#         currentP = 0
#         pids = []
# 
#         while ind < len(self.x):
#             if currentP != self.procLim:  # currentP < procLim:
#                 # Send another simulation
#                 pid = subprocess.Popen(
#                     ['mpirun', '-np', str(self.nProc), self.solver, '-case', 'ind%i' % ind, '-parallel'])
#                 # os.system('mpirun -np %i %s -case ind%i -parallel > RUNg%ii%i 2>&1 &'
#                 #        %(self.nProc, self.solver, ind, self.gen, ind))
#                 # Store the PID of the above process
#                 pids.append(pid)
#                 # print('## Sending ind%i to simulation...' %ind)
#                 # counters
#                 ind += 1
#                 currentP = currentP + self.nProc
#             # Then, wait until completion and fill processors again
#             else:
#                 # Wait until all PID in the list has been completed
#                 self.wait(pids)
#                 # Delete the current array with all PID
#                 pids.clear()
#                 # Restart the number of processors currently in use
#                 currentP = 0
# 
#         # Wait until all PID in the list has been completed
#         self.wait(pids)
#         ##### Recompose mesh #####
#         recompMesh()
#         # Return to main directory
#         os.chdir(ogDir)
# 
#     # run individual's case
#     def serialRun(self):
#         # print('SERIAL RUN')
#         ##### Serial individual computing #####
#         # Move to the generation folder
#         ogDir = os.getcwd()
#         os.chdir('./cases/gen%i' % self.gen)
#         print(os.getcwd())
# 
#         # All processors will be queued until all are used
#         pids = []
#         ind = 0
#         currentP = 0
#         while ind < len(self.x):
#             # If all processors are not in use yet
#             if currentP != self.procLim:  # currentP < procLim:
#                 # Send another simulation
#                 #pid = subprocess.Popen([os.getcwd(), self.solver, '-case', 'ind%i' % ind], shell=True).pid
#                 pid = subprocess.Popen([os.getcwd(), self.solver, '-case', 'ind%i' % ind], shell=True).pid
#                 # os.system('%s -case ind%i > RUNg%ii%i 2>&1 &' %(self.solver, ind, self.gen, ind))
#                 # Store the PID of the above process
#                 pids.append(pid)
#                 # counters
#                 currentP = currentP + self.nProc
#                 ind += 1
#             # Then, wait until completion and fill processors again
#             else:
#                 # Wait until all PID in the list has been completed
#                 self.wait(pids)
#                 # Delete the current array with all PID
#                 pids.clear()
#                 # Restart the number of processors currently in use
#                 currentP = 0
# 
#         # Wait until all PID in the list has been completed
#         self.wait(pids)
# 
#         # Return to main directory
#         os.chdir(ogDir)
# 
# =============================================================================

import os
import subprocess
from pymooCFD.setupOpt import *

def execSims(dir, subdir, n_sims): #, procLim=procLim, nProc=nProc, solverFile=solverFile):
    print('EXECUTING BATCH OF SIMULATIONS')
    def wait(pids):
       print('WAITING')
       for pid in pids:
           # print(pid)
           pid.wait()
           #os.kill(pid, 0)
           # os.waitpid(pid, 0)
    # All processors will be queued until all are used
    n = 0
    currentP = 0
    pids = []

    while n < n_sims:
        caseDir = f'{dir}/{subdir}{n}'
        if os.path.exists(f'{caseDir}/solver01_rank00.log'):
            n +=1
            continue
            
        if currentP < procLim: # currentP != procLim: 
            print(f'## Sending {subdir}{n} to simulation...')
            # cmd = solverExec
            cmd = f'cd {caseDir} && mpirun -np {nProc} {solverExec} > output.dat'
            # Send another simulation
            pid = subprocess.Popen(cmd, shell=True)
            # Store the PID of the above process
            pids.append(pid)
            # counters
            n += 1
            currentP = currentP + nProc
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

    print('BATCH OF SIMULATIONS COMPLETE')
