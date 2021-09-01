import os
import subprocess
import numpy as np
import time

from pymooCFD.setupOpt import solverExec, procLim, nProc

def singleNodeExec(path, subDirs): #, procLim=procLim, nProc=nProc, solverFile=solverFile):
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
    n_sims = len(subDirs)
    
    while n < n_sims:
        caseDir = f'{path}/{subDirs[n]}'
        # if os.path.exists(f'{caseDir}/obj.txt'): #solver01_rank00.log'):
        #     n +=1
        #     continue

        if currentP < procLim: # currentP != procLim:
            print(f'## Sending {caseDir} to simulation...')
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

    
def slurmExec(dir, subdir, n_sims):
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
    for n in range(n_sims):
        caseDir = f'{dir}/{subdir}{n}'
        if os.path.exists(f'{caseDir}/solver01_rank00.log'):
            pass
        else:
            # create string for directory of individuals job slurm shell file
            jobDir = f'{caseDir}/jobslurm.sh'
            out = subprocess.check_output(['sbatch', jobDir])
            # Extract number from following: 'Submitted batch job 1847433'
            # print(int(out[20:]))
            batchIDs.append(int(out[20:]))
    # print(batchIDs)

    waiting = True
    count = np.ones(len(batchIDs))
    # processes = []
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

    print('BATCH OF SLURM SIMULATIONS COMPLETE')
