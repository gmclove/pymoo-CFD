from pymooIN import *

def execSims(batchDir, caseDir, n_sims):
    # check if meshStudy is already complete
    completeCount = 0
    for n in range(n_sims):
        simDir = f'{batchDir}/{caseDir}{n}'
        if os.path.exists(f'{simDir}/solver01_rank00.log'):
            completeCount += 1
    if completeCount == n_sims:
        return

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for n in range(n_sims):
        simDir = f'{batchDir}/{caseDir}{n}'
        if os.path.exists(f'{simDir}/{output_file}'):
            pass
        else:
            # create string for directory of individuals job slurm shell file
            jobDir = f'{simDir}/jobslurm.sh'
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

    print('BATCH OF SLURM SIMULATIONS COMPLETE')
