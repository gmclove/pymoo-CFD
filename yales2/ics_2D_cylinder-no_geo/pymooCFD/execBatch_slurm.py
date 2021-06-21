def execSims(gen, n_sims):
    genDir = f'../gen{gen}'

    # Queue all the individuals in the generation using SLURM
    batchIDs = []  # collect batch IDs
    for ind in range(n_sims):
        # create string for directory of individuals job slurm shell file
        indDir = genDir + '/ind%i/jobslurm.sh' % ind
        if os.path.exists(indDir+'solver01_rank00.log'):
            out =   subprocess.check_output(['sbatch', indDir])
            # Extract number from following: 'Submitted batch job 1847433'
            # print(int(out[20:]))
            batchIDs.append(int(out[20:]))
        else:
            pass
    # print(batchIDs)

    waiting = True
    count = np.ones(n_sims)
    processes = []
    while waiting:
        for bID_i in range(len(batchIDs)):
            # grep for batch ID of each individual
            out = subprocess.check_output('squeue | grep --count %i || :' % batchIDs[bID_i], shell=True)  # '|| :' ignores non-zero exit status error
            count[bID_i] = int(out)
            # if job batch number can not be found then start post-processing
            # if count[bID_i] == 0:
            #     postProc(bID_i)
            #     # Run post processing once simulation finishes
            #     # proc = Process(target=postProc(bID_i))
            #     # proc.start()
            #     # processes.append(proc)

        # print(count)
        # check if all batch jobs are done
        if sum(count) == 0:
            # wait for post processing to complete
            # for proc in processes:
            #     proc.join()
            # end while loop
            waiting = False
        # print(count)
        # print('SUM OF COUNT = %i' % sum(count))
        time.sleep(1)


    # print('GEN%i: EXECUTING SIMULATION COMPLETE' % gen)
