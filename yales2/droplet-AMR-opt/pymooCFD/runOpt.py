# import numpy as np
# import os
from pymooCFD.setupOpt import dataDir, setAlgorithm

def runOpt(restart=True):
    if restart:
        # try:
        from pymooCFD.util.handleData import loadCP
        algorithm = loadCP()
        print("Loaded Checkpoint:", algorithm)
        print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        setAlgorithm(algorithm)
            
    else:
        print('STARTING NEW OPTIMIZATION STUDY')
        # load algorithm initialize in setupOpt.py module
        from pymooCFD.setupOpt import algorithm
        setAlgorithm(algorithm)
        # archive previous runs data directory
        from pymooCFD.util.sysTools import emptyDir
        emptyDir(dataDir)

    ########################################################################################################################
    ######    OPTIMIZATION    ######
    from pymoo.optimize import minimize
    from pymooCFD.setupOpt import problem, callback, display, n_gen #, n_workers # , MyDisplay  # , termination

    # from dask_jobqueue import SLURMCluster
    # cluster = SLURMCluster()
    # cluster.scale(jobs=algorithm.pop_size)    # Deploy ten single-node jobs

    # from dask.distributed import LocalCluster
    # cluster = LocalCluster(n_workers = 10, processes = True)

    # from dask.distributed import Client  # , LocalCluster 
    # client = Client(cluster(n_workers=n_workers))
    # problem = problem(client)

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   # termination=termination,
                   termination=('n_gen', n_gen),
                   callback=callback,
                   display=display,
                   # display=MyDisplay(),
                   seed=1,
                   copy_algorithm=True,
                   # pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   verbose=True
                   )
    # client.close()
    # np.save("checkpoint", algorithm)
    print("EXEC TIME: %.3f seconds" % res.exec_time)









        # from pymooCFD.util.handleData import loadTxt
            # algorithm = loadTxt('dump/gen13X.txt', 'dump/gen13F.txt')
        # except OSError as err:
        #     print(err)
        #     from pymooCFD.setupOpt import checkpointFile
        #     print(f'{checkpointFile} load failed.')
        #     print('RESTART FAILED')
        #     return
        
        
        
        
        
        

        # try:
        #     os.remove(f'{dataDir}/obj.npy')
        # except OSError as err:
        #     print(err)
        # try:
        #     checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
        #     print("Loaded Checkpoint:", checkpoint)
        #     # only necessary if for the checkpoint the termination criterion has been met
        #     checkpoint.has_terminated = False
        #     algorithm = checkpoint
        #     print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        # except OSError as err:
        #     print(err)
        #     from pymooCFD.setupOpt import algorithm
            # try:
            #     X = np.loadtxt('var.txt')
            #     F = np.loadtxt('obj.txt')
            #     from pymoo.model.evaluator import Evaluator
            #     from pymoo.model.population import Population
            #     from pymoo.model.problem import StaticProblem
            #
            #     # now the population object with all its attributes is created (CV, feasible, ...)
            #     pop = Population.new("X", X)
            #     pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)
    ########################################################################################################################
    ######    PARALLELIZE    ######
    # from pymooCFD.setupOpt import GA_CFD, n_parallel_proc
    # if parallel.lower() == 'multiproc':
    #     ###### Multiple Processors ######
    #     # parallelize across multiple CPUs on a single machine (node)
    #     # NOTE: only one CFD simulation per CPU???
    #     import multiprocessing
    #     # the number of processes to be used
    #     n_proccess = n_parallel_proc
    #     pool = multiprocessing.Pool(n_proccess)
    #     problem = GA_CFD(parallelization=('starmap', pool.starmap))
    # elif any(selection.lower() == parallel.lower() for selection in multiProcSelect):
    #     from pymooCFD.setupCFD import runCFD
    #     ###### Multiple Nodes ######
    #     # Use Dask library to parallelize across
    #     ### SLURM Cluster ###
    #     from dask_jobqueue import SLURMCluster
    #     cluster = SLURMCluster()
    #     # ask to scale to a certain number of nodes
    #     cluster.scale(jobs=n_parallel_proc)  # Deploy n_parallel_proc single-node jobs
    #     ### Launch Client ###
    #     from dask.distributed import Client
    #     client = Client(cluster)
    #     problem = GA_CFD(parallelization=("dask", client, runCFD))
    #     pool = client
    # else:
    #     ###### Multiple Threads ######
    #     # Run on multiple threads within a single CPU
    #     from multiprocessing.pool import ThreadPool
    #     # the number of threads to be used
    #     n_threads = n_parallel_proc
    #     # initialize the pool
    #     pool = ThreadPool(n_threads)
    #     # define the problem by passing the starmap interface of the thread pool
    #     problem = GA_CFD(parallelization=('starmap', pool.starmap))
