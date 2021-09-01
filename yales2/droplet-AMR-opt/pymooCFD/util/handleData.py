from pymooCFD.setupOpt import checkpointFile, dataDir, nCP, archDir, \
    preProcDir, cluster
from pymooCFD.util.sysTools import removeDir #, makeDir, emptyDir
from pymooCFD.setupCFD import runCase


import numpy as np
import time
import os
import tarfile
from dask.distributed import Client

from sys import exit


# def getGen(checkpointFile=checkpointFile):
#     try: 
#         loadCP(checkpointFile=checkpointFile)
#     except FileNotFoundError as err:
#         print(err)
#         return 0

def archive(dirToComp, archDir=archDir, background=True):
    if background:
        from multiprocessing import Process
        p = Process(target=compressDir, args=(dirToComp, archDir))
        p.start()
    else:
        compressDir(dirToComp, archDir)


def compressDir(dirToComp, archDir):
    print(f'{dirToComp} compression started')
    # destination file naming
    timestr = time.strftime("%y%m%d-%H%M")
    try:
        fname = f'{dirToComp[dirToComp.rindex("/"):]}_{timestr}'
    except ValueError:
        fname = f'{dirToComp}_{timestr}'
    # concatenate compression file path and name
    compFile = os.path.join(archDir, f'{fname}.tar.gz')
    with tarfile.open(compFile, 'w:gz') as tar:
        tar.add(dirToComp)
    print(f'{dirToComp} compression finished')
    removeDir(dirToComp)


def saveData(algorithm):
    gen = algorithm.n_gen
    # genDir = f'gen{gen}'
    # retrieve population from lastest generation
    genX = algorithm.pop.get('X')
    genF = algorithm.pop.get('F')
    # save checkpoint after each generation
    np.save(os.path.join(dataDir, 'checkpoint'), algorithm)
    # gen0 and every nCP generations save additional static checkpoint
    if gen % nCP == 1:
        np.save(f"{dataDir}/checkpoint-gen%i" % gen, algorithm)
    # save text file of variables and objectives as well
    # this provides more options for post-processesing data
    with open(f'{dataDir}/gen{gen}X.txt', "w+") as file:  # write file
        np.savetxt(file, genX)
    with open(f'{dataDir}/gen{gen}F.txt', "w+") as file:  # write file
        np.savetxt(file, genF)


def loadCP(checkpointFile=checkpointFile, hasTerminated=False):
    try:
        checkpoint, = np.load(checkpointFile, allow_pickle=True).flatten()
        # only necessary if for the checkpoint the termination criterion has been met
        checkpoint.has_terminated = hasTerminated
        alg = checkpoint
        # Update any changes made to the algorithms between runs
        from pymooCFD.setupOpt import pop_size, n_offsprings, xl, xu
        alg.pop_size = pop_size
        alg.n_offsprings = n_offsprings
        alg.problem.xl = xl
        alg.problem.xu = xu
        return alg
    except FileNotFoundError as err:
        print(err)
        raise Exception(f'{checkpointFile} load failed.')
        # return None

def printArray(array, labels, title):
    print(title, ' - ', end='')
    for i, label in enumerate(labels):
        print(f'{label}: {array[i]} / ', end='')
    print()



def runPop(X):
    client = Client(cluster())
    
    def fun(x_i, x):
        caseDir = os.path.join(preProcDir, f'lim_perm_sim-{x_i}') 
        f = runCase(caseDir, x)
        return f
    jobs = [client.submit(fun, x_i, x) for x_i, x in enumerate(X)]
    obj = np.row_stack([job.result() for job in jobs])
    
    client.close()
    return obj


def loadTxt(folder, fname):
    file = os.path.join(folder, fname)
    dat = np.loadtxt(file)
    return dat


def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


# def popGen(gen, checkpointFile=checkpointFile):
#     '''

#     Parameters
#     ----------
#     gen : int
#         generation you wish to get population from
#     checkpointFile : str, optional
#         checkpoint file path where Algorithm object was saved using numpy.save().
#         The default is checkpointFile (defined in beginning of setupOpt.py).

#     Returns
#     -------
#     pop :
#         Contains StaticProblem object with population of individuals from
#         generation <gen>.

#     Notes
#     -----
#         - development needed to handle constraints
#     '''
#     alg = loadCP(checkpointFile=checkpointFile)
#     X = alg.callback.data['var'][gen]
#     F = alg.callback.data['obj'][gen]

#     from pymoo.model.evaluator import Evaluator
#     from pymoo.model.population import Population
#     from pymoo.model.problem import StaticProblem
#     # now the population object with all its attributes is created (CV, feasible, ...)
#     pop = Population.new("X", X)
#     pop = Evaluator().eval(StaticProblem(problem, F=F), pop)  # , G=G), pop)
#     return pop, alg


# def loadTxt(fileX, fileF, fileG=None):
#     print(f'Loading population from files {fileX} and {fileF}...')
#     X = np.loadtxt(fileX)
#     F = np.loadtxt(fileF)
#     # F = np.loadtxt(f'{dataDir}/{fileF}')
#     if fileG is not None:
#         # G = np.loadtxt(f'{dataDir}/{fileG}')
#         G = np.loadtxt(fileG)
#     else:
#         G = None

#     from pymoo.model.evaluator import Evaluator
#     from pymoo.model.population import Population
#     from pymoo.model.problem import StaticProblem
#     # now the population object with all its attributes is created (CV, feasible, ...)
#     pop = Population.new("X", X)
#     pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)

#     from pymooCFD.setupOpt import pop_size
#     # from pymoo.algorithms.so_genetic_algorithm import GA
#     # # the algorithm is now called with the population - biased initialization
#     # algorithm = GA(pop_size=pop_size, sampling=pop)
#     from pymoo.algorithms.nsga2 import NSGA2
#     algorithm = NSGA2(pop_size=pop_size, sampling=pop)

#     return algorithm




# def restartGen(gen, checkpointFile=checkpointFile):
#     pop, alg = popGen(gen, checkpointFile=checkpointFile)
#     alg.sampling()

#     # from pymoo.algorithms.so_genetic_algorithm import GA
#     # the algorithm is now called with the population - biased initialization
#     # algorithm = GA(pop_size=100, sampling=pop)

#     from pymoo.optimize import minimize
#     from pymooCFD.setupOpt import problem
#     res = minimize(problem,
#                    alg,
#                    ('n_gen', 10),
#                    seed=1,
#                    verbose=True)
#     return res


# def loadTxt():
#     try:
#         print('Loading from text files')
#         X = np.loadtxt('var.txt')
#         F = np.loadtxt('obj.txt')
#     except OSError as err:
#         print(err)
#         print('Failed to load text files')
#         print('Data loading failed returning "None, None"...')
#         return None, None

# def archive(dirName, archName = 'archive.tar.gz'):
#     with tarfile.open(archName, 'a') as tar:
#         tar.add(dirName)

# compressDir('../../dump')


# print('creating archive')
# out = tarfile.open('example.tar.gz', mode='a')
# try:
#     print('adding README.txt')
#     out.add('../dump')
# finally:
#     print('closing tar archive')
#     out.close()
#
# print('Contents of archived file:')
# t = tarfile.open('example.tar.gz', 'r')
# for member in t.getmembers():
#     print(member.name)
