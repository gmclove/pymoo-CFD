import tarfile
from pymooCFD.setupOpt import checkpointFile, dataDir, problem, nCP
import shutil
import numpy as np


def archive(dirToComp, archDir, background=True):
    if background == True:
        from multiprocessing import Process
        p = Process(target=compressDir, args=(dirToComp, ))
        p.start()
    else:
        compressDir(dirToComp, archDir)
    

def compressDir(dirToComp, archDir):
    print(f'{dirToComp} compression started')
    try:
        fname = dirToComp[dirToComp.rindex("/"):]
    except ValueError:
        fname = dirToComp
    compFile = f'{archDir}/{fname}.tar.gz'
    with tarfile.open(compFile, 'w:gz') as tar:
        tar.add(dirToComp)
    print(f'{dirToComp} compression finished')
    removeDir(dirToComp)
    
def removeDir(path):
    print(f'removing {path}..')
    try:
        shutil.rmtree(path)        
        print(f"{path} removed successfully")
    except OSError as err:
        print(err)

def saveData(algorithm):
    gen = algorithm.n_gen
    genDir = f'gen{gen}'
    # retrieve population from lastest generation
    genX = algorithm.pop.get('X')
    genF = algorithm.pop.get('F')
    # save checkpoint after each generation
    np.save(f"{dataDir}/checkpoint", algorithm)
    # gen0 and every nCP generations save additional static checkpoint
    if gen % nCP == 1:
        np.save(f"{dataDir}/checkpoint-gen%i" % gen, algorithm)
    # save text file of variables and objectives as well
    # this provides more options for post-processesing data
    with open(f'{dataDir}/gen{gen}X.txt', "w+") as file: # write file
        np.savetxt(file, genF)
    with open(f'{dataDir}/gen{gen}F.txt', "w+") as file: # write file
        np.savetxt(file, genX)


def loadCP(checkpointFile=checkpointFile, hasTerminated=False):
    checkpoint, = np.load(checkpointFile, allow_pickle=True).flatten()
    print("Loaded Checkpoint:", checkpoint)
    # only necessary if for the checkpoint the termination criterion has been met
    checkpoint.has_terminated = hasTerminated
    alg = checkpoint
    print('Last checkpoint at generation %i' % len(alg.callback.data['var']))
    
    # Update any changes made to the algorithms between runs 
    from pymooCFD.setupCFD import n_ind
    alg.pop_size = n_ind
    return alg


def loadTxt(fileX, fileF, fileG=None):
    print(f'Loading population from files {fileX} and {fileF}...')
    X = np.loadtxt(fileX)
    F = np.loadtxt(fileF)
    # F = np.loadtxt(f'{dataDir}/{fileF}')
    if fileG is not None:
        # G = np.loadtxt(f'{dataDir}/{fileG}')
        G = np.loadtxt(fileG)
    else:
        G = None 

    from pymoo.model.evaluator import Evaluator
    from pymoo.model.population import Population
    from pymoo.model.problem import StaticProblem
    # now the population object with all its attributes is created (CV, feasible, ...)
    pop = Population.new("X", X)
    pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)
    
    from pymooCFD.setupOpt import n_ind
    # from pymoo.algorithms.so_genetic_algorithm import GA
    # # the algorithm is now called with the population - biased initialization
    # algorithm = GA(pop_size=n_ind, sampling=pop)
    from pymoo.algorithms.nsga2 import NSGA2
    algorithm = NSGA2(pop_size=n_ind, sampling=pop)
    
    return algorithm
    
    


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
