import os
import numpy as np

baseCaseDir = 'base_case'
inputFile = '2D_cylinder.in'
#### SLURM JOB #####
# jobFile = 'jobslurm.sh'
# jobName = 'moo-new'  # ___ characters max ???????????????

#### Single Node Job #####
procLim = 60  # Maximum processors to be used, defined in jobslurm.sh as well
nProc = 12  # Number of processors for each individual (EQUAL or SMALLER than procLim)
solverExec = '2D_cylinder'

n_gen = 1
pop = 5
#####################################
####### Define Design Space #########
#####################################
n_var = 2
var_labels = ['Amplitude', 'Frequency']
geoVarsI =   [False, False]
varType =    ['real', 'real']  # options: 'int' or 'real'
xl =         [0.1, 0.1]  # lower limits of parameters/variables
xu =         [3.0, 1]  # upper limits of variables
#######################################
####### Define Objective Space ########
#######################################
obj_labels = ['Drag on Cylinder', 'Power Input']
n_obj = 2
n_constr = 0
# values used to normalize objectives
def normalize(obj):
    obj_max = [1.0, 2816.971884694359] # maximum possible value
    # utopia point (ideal value), aspiration point, target value, or goal
    obj_o = [0, 0]
    # for loop through each individual
    for obj_ind in obj:
        # objective 1 normalization
        obj_norm = np.subtract(obj_ind, obj_o)/np.subtract(obj_max, obj_o)
#####################################
##### Define Mesh Parameters ########
#####################################
# Generation 0 (random sampling of design space) mesh convergence study
# meshSizeMin = 0.05
# meshSizeMax = 0.5
# nMeshes = 5
# meshSizes = np.linspace(meshSizeMin, meshSizeMax, nMeshes)

####################################
####### Define Data Handling #######
####################################
# dirCP =
nCP = 10  # number of generations between extra checkpoints
dataDir = 'dump'
checkpointFile = f'{dataDir}/checkpoint.npy'
try:
    os.mkdir(dataDir)
except OSError as err:
    print(err)
    print('data directory already exists')

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
    algorithm = checkpoint
    print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
    return algorithm

#####################################################
###### Define Optimization Pre/Post Processing ######
#####################################################
plotDir = './plots'
try:
    os.mkdir(plotDir)
except OSError as err:
    print(err)
    print(f'{plotDir} already exists.')
'''
pyMOO SETUP
pymoo.org
'''
########################################################################################################################
######    MIXED VARIABLE    ######
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

sampling = MixedVariableSampling(varType, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(varType, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(varType, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})

########################################################################################################################
######    DISPLAY    ######
from pymoo.util.display import Display


class MyDisplay(Display):
    # bestObj = []
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(n_obj):
            self.output.append("mean obj."+str(obj), np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append("best obj."+str(obj), algorithm.pop.get('F')[:, obj].min())

display = MyDisplay()
########################################################################################################################
######    CALLBACK    ######
from pymoo.model.callback import Callback


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        # self.gen = algorithm.n_gen
        for obj in range(n_obj):
            self.data["best_obj"+str(obj)] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        for obj in range(n_obj):
            self.data["best_obj"+str(obj)].append(algorithm.pop.get('F')[:, obj].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))

        saveData(algorithm)

callback = MyCallback()
########################################################################################################################
######    TERMINATION CRITERION  ######
# https://pymoo.org/interface/termination.html
from pymoo.factory import get_termination
termination = get_termination("n_gen", n_gen)

# from pymoo.util.termination.default import MultiObjectiveDefaultTermination
# termination = MultiObjectiveDefaultTermination(
#     x_tol=1e-8,
#     cv_tol=1e-6,
#     f_tol=0.0025,
#     nth_gen=5,
#     n_last=30,
#     n_max_gen=1000,
#     n_max_evals=100000
# )

########################################################################################################################
######    PROBLEM    ######
from pymoo.model.problem import Problem

class GA_CFD(Problem):
    def __init__(self):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x)
        ###### Initialize Generation ######
        # gen = algorithm.n_gen. A thick orange line illustrates the pareto-optimal set. Through the combination of both constraints, the pareto-set is split into two parts. Analytically, the pareto-optimal set is given by PS={(x1,x2)|(0.1≤x1≤0.4)∨(0.6≤x1≤0.9)∧x2=0}
        print(algorithm)
        print('algorithm.n_gen:' + str(algorithm.n_gen) + ' len(alg...data[''var'']):' + str(len(algorithm.callback.data['var'])))
        # gen = len(algorithm.callback.data['var'])
        # algorithm = loadCP()
        gen = algorithm.n_gen
        if gen is None:
            print('gen is None. exiting...')
            exit()
        genDir = f'gen{gen}'
        subdir = 'ind'
        # print('GEN:%i' % gen)

        ###### RUN GENERATION ######
        from distutils.dir_util import copy_tree
        from pymooCFD.setupCFD import preProc
        for ind in range(len(x)):
            indDir = f'{genDir}/{subdir}{ind}'
            copy_tree(baseCaseDir, indDir)
            preProc(indDir, x[ind, :])

        from pymooCFD.execSimsBatch.singleNode import execSims
        execSims(genDir, subdir, len(x))

        from pymooCFD.setupCFD import postProc
        obj = np.ones((pop, n_obj))
        for ind in range(len(x)):
            indDir = f'{genDir}/{subdir}{ind}'
            obj[ind] = postProc(indDir, x[ind, :])
        # create sim object for this generation and it's population
        out['F'] = obj

        print('GEN%i COMPLETE' % gen)

problem = GA_CFD()

# TEST PROBLEM
# from pymoo.factory import get_problem
# problem = get_problem("bnh")
########################################################################################################################
######    ALGORITHM    ######
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

# initialize algorithm here
# will be overwritten in runOpt() if checkpoint already exists
algorithm = NSGA2(
    pop_size=pop,
    # n_offsprings=2,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True
    )
