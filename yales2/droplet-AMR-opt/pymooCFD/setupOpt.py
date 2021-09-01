import os
import numpy as np
from pymooCFD.util.sysTools import makeDir

# inputFile = '2D_cylinder.in'
#### SLURM JOB #####
# jobFile = 'jobslurm.sh'
# jobName = 'moo-new'  # ___ characters max ???????????????

#### Single Node Job #####
procLim = 80  # Maximum processors to be used, defined in jobslurm.sh as well
nProc = 20  # Number of processors for each individual (EQUAL or SMALLER than procLim)


#### Parallelize Criteria #####
# n_parallel_proc = None
# parallel = 'multiProc'
# multiProcSelect = ['multiProc', 'singleMachine', 'singleNode']

####################################
### Setup Dask Computing Cluster ###
####################################
# HPC setup: https://docs.dask.org/en/latest/setup/hpc.html
from dask.distributed import Client, LocalCluster
n_workers = 8 # default: None
processes = True
threads_per_worker = None
cluster = LocalCluster()
client = Client(cluster)
print('### Dask Distributed Computing Client Started Sucessfully')
print(client)
print('Number of Cores: ')
for key, value in client.ncores().items():
    print(f'    {key} - {value} CPUs')
    

solverExec = 'droplet_convection'

#####################################
#### Genetic Algorithm Criteria #####
#####################################
n_gen = 2
pop_size = 2
n_offsprings = int(pop_size * (1 / 2)) # = number of evaluations each generation
#####################################
####### Define Design Space #########
#####################################
n_var = 3
var_labels = ['Threshold', 'Propagation Steps', 'Max. Number of Steps']
# use boolean to indicate if re-meshing is necessary because parameter is geomtric
geoVarsI =   [False, False, False]  # , False]
varType =    ['real', 'int', 'int']  # options: 'int' or 'real'
xl =         [0.05, 1, 1]  # , 0.10]  # lower limits of parameters/variables
xu =         [0.3, 20, 4]  # , 0.40]  # upper limits of variables
# n_var = len(var_labels)
if not len(xl) == len(xu) and len(xu) == len(var_labels) and len(var_labels) == n_var:
    raise Exception("Design Space Definition Incorrect")
#######################################
####### Define Objective Space ########
#######################################
obj_labels = ['Simulation Time', 'Vortex Locations Error']
n_obj = 2
n_constr = 0
# values used to normalize objectives
# often these values are explored in optimization pre-processing
# def normalize(obj):
#     obj_max = [2220, 3.602576326924016937e-12]  # maximum possible value
#     # utopia point (ideal values), aspiration point, target value, or goal
#     obj_o = [3.230000000000000000e+02, ]
#     # for loop through each individual
#     for obj_ind in obj:
#         # objective 1 normalization
#         obj_norm = np.subtract(obj_ind, obj_o) / np.subtract(obj_max, obj_o)
#     return obj_norm
#####################################
##### Define Mesh Parameters ########
#####################################
# NOTE: only used in optimization studies with geometric parameters
# Generation 0 (random sampling of design space) mesh convergence study
# meshSizeMin = 0.05
# meshSizeMax = 0.5
# nMeshes = 5
# meshSizes = np.linspace(meshSizeMin, meshSizeMax, nMeshes)

####################################
####### Define Data Handling #######
####################################
archDir = 'archive'
makeDir(archDir)
nCP = 10  # number of generations between extra checkpoints
dataDir = 'dump-test'
makeDir(dataDir)
checkpointFile = os.path.join(dataDir, 'checkpoint.npy')

############################################
###### Define CFD Pre/Post Processing ######
############################################
inputFile = 'DynamicAdaptation.in'
baseCaseDir = 'base_case'
hqSimDatPath = 'hq_sim_coor.txt'

#####################################################
###### Define Optimization Pre/Post Processing ######
#####################################################
preProcDir = 'preProcOpt'
makeDir(preProcDir)
plotDir = os.path.join(preProcDir, 'plots')
makeDir(plotDir)
mapDir = os.path.join(plotDir, 'mapGen')
makeDir(mapDir)
#### Mesh Sensitivity Study ####
studyDir = os.path.join(preProcDir, 'meshStudy')
makeDir(studyDir)
baseCaseMS = os.path.join(studyDir, 'cartMeshCase')
# copy_tree(baseCaseDir, baseCaseMS)
# mesh size factors 
meshSF = [0.25, 0.5, 1, 1.5, 2.0, 2.5, 3.0]   # np.linspace(0.5,4,8) #[0.5, 1, 1.5, 1.75, 2, 2.25, 2.5]


'''
pyMOO SETUP
-----------
pymoo.org
'''
########################################################################################################################
######    OPERATORS   ######
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

sampling = MixedVariableSampling(varType, {
    "real": get_sampling("real_lhs"),  # "real_random"),
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
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        for obj in range(n_obj):
            self.output.append("mean obj." + str(obj), np.mean(algorithm.pop.get('F')[:, obj]))
            self.output.append("best obj." + str(obj), algorithm.pop.get('F')[:, obj].min())

display = MyDisplay()
########################################################################################################################
######    CALLBACK    ######
from pymoo.model.callback import Callback

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        for obj in range(n_obj):
            self.data["best_obj" + str(obj)] = []
        self.data['var'] = []
        self.data['obj'] = []
        # convergence stats
        self.n_evals = []
        self.opt = []

    def notify(self, alg):
        from pymooCFD.util.handleData import saveData
        print(alg)
        global algorithm
        algorithm = alg
        print(algorithm)
        saveData(algorithm)
        # self.output.header()
        self.display_header = True
        # optimization convergence stats
        self.n_evals.append(alg.evaluator.n_eval)
        self.opt.append(alg.opt[0].F)
        for obj in range(n_obj):
            self.data["best_obj" + str(obj)].append(alg.pop.get('F')[:, obj].min())
        self.data['var'].append(alg.pop.get('X'))
        self.data['obj'].append(alg.pop.get('F'))
        # saveData(alg)
        # global algorithm
        # algorithm = alg

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
# import time


class GA_CFD(Problem):
    def __init__(self, client, *args, **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu,
                         # elementwise evaluation: 
                         # one call of _evaluate per individual
                         elementwise_evaluation=False,
                         *args,
                         **kwargs
                         )
        self.client = client

    def _evaluate(self, X, out, *args, **kwargs):
        # print(x)
        # print('algorithm.n_gen:' + str(algorithm.n_gen) + ' len(alg...data[''var'']):' + str(len(algorithm.callback.data['var'])))
        
        ###### Initialize Generation ######
        global algorithm    
        gen = algorithm.n_gen
        if gen is None:
            gen = 1
        # create generation directory for storing data/executing simulations
        genDir = os.path.join(dataDir, f'gen{gen}')
        print('Starting ', genDir)
        
        ###### ARCHIVE/REMOVE PREVIOUS GENERATION DATA ######
        if gen != 1: 
            prev_genDir = os.path.join(dataDir, f'gen{gen - 1}')
            # archive/remove generation folder to prevent build up of data
            from pymooCFD.util.handleData import removeDir  # archive
            removeDir(prev_genDir)
            # archive(genDir, archDir, background=True)
        
        ##### PRE-PROCCESS GENERATION #####
        ### OPTIONAL: usually best practice to parallelize pre/post processing
        
        ###### RUN GENERATION ######
        # from pymooCFD.setupCFD import runGen
        # obj = runGen(genDir, X)
        ###### RUN GENERATION ######
        from pymooCFD.setupCFD import runCase
        def fun(x_i, x):
            caseDir = os.path.join(genDir, f'ind{x_i + 1}') 
            f = runCase(caseDir, x)
            return f
        jobs = [self.client.submit(fun, x_i, x) for x_i, x in enumerate(X)]
        obj = np.row_stack([job.result() for job in jobs])
        
        
        # make sure the setupCFD.runCFD does not return all zeros
        if not np.all(obj):
            print("ALL OBJECTIVES = 0")
            exit()

        out['F'] = obj

        print(f'GENERATION {gen} COMPLETE')
    
    # handle pickle sealization when saving algorithm object as checkpoint
    # self.client has an active network connect so it can not be serialized
    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        del state['client']
        return state
    
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        # print(state)
        self.__dict__.update(state)
        self.client = client

problem = GA_CFD(client)
# TEST PROBLEM
# from pymoo.factory import get_problem
# problem = get_problem("bnh")


########################################################################################################################
######    ALGORITHM    ######
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

# initialize algorithm here
# will be overwritten in runOpt() if checkpoint already exists
algorithm = NSGA2(pop_size=pop_size,
                  n_offsprings=n_offsprings,
                  sampling=sampling,
                  crossover=crossover,
                  mutation=mutation,
                  eliminate_duplicates=True
                  )

def setAlgorithm(alg):
    global algorithm
    algorithm = alg
