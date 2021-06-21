import os
import numpy as np

# SLURM JOB
job_name = 'dip-moo'  # ___ characters max ???????????????
procLim = 16  # Maximum processors to be used, defined in jobslurm.sh as well
nProc = 8  # Number of processors for each individual (EQUAL or SMALLER than procLim)
solverExec = f'mpirun -n {nProc} 2D_vortex'

n_gen = 1
pop = 20

# Define Design Space
n_var = 2
var_labels = ['Threshold', 'Propagation Steps']
geoVarsI =   [False, False]
varType =    ['real', 'int']  # options: 'int' or 'real'
xl =         [0.05, 1]  # lower limits of parameters/variables
xu =         [0.15, 10]  # upper limits of variables

# Define Objective Space
obj_labels = ['Simulation Time', 'Vortex Locations Error']
n_obj = 2
n_constr = 0

# Generation 0 (random sampling of design space) mesh convergence study
# meshSizeMin = 0.05
# meshSizeMax = 0.5
# nMeshes = 5
# meshSizes = np.linspace(meshSizeMin, meshSizeMax, nMeshes)

# dirCP =
nCP = 10  # number of generations between extra checkpoints

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
            self.data["best_obj"+str(obj)].append(algorithm.pop.get("F")[:, obj].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        # self.gen += 1 #= algorithm.n_gen

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
# from runYALES2 import RunYALES2
from runGen_sbatch import runGen


class MyProblem(Problem):
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
        # print('algorithm.n_gen:' + str(algorithm.n_gen) + ' len(alg...data[''var'']):' + str(len(algorithm.callback.data['var'])))
        gen = len(algorithm.callback.data['var'])
        # print('GEN:%i' % gen)

        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # create sim object for this generation and it's population
        # sim = RunYALES2(x, gen, procLim, nProc)
        # out['F'] = sim.obj
        out['F'] = runGen(x, gen)

        # out['F'] = np.zeros((pop, n_obj))
        # out['F'] = [[0, 1], [0, 1]]

        # save checkpoint after each generation
        np.save("checkpoint", algorithm)
        # every nCP generations save additional static checkpoint
        if gen % nCP == 0 and gen is not 0:
            np.save("checkpoint-gen%i" % gen, algorithm)
        # print('GEN%i COMPLETE' % gen)


problem = MyProblem()

# TEST PROBLEM
# from pymoo.factory import get_problem
# problem = get_problem("bnh")
########################################################################################################################
######    ALGORITHM    ######
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

if os.path.exists('checkpoint.npy'):
    checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
    print("Loaded Checkpoint:", checkpoint)
    # only necessary if for the checkpoint the termination criterion has been met
    checkpoint.has_terminated = False
    algorithm = checkpoint
    print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
else:
    algorithm = NSGA2(
        pop_size=pop,
        # n_offsprings=2,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )
