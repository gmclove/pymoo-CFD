import os
import numpy as np

# SLURM JOB
job_name = 'cyl-moo'  # _ characters max ???????????????
procLim = 24  # Maximum processors to be used, defined in jobslurm.sh as well
nProc = 8  # Number of processors for each individual (EQUAL or SMALLER than procLim)

# cwd = '~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cyl-no_geo-test'
n_gen = 2
pop = 3

# Define Design Space
n_var = 3
            # C_D[m], omega[rad./s], freq[1/s]
# var_labels = ['cylD', 'omega', 'freq']
# geoVarsI =   [True, False, False]
# xl =         [0.04, 0.1, 0.1]  # lower limits of parameters/variables
# xu =         [0.06, 3, 1]  # upper limits of variables
var_labels = ['cylD', 'omega', 'freq']
geoVarsI =   [True, False, False]
xl =         [0.04, 0.1, 0.1]  # lower limits of parameters/variables
xu =         [0.06, 3, 1]  # upper limits of variables

# Define Objective Space
n_obj = 2
n_constr = 0

# Generation 0 (random sampling of design space) mesh convergence study
meshSizeMin = 0.05
meshSizeMax = 0.5
nMeshes = 5
meshSizes = np.linspace(meshSizeMin, meshSizeMax, nMeshes)

# dirCP =
nCP = 10  # number of generations between extra checkpoints

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

# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4
from runYALES2 import RunYALES2


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
        # gen = algorithm.n_gen
        # print('algorithm.n_gen:' + str(algorithm.n_gen) + ' len(alg...data[''var'']):' + str(len(algorithm.callback.data['var'])))
        gen = len(algorithm.callback.data['var'])
        # print('GEN:%i' % gen)

        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen, procLim, nProc)
        # out['F'] = sim.obj

        # out['F'] = np.zeros((pop, n_obj))
        # out['F'] = [[0, 1], [0, 1]]

        if gen % nCP == 0 and gen is not 0:
            np.save("checkpoint-gen%i" % gen, algorithm)
        np.save("checkpoint", algorithm)

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
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )


# def getProcLim():
#     return procLim