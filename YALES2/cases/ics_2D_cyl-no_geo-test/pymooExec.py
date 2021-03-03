import os
import numpy as np


# cwd = '~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cyl-no_geo-test'
n_gen = 6
pop = 2

# dirCP =
nCP = 10
########################################################################################################################
######    DISPLAY    ######
from pymoo.util.display import Display


class MyDisplay(Display):
    # bestObj = []
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("metric_a", np.mean(algorithm.pop.get('F')))
        self.output.append("metric_b", np.mean(algorithm.pop.get('F')))
        # self.output.append('Best Drag [N]', algorithm.pop.get("F")[:, 0].min())
        # self.output.append('Best Drag [N]', np.mean(algorithm.pop.get("F")[:, 1].min()))
        # if


########################################################################################################################
######    CALLBACK    ######
from pymoo.model.callback import Callback


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        # self.gen = algorithm.n_gen
        self.data["best_obj1"] = []
        self.data['best_obj2'] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        self.data["best_obj1"].append(algorithm.pop.get("F")[:, 0].min())
        self.data['best_obj2'].append(algorithm.pop.get('F')[:, 1].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        # self.gen += 1 #= algorithm.n_gen

########################################################################################################################
######    PROBLEM    ######
from pymoo.model.problem import Problem

# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4
from runYALES2 import RunYALES2


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         # omega freq
                         xl=np.array([0.1, 0.1]),
                         xu=np.array([3, 1])
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        ###### Initialize Generation ######
        # gen = algorithm.n_gen
        print('algorithm.n_gen:' + str(algorithm.n_gen) + ' len(alg...data[''var'']):' + str(len(algorithm.callback.data['var'])))
        gen = len(algorithm.callback.data['var'])
        # print('GEN:%i' % gen)

        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # create sim object for this generation and it's population
        # sim = RunYALES2(x, gen)
        # out['F'] = sim.obj

        out['F'] = np.zeros((pop, 2))
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
########################################################################################################################
######    OPTIMIZATION    ######
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               ("n_gen", n_gen),
               callback=MyCallback(),
               seed=1,
               copy_algorithm=False,
               # pf=problem.pareto_front(use_cache=False),
               save_history=True,
               display=MyDisplay(),
               verbose=True
               )

# np.save("checkpoint", algorithm)
print("EXEC TIME: %.3f seconds" % res.exec_time)