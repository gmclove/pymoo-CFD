import os
import numpy as np
########################################################################################################################
######    DISPLAY    ######
from pymoo.util.display import Display


class MyDisplay(Display):
    # bestObj = []
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
        self.output.append("metric_b", np.mean(algorithm.pop.get("F")))
        # self.output.append('Best Drag [N]', algorithm.pop.get("F")[:, 0].min())
        # self.output.append('Best Drag [N]', np.mean(algorithm.pop.get("F")[:, 1].min()))
        # if


display = MyDisplay()
########################################################################################################################
######    CALLBACK    ######
from pymoo.model.callback import Callback


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        # self.gen = 0
        self.data["best_obj1"] = []
        self.data['best_obj2'] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        self.data["best_obj1"].append(algorithm.pop.get("F")[:, 0].min())
        self.data['best_obj2'].append(algorithm.pop.get('F')[:, 1].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        # self.gen += 1


callback = MyCallback()

########################################################################################################################
######    PROBLEM    ######
from pymoo.model.problem import Problem
from pymoo.util.misc import stack

# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4
from RunYALES2 import RunYALES2


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
        gen = algorithm.n_gen
        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # Maximum processors to be used
        # procLim = 1
        # Number of processors for each individual (EQUAL or SMALLER than procLim)
        # nProc = 1

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen)

        out['F'] = sim.obj

        np.save("checkpoint", algorithm)

        # print('GEN%i COMPLETE' % gen)

        # objectives unconstrainted
        # g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        # g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8
        # out["G"] = np.column_stack([g1, g2])
    #
    # # --------------------------------------------------
    # # Pareto-front - not necessary but used for plotting
    # # --------------------------------------------------
    # def _calc_pareto_front(self, flatten=True, **kwargs):
    #     f1_a = np.linspace(0.1 ** 2, 0.4 ** 2, 100)
    #     f2_a = (np.sqrt(f1_a) - 1) ** 2
    #
    #     f1_b = np.linspace(0.6 ** 2, 0.9 ** 2, 100)
    #     f2_b = (np.sqrt(f1_b) - 1) ** 2
    #
    #     a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
    #     return stack(a, b, flatten=flatten)
    #
    # # --------------------------------------------------
    # # Pareto-set - not necessary but used for plotting
    # # --------------------------------------------------
    # def _calc_pareto_set(self, flatten=True, **kwargs):
    #     x1_a = np.linspace(0.1, 0.4, 50)
    #     x1_b = np.linspace(0.6, 0.9, 50)
    #     x2 = np.zeros(50)
    #
    #     a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
    #     return stack(a, b, flatten=flatten)


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
else:
    algorithm = NSGA2(
        pop_size=10,
        n_offsprings=2,
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
               ("n_gen", 20),
               callback=callback,
               seed=1,
               copy_algorithm=False,
               # pf=problem.pareto_front(use_cache=False),
               save_history=True,
               display=display,
               verbose=True
               )

np.save("checkpoint", algorithm)
print("EXEC TIME: " + str(res.exec_time))

# print('Variables:')
# print(callback.data['var'])
# print('Objectives:')
# print(callback.data['obj'])
# print('Best Objective 1:')
# for i in range(len(callback.data['best_obj1'])):
#     print('%.6f' % callback.data['best_obj1'][i])
# print('Best Objective 2:')
# for i in range(len(callback.data['best_obj2'])):
#     print('%.6f' % callback.data['best_obj2'][i])
#
# # print("Time Elapsed:")
# # print(res.time)
# print("Objectives:")
# print(res.pop.get('F'))
# # print(res.F)
# print("Variables:")
# print(res.pop.get('X'))
# # print(res.X)
########################################################################################################################
###### VISUALIZATION ######
checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
algorithm = checkpoint

try:
    os.mkdir('./plots')
except OSError:
    print('./plots directory already exists')

from pymoo.visualization.scatter import Scatter
# Design space
ps = algorithm.problem.pareto_set()  # use_cache=False, flatten=False)
plot = Scatter(title='Design Space')
plot.add(res.X)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.save('plots/design_space.png')

# Objective Space
pf = algorithm.problem.pareto_front()
plot = Scatter(title="Objective Space")
plot.add(res.F)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save('plots/obj_space.png')

# All design points
plot = Scatter(title='Entire Design Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['var'][i], label='GEN %i' % i)
plot.save('plots/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][i], label='GEN %i' % i)
plot.save('plots/entire_obj_space.png')
########################################################################################################################
