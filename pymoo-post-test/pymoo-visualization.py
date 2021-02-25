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
########################################################################################################################
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

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen)

        out['F'] = sim.obj

        np.save("checkpoint", algorithm)

        print('GEN%i COMPLETE' % gen)

        # objectives unconstrainted


problem = MyProblem()
########################################################################################################################
dir = './test_plots'

checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
algorithm = checkpoint

# from pymoo.optimize import minimize
# res = minimize(problem,
#                algorithm)

# print(res.X)
# print(algorithm.pop.get('X'))
print(algorithm.callback.data['var'])
print(len(algorithm.callback.data['var']))
# print(algorithm.callback.data['var'][len(algorithm.callback.data['var'])])
print(algorithm.callback.data['var'][:][0])
print()
print(algorithm.pop.get('X'))
print(algorithm.opt)

try:
    os.mkdir(dir)
except OSError:
    print(dir + ' directory already exists')

from pymoo.visualization.scatter import Scatter

# Design space
ps = algorithm.problem.pareto_set()  # use_cache=False, flatten=False)
plot = Scatter(title='Design Space')
# plot.add(res.X)
plot.add(algorithm.callback.data['var'])
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.save(dir + '/design_space.png')

# Objective Space
pf = algorithm.problem.pareto_front()
plot = Scatter(title="Objective Space")
# plot.add(res.F)
plot.add(algorithm.callback.data['obj'])
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save(dir + '/obj_space.png')

# All design points
plot = Scatter(title='Entire Design Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['var'][:][i], label='GEN %i' % i)
plot.save(dir + '/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=True)
for i in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][i], label='GEN %i' % i)
plot.save(dir + '/entire_obj_space.png')

########################################################################################################################
from pymoo.util.running_metric import RunningMetric

running = RunningMetric(delta_gen=10,
                        n_plots=4,
                        only_if_n_plots=True,
                        key_press=False,
                        do_show=True)

# for algorithm in res.history:
#     running.notify(algorithm)


import matplotlib.pyplot as plt

# val = res.algorithm.callback.data["best"]
# plt.plot(np.arange(len(val)), val)
# plt.show()


# plt.clf()
# plt.title('Entire Design Space')
# for i in range(len(algorithm.callback.data['var'])):
#     plt.scatter(algorithm.callback.data['var'][i][0], algorithm.callback.data['var'][i][1], label='GEN %i' % i)#, marker='o')
# plt.savefig('plots/matplot_design_space.png')
# plt.close()



