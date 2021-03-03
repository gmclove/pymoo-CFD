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
        gen = algorithm.n_gen
        print('GEN:%i' % gen)
        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen)

        out['F'] = sim.obj

        np.save("checkpoint-gen%i" % gen, algorithm)
        np.save("checkpoint", algorithm)

        print('GEN%i COMPLETE' % gen)

problem = MyProblem()

# TEST PROBLEM
# from pymoo.factory import get_problem
# problem = get_problem("bnh")

########################################################################################################################
plotDir = './test_plots'

checkpoint, = np.load("savedCPs/checkpoint.npy", allow_pickle=True).flatten()
algorithm = checkpoint

pf = algorithm.problem.pareto_front()
ps = algorithm.problem.pareto_set()

# from pymoo.optimize import minimize
# res = minimize(problem,
#                algorithm)

print('Number of individuals in final population: ' + str(len(algorithm.pop.get('X'))))
print('Number of generations: ', str(algorithm.n_gen), str(len(algorithm.callback.data['var'])), str(len(algorithm.callback.data['obj'])))

print('        Final Population')
print('     Parameters              Objectives')
float_formatter = '{:.4f}'.format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(np.c_[algorithm.pop.get('X'), algorithm.pop.get('F')])
A = [algorithm.pop.get('X'), algorithm.pop.get('F')]
for i in A:
    for j in i:
        for k in j:
            print('%.4f ' % k, end='')
        print()
# notes
'''
algorithm.pop.get('X') - returns array of individuals in final population
        equivalent to: algorithm.callback.data['var'][:][algorithm.n_gen-1]

algorithm.callback.data['var' or 'obj'][$generation$][$individual$]
        algorithm.callback.data['var'][algorithm.n_gen-1][0] - last generation, first individual 
        algorithm.callback.data['var'][$generation$] == algorithm.callback.data['var'][$generation$][:]
        algorithm.callback.data['var'][algorithm.n_gen-1][:] == algorithm.callback.data['var'][:][algorithm.n_gen-1]????
                - if [:] index is used other index is assumed to be generation no matter order of indices 
        algorithm.callback.data['var'][:]/[:][:] - returns array of array objects, use for loop to graph?????

algorithm.opt.get('X' or 'F')

'''

# print("Optimum:")
# print('Parameters: ' + str(algorithm.opt.get('X')))
# print('Objectives: ' + str(algorithm.opt.get('F')))

try:
    os.mkdir(plotDir)
except OSError:
    print(plotDir + ' directory already exists')

########################################################################################################################
from pymoo.visualization.scatter import Scatter

# All design points
plot = Scatter(title='Entire Design Space', legend=False)
for g in range(len(algorithm.callback.data['var'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.save(plotDir + '/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=False)
for g in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save(plotDir + '/entire_obj_space.png')

# Last 10 generations
# last 10 gens. design points
if algorithm.n_gen > 10:
    plot = Scatter(title='Last 10 Generations Design Space', legend=True)
    for g in range(algorithm.n_gen-10, algorithm.n_gen):  # algorithm.n_gen == len(algorithm.callback.data['var'])
        plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.save(plotDir + '/final_10_design_space.png')
    # last 10 gens. objective points
    plot = Scatter(title='Last 10 Generations Objective Space', legend=True)
    for g in range(len(algorithm.callback.data['obj'])-10, len(algorithm.callback.data['obj'])):  # range(algorithm.n_gen)
        plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.save(plotDir + '/final_10_obj_space.png')

########################################################################################################################
####### DECISION MAKING ##########
if pf is not None:
    from pymoo.factory import get_decision_making

    dm = get_decision_making("high-tradeoff")

    I = dm.do(pf)

    plot = Scatter(title='Pareto Front: High Tradeoff Points')
    plot.add(pf, alpha=0.2)
    plot.add(pf[I], color="red", s=100)
    plot.save(plotDir + '/pf-high-tradeoff')
########################################################################################################################
####### CONVERGENCE ##########
# n_evals = []    # corresponding number of function evaluations\
# F = []          # the objective space values in each generation
# cv = []         # constraint violation in each generation
# for data in algorithm.callback.data:
#     print(data)
#     # store the number of function evaluations
#     n_evals.append(data['n_evals'][:])
#
#     # store the least contraint violation in this generation
#     cv.append(data['opt'].get("CV").min())
#
#     # filter out only the feasible and append
#     feas = np.where(data['opt'].get("feasible"))[0]
#     _F = data['opt'].get("F")[feas]
#     F.append(_F)
# print(F)
# plot = Scatter()
# plot.add(F, n_evals)
# plot.save(plotDir + '/opt_convergence')
########################################################################################################################
####### PERFORANCE INDICATORS ##########





from pymoo.util.running_metric import RunningMetric

running = RunningMetric(
                        delta_gen=5,
                        # n_plots=4,
                        only_if_n_plots=True,
                        key_press=False,
                        do_show=True)

running.notify(algorithm)


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



