import os
import numpy as np

########################################################################################################################
######    DISPLAY    ######
from displays import MyDisplay
########################################################################################################################
########################################################################################################################
######    CALLBACK    ######
from callbacks import MyCallback

########################################################################################################################
######    PROBLEM    ######
# from custom_problems import MyProblem

# TEST PROBLEM
from pymoo.factory import get_problem
problem = get_problem("bnh")

########################################################################################################################
dir = './test_plots'

checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
algorithm = checkpoint

pf = algorithm.problem.pareto_front()
ps = algorithm.problem.pareto_set()

# from pymoo.optimize import minimize
# res = minimize(problem,
#                algorithm)

print('Number of individuals in final population: ' + str(len(algorithm.pop.get('X'))))
print('Number of generations: ' + str(algorithm.n_gen) + ' ' + str(len(algorithm.callback.data['var'])))
print(algorithm.pop.get('X'))
# print(algorithm.callback.data['var'][:][algorithm.n_gen-1])
print(algorithm.callback.data['var'][9] == algorithm.callback.data['var'][9][:])


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
    os.mkdir(dir)
except OSError:
    print(dir + ' directory already exists')

########################################################################################################################
from pymoo.visualization.scatter import Scatter

# All design points
plot = Scatter(title='Entire Design Space', legend=False)
for g in range(len(algorithm.callback.data['var'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.save(dir + '/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=False)
for g in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save(dir + '/entire_obj_space.png')

# Last 10 generations
# last 10 gens. design points
plot = Scatter(title='Last 10 Generations Design Space', legend=True)
for g in range(len(algorithm.callback.data['var'])-10, len(algorithm.callback.data['var'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.save(dir + '/final_10_design_space.png')
# last 10 gens. objective points
plot = Scatter(title='Last 10 Generations Objective Space', legend=True)
for g in range(len(algorithm.callback.data['obj'])-10, len(algorithm.callback.data['obj'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save(dir + '/final_10_obj_space.png')

########################################################################################################################
####### DECISION MAKING ##########
from pymoo.factory import get_decision_making

dm = get_decision_making("high-tradeoff")

I = dm.do(pf)

plot = Scatter(title='Pareto Front: High Tradeoff Points')
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.save(dir + '/pf-high-tradeoff')
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
# plot.save(dir + '/opt_convergence')
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



