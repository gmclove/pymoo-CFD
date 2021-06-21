import os
import numpy as np
from pymooIN import *

# checkpointDir = 'checkpoint' + '.npy'
checkpointDir = 'checkpoint.npy'

plotDir = './plots'

########################################################################################################################
checkpoint, = np.load(checkpointDir, allow_pickle=True).flatten()
algorithm = checkpoint

pf = algorithm.problem.pareto_front()
ps = algorithm.problem.pareto_set()

print('Number of individuals in final population: ' + str(len(algorithm.pop.get('X'))))
print('Number of generations: ', str(algorithm.n_gen), str(len(algorithm.callback.data['var'])), str(len(algorithm.callback.data['obj'])))

# print('        Final Population')
# print('     Parameters              Objectives')
# float_formatter = '{:.4f}'.format
# np.set_printoptions(formatter={'float_kind':float_formatter})
# print(np.c_[algorithm.pop.get('X'), algorithm.pop.get('F')])
# A = [algorithm.pop.get('X'), algorithm.pop.get('F')]
# for i in A:
#     for j in i:
#         for k in j:
#             print('%.4f ' % k, end='')
#         print()

print('FINAL POPULATION')
print('Parameters')
print(algorithm.pop.get('X'))
print('Objectives')
print(algorithm.pop.get('F'))


print('EVERY GENERATION')
for gen in range(algorithm.n_gen):
    print(f'generation {gen}')
    var_g = algorithm.callback.data['var'][gen]
    obj_g = algorithm.callback.data['obj'][gen]
    for ind in range(len(var_g)):
        var_i = var_g[ind]
        obj_i = obj_g[ind]
        print(f'gen{gen} ind{ind}: ', end='')
        for n in range(len(var_i)):
            print(f'{var_labels[n]}: {var_i[n]}', end=' ')
        print(' // ', end='')
        for n in range(len(obj_i)):
            print(obj_labels[n] + ':' + '%.3f' % obj_i[n], end=' ')
        print()
        # print(f'var: {var_i}')
        # print(f'obj: {obj_i}')

#
# for ind in range(n_ind):
#
#     for obj in range(n_obj):
#         print(f'Objective {obj+1}:')
#         print('%.3f' % algorithm.callback.


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
##### SCATTER PLOTS #######
###########################
from pymoo.visualization.scatter import Scatter

# All design points
if algorithm.n_gen < 10:
    leg = True
else:
    leg = False
plot = Scatter(title='Entire Design Space', legend=leg, labels=var_labels)
for g in range(len(algorithm.callback.data['var'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.save(plotDir + '/entire_design_space.png')

# All objective points
plot = Scatter(title='Entire Objective Space', legend=leg, labels=obj_labels)
for g in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.save(plotDir + '/entire_obj_space.png')

# Last 10 generations
# last 10 gens. design points
if algorithm.n_gen > 10:
    plot = Scatter(title='Last 10 Generations Design Space', legend=True, labels=var_labels)
    for g in range(algorithm.n_gen-10, algorithm.n_gen):  # algorithm.n_gen == len(algorithm.callback.data['var'])
        plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.save(plotDir + '/final_10_design_space.png')
    # last 10 gens. objective points
    plot = Scatter(title='Last 10 Generations Objective Space', legend=True, labels=obj_labels)
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

# print('Reynolds Number:')
# U = 1  # [m/s]
# v = 0.01  # [m^2/s] kinematic viscosity of air
# # print(v)
# for gen in range(len(algorithm.callback.data['var'])):
#     print('gen%i:' % gen)
#     for ind in range(len(algorithm.callback.data['var'])):
#         x = algorithm.callback.data['var'][gen][ind]
#         cyl_D = x[0]
#         Re = U*cyl_D / v
#         print('     ind%i: Re=%f, cyl_D=%f' % (ind, Re, cyl_D))
