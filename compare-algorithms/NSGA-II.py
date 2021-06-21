import numpy as np
import os

# Example Problem: Kursawe - number of objectives=2, number of parameters=3

#################################
######### PROBLEM ###############
#################################
from pymoo.factory import get_problem
problem = get_problem("kursawe")

#################################
######### ALGORITHM #############
#################################
from pymoo.algorithms.nsga2 import NSGA2
algorithm = NSGA2(pop_size=100)

#################################
######### CALLBACK ##############
#################################
from pymoo.model.callback import Callback
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        # save checkpoint after each generation
        np.save("checkpoint", algorithm)

callback = MyCallback()

#################################
######### OPTIMIZATION ##########
#################################
from pymoo.optimize import minimize
n_gen = 50
res = minimize(problem,
               algorithm,
               ('n_gen', n_gen),
               callback=callback,
               seed=1,
               verbose=False)

#################################
######### PLOTTING ##############
#################################
checkpointDir = './checkpoint.npy'
plotDir = './plots'

# load checkpoint and make plot directory
checkpoint, = np.load(checkpointDir, allow_pickle=True).flatten()
algorithm = checkpoint
try:
    os.mkdir(plotDir)
except OSError:
    print(plotDir + ' directory already exists')
algorithm = res.algorithm
# load pareto front and set
ps = algorithm.problem.pareto_set()
pf = algorithm.problem.pareto_front()

# All design points
from pymoo.visualization.scatter import Scatter
if algorithm.n_gen < 10:
    leg = True
else:
    leg = False
plot = Scatter(title='Entire Design Space')
for g in range(len(algorithm.callback.data['var'])):  # range(algorithm.n_gen)
    plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
if ps is not None:
    plot.add(ps, plot_type="line", color="black", alpha=0.7)
# plot.show()
plot.save(plotDir + '/entire_design_space.png')

plot = Scatter(title='Entire Objective Space', legend=leg)#, labels=obj_labels)
for g in range(len(algorithm.callback.data['var'])):
    plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
if pf is not None:
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
# plot.show()
plot.save(plotDir + '/entire_obj_space.png')


# Last 10 generations
# last 10 gens. design points
if algorithm.n_gen > 10:
    plot = Scatter(title='Last 10 Generations Design Space', legend=True) #, labels=var_labels)
    for g in range(algorithm.n_gen-10, algorithm.n_gen):  # algorithm.n_gen == len(algorithm.callback.data['var'])
        plot.add(algorithm.callback.data['var'][g][:], label='GEN %i' % g)
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.save(plotDir + '/final_10_design_space.png')
    # last 10 gens. objective points
    plot = Scatter(title='Last 10 Generations Objective Space', legend=True) #, labels=obj_labels)
    for g in range(len(algorithm.callback.data['obj'])-10, len(algorithm.callback.data['obj'])):  # range(algorithm.n_gen)
        plot.add(algorithm.callback.data['obj'][g][:], label='GEN %i' % g)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.save(plotDir + '/final_10_obj_space.png')

print(pf,ps)
