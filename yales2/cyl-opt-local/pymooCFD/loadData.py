import numpy as np

from pymooCFD.setupOpt import *

X = []
F = []
for ind in range(30):
    X.append(np.loadtxt(f'gen0/ind{ind}/var.txt'))
    F.append(np.loadtxt(f'gen0/ind{ind}/obj.txt'))

print(X)
print(F)
np.savetxt('obj.txt', F)
np.savetxt('var.txt', X)


# here F and G is re-evaluated - in practice you want to load them from files too
# F, G = problem.evaluate(X, return_values_of=["F", "G"])
#
#
# #################################
# from pymoo.model.evaluator import Evaluator
# from pymoo.model.population import Population
# from pymoo.model.problem import StaticProblem
#
# # now the population object with all its attributes is created (CV, feasible, ...)
# pop = Population.new("X", X)
# pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)
#
#
# ################################
# from pymoo.algorithms.so_genetic_algorithm import GA
# from pymoo.optimize import minimize
#
# # the algorithm is now called with the population - biased initialization
# algorithm = GA(pop_size=100, sampling=pop)
#
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 10),
#                seed=1,
#                verbose=True)
