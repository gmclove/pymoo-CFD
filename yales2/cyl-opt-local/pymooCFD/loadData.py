import numpy as np

from pymooCFD.setupOpt import *


def loadData():
    np.loadtxt('obj.txt', F)
    np.loadtxt('var.txt', X)

    try:
        checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
        print("Loaded Checkpoint:", checkpoint)
        # only necessary if for the checkpoint the termination criterion has been met
        checkpoint.has_terminated = False
        algorithm = checkpoint
        print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        return algorithm
    except OSError as err:
        print(err)


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
