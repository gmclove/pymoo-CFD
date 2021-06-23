import numpy as np

from pymooCFD.setupOpt import *


def loadCP(checkpointFile='checkpoint.npy', hasTerminated=False):
    checkpoint, = np.load(checkpointFile, allow_pickle=True).flatten()
    print("Loaded Checkpoint:", checkpoint)
    # only necessary if for the checkpoint the termination criterion has been met
    checkpoint.has_terminated = hasTerminated
    algorithm = checkpoint
    print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
    return algorithm

def loadTxt():
    try:
        print('Loading from text files')
        X = np.loadtxt('var.txt')
        F = np.loadtxt('obj.txt')
    except OSError as err:
        print(err)
        print('Failed to load text files')
        print('Data loading failed returning "None, None"...')
        return None, None


    # def loadData(checkpointFile='checkpoint.npy', hasTerminated=False):
    #     try:
    #         checkpoint, = np.load(checkpointFile, allow_pickle=True).flatten()
    #         print("Loaded Checkpoint:", checkpoint)
    #         # only necessary if for the checkpoint the termination criterion has been met
    #         checkpoint.has_terminated = hasTerminated
    #         algorithm = checkpoint
    #         print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
    #         X = algorithm.callback.data['var']
    #         F = algorithm.callback.data['obj']
    #         return X, F
    #         # return algorithm
    #     except OSError as err:
    #         print(err)
    #         print(f'{checkpointFile} load failed.')
    #         try:
    #             print('Loading from text files...')
    #             X = []
    #             F = []
    #             # load in 2D text file arrays to construct 3D arrays
    #             for file in os.listdir(dataDir):
    #                 if file.startswith("gen"):
    #                     if ent.endswitth('X.txt'):
    #                         X.append(np.loadtxt(file))
    #                     elif file.endswith('F.txt'):
    #                         F.append(np.loadtxt(file))
    #             return X, F
                # # re-establish algorithm object
                # from pymoo.model.population import Population
                # from pymoo.model.evaluator import Evaluator
                # from pymoo.model.problem import StaticProblem
                # # first the population object with all its attributes is created (CV, feasible, ...)
                # pop = Population.new('X', X)
                # pop = Evaluator().eval(StaticProblem(problem, F=F), pop)
                # # the algorithm is now called with the population - biased initialization
                # algorithm = NSGA(pop_size=n_ind, sampling=pop)
                # # from pymooCFD.setupOpt import algorithm
                # # algorithm = algorithm
                # return algorithm



# def loadData(checkpointFile='checkpoint.npy', hasTerminated=False):
#     try:
#         checkpoint, = np.load(checkpointFile, allow_pickle=True).flatten()
#         print("Loaded Checkpoint:", checkpoint)
#         # only necessary if for the checkpoint the termination criterion has been met
#         checkpoint.has_terminated = hasTerminated
#         algorithm = checkpoint
#         print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
#         return algorithm
#     except OSError as err:
#         print(err)
#         print(f'{checkpointFile} load failed.')
#         print('Data loading failed returning "None"...')
#         return None
#         try:
#             print('Loading from text files')
#             X = np.loadtxt('var.txt')
#             F = np.loadtxt('obj.txt')
#         except OSError as err:
#             print(err)
#             print('Failed to load text files')
#             print('Data loading failed returning "None, None"...')
#             return None, None

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
