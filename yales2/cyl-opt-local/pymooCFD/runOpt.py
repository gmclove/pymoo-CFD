import numpy as np

def runOpt(restart=True):
    if restart == True:
        try:
            checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
            print("Loaded Checkpoint:", checkpoint)
            # only necessary if for the checkpoint the termination criterion has been met
            checkpoint.has_terminated = False
            algorithm = checkpoint
            print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        except OSError as err:
            print(err)
            from pymooCFD.setupOpt import algorithm
            # try:
            #     X = np.loadtxt('var.txt')
            #     F = np.loadtxt('obj.txt')
            #     from pymoo.model.evaluator import Evaluator
            #     from pymoo.model.population import Population
            #     from pymoo.model.problem import StaticProblem
            #
            #     # now the population object with all its attributes is created (CV, feasible, ...)
            #     pop = Population.new("X", X)
            #     pop = Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)

    ########################################################################################################################
    ######    OPTIMIZATION    ######
    from pymoo.optimize import minimize
    from pymooCFD.setupOpt import problem, callback, termination, display

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   # ('n_gen', n_gen),
                   callback=callback,
                   seed=1,
                   copy_algorithm=False,
                   # pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   display=display,
                   verbose=True
                   )

    # np.save("checkpoint", algorithm)
    print("EXEC TIME: %.3f seconds" % res.exec_time)

# if __name__ == "__main__":
#     runOpt()
