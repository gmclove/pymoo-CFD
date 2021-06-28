# import numpy as np

def runOpt(restart=True):
    if restart == True:
        try:
            from pymooCFD.util.handleData import loadCP
            algorithm = loadCP()
            # from pymooCFD.util.handleData import loadTxt
            # algorithm = loadTxt('dump/gen13X.txt', 'dump/gen13F.txt')
        except OSError as err:
            print(err)
            from pymooCFD.setupOpt import checkpointFile
            print(f'{checkpointFile} load failed.')
            print('Data loading failed returning "None"...')
            print('RESTART FAILED')
            return

    else:
        # load algorithm initialize in setupOpt.py module
        from pymooCFD.setupOpt import algorithm
        # try:
        #     checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
        #     print("Loaded Checkpoint:", checkpoint)
        #     # only necessary if for the checkpoint the termination criterion has been met
        #     checkpoint.has_terminated = False
        #     algorithm = checkpoint
        #     print('Last checkpoint at generation %i' % len(algorithm.callback.data['var']))
        # except OSError as err:
        #     print(err)
        #     from pymooCFD.setupOpt import algorithm
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
    from pymooCFD.setupOpt import problem, callback, display, n_gen #, termination

    res = minimize(problem=problem,
                   algorithm=algorithm,
                   # termination=termination,
                   termination=('n_gen', n_gen),
                   callback=callback,
                   seed=1,
                   copy_algorithm=True,
                   # pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   display=display,
                   verbose=True
                   )

    # np.save("checkpoint", algorithm)
    print("EXEC TIME: %.3f seconds" % res.exec_time)

# if __name__ == "__main__":
#     runOpt()
