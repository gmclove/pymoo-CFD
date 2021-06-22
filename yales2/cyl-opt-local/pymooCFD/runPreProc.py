from setup import *

def runPreProc():
    ########################################################################################################################
    ######    RUN GENERATION 0    #######
    from pymoo.optimize import minimize

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 1),
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

    ########################################################################################################################
    ######     ######
    # from optPreProc import gen0Map

if __name__ == "__main__":
    runPreProc()
