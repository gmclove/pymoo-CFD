from pymooIN import *

########################################################################################################################
######    OPTIMIZATION    ######
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               ('n_gen', n_gen),
               callback=callback,
               seed=1,
               copy_algorithm=False,
               # pf=problem.pareto_front(use_cache=False),
               save_history=True,
               display=display,
               verbose=True
               )

np.save("checkpoint", algorithm)
print("EXEC TIME: %.3f seconds" % res.exec_time)
