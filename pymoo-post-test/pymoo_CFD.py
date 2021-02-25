import os
import numpy as np
########################################################################################################################
######    DISPLAY    ######
from displays import MyDisplay

########################################################################################################################
######    CALLBACK    ######
from callbacks import MyCallback

########################################################################################################################
######    PROBLEM    ######
# from problems import MyProblem
# problem = MyProblem()

# TEST PROBLEM
from pymoo.factory import get_problem
problem = get_problem("bnh")

########################################################################################################################
######    ALGORITHM    ######
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
if os.path.exists('checkpoint.npy'):
    checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
    print("Loaded Checkpoint:", checkpoint)
    # only necessary if for the checkpoint the termination criterion has been met
    checkpoint.has_terminated = False
    algorithm = checkpoint
    print('Last checkpoint at generation %i' % algorithm.n_gen)
else:
    algorithm = NSGA2(
        pop_size=10,
        # n_offsprings=1,
        # sampling=get_sampling("real_random"),
        # crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        # mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )
########################################################################################################################
######    OPTIMIZATION    ######
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               ("n_gen", 50),
               callback=MyCallback(),
               seed=1,
               copy_algorithm=False,
               # pf=problem.pareto_front(use_cache=False),
               save_history=True,
               display=MyDisplay(),
               verbose=True
               )

np.save("checkpoint", algorithm)
print("EXEC TIME: " + str(res.exec_time))
