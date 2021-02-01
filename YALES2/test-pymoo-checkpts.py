import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt1", n_var=5)

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 5),
               seed=1,
               copy_algorithm=False,
               verbose=True)

np.save("checkpoint", algorithm)

checkpoint, = np.load("checkpoint.npy", allow_pickle=True).flatten()
print("Loaded Checkpoint:", checkpoint)

# only necessary if for the checkpoint the termination criterion has been met
checkpoint.has_terminated = False

res = minimize(problem,
               checkpoint,
               ('n_gen', 20),
               seed=1,
               copy_algorithm=False,
               verbose=True)

