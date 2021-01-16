from pymoo.optimize import minimize

#from Pymoo import algorithm, callback, display

from Pymoo.problem import problem
from Pymoo.algorithm import algorithm
from Pymoo.callback import callback
from Pymoo.display import display

res = minimize(problem,
               algorithm,
               ("n_gen", 5),
               callback=callback,
               seed=1,
               # pf=problem.pareto_front(use_cache=False),
               save_history=True,
               display=display,
               verbose=True
               )

print('Variables:')
print(callback.data['var'])
print('Objectives:')
print(callback.data['obj'])
print('Best Objective 1:')
for i in range(len(callback.data['best_obj1'])):
    print('%.6f' % callback.data['best_obj1'][i])
print('Best Objective 2:')
for i in range(len(callback.data['best_obj2'])):
    print('%.6f' % callback.data['best_obj2'][i])


# print("Time Elapsed:")
# print(res.time)
print("Objectives:")
print(res.pop.get('F'))
# print(res.F)
print("Variables:")
print(res.pop.get('X'))
# print(res.X)
