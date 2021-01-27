from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(
    pop_size=7,
    #n_offsprings=2,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
    )

########################################################################################################################
from pymoo.model.callback import Callback


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.gen = 0
        self.data["best_obj1"] = []
        self.data['best_obj2'] = []
        self.data['var'] = []
        self.data['obj'] = []

    def notify(self, algorithm):
        self.data["best_obj1"].append(algorithm.pop.get("F")[:, 0].min())
        self.data['best_obj2'].append(algorithm.pop.get('F')[:, 1].min())
        self.data['var'].append(algorithm.pop.get('X'))
        self.data['obj'].append(algorithm.pop.get('F'))
        self.gen += 1


callback = MyCallback()
########################################################################################################################
from pymoo.util.display import Display


class MyDisplay(Display):
    bestObj = []

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        # self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
        # self.output.append("metric_b", np.mean(algorithm.pop.get("F")))
        self.output.append('Best Lift [N]', np.mean(algorithm.pop.get("F")[:, 0].min()))
        self.output.append('Best Drag [N]', np.mean(algorithm.pop.get("F")[:, 1].min()))
        # if


display = MyDisplay()
########################################################################################################################
import numpy as np
from pymoo.model.problem import Problem
from pymoo.util.misc import stack

# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4
from RunYALES2 import RunYALES2


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                                    # mu_x  mu_y
                         xl=np.array([-0.3, 0]),
                         xu=np.array([-0.1, 0.15])
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        ###### Initialize Generation ######
        gen = callback.gen

        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # Maximum processors to be used
        # procLim = 1
        # Number of processors for each individual (EQUAL or SMALLER than procLim)
        # nProc = 1

        # print(x)

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen)

        out['F'] = sim.obj

        # objectives unconstrainted
        # g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        # g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8
        # out["G"] = np.column_stack([g1, g2])

    # --------------------------------------------------
    # Pareto-front - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_front(self, flatten=True, **kwargs):
        f1_a = np.linspace(0.1 ** 2, 0.4 ** 2, 100)
        f2_a = (np.sqrt(f1_a) - 1) ** 2

        f1_b = np.linspace(0.6 ** 2, 0.9 ** 2, 100)
        f2_b = (np.sqrt(f1_b) - 1) ** 2

        a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
        return stack(a, b, flatten=flatten)

    # --------------------------------------------------
    # Pareto-set - not necessary but used for plotting
    # --------------------------------------------------
    def _calc_pareto_set(self, flatten=True, **kwargs):
        x1_a = np.linspace(0.1, 0.4, 50)
        x1_b = np.linspace(0.6, 0.9, 50)
        x2 = np.zeros(50)

        a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
        return stack(a, b, flatten=flatten)


problem = MyProblem()
########################################################################################################################
from pymoo.optimize import minimize

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


########################################################################################################################
#
# # the global factory method
# from pymoo.factory import get_visualization
# plot = get_visualization("scatter")
#
# for gen in callback.data['obj']:
#     #print(gen)
#     plot.add(gen)
#
# #plot.add(res.pop.get('F'), color="green", marker="x")
# #print(res.pop.get('F'))
# #plot.add(B, color="red", marker="*")
# plot.show()
#
# from pymoo.visualization.scatter import Scatter
#
# # get the pareto-set and pareto-front for plotting
# ps = problem.pareto_set(use_cache=False, flatten=False)
# pf = problem.pareto_front(use_cache=False, flatten=False)
#
# # Design Space
# plot = Scatter(title = "Design Space", axis_labels="x")
# plot.add(res.X, s=30, facecolors='none', edgecolors='r')
# plot.add(ps, plot_type="line", color="black", alpha=0.7)
# plot.do()
# plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
# plot.apply(lambda ax: ax.set_ylim(-2, 2))
# plot.show()
#
# # Objective Space
# plot = Scatter(title = "Objective Space")
# plot.add(res.F)
# plot.add(pf, plot_type="line", color="black", alpha=0.7)
# plot.show()
#
# import matplotlib.pyplot as plt
# matplotlib.use('GTK3Agg')
#
# val = [e.pop.get("F").min() for e in res.history]
# plt.plot(np.arange(len(val)), val)
# plt.show()
#
#

########################################################################################################################
