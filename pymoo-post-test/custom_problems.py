from pymoo.model.problem import Problem
from pymoo.util.misc import stack

# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4
from RunYALES2 import RunYALES2

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                                  # omega freq
                         xl=np.array([0.1, 0.1]),
                         xu=np.array([3, 1])
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        ###### Initialize Generation ######
        gen = algorithm.n_gen
        print('GEN:%i' % gen)
        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # Maximum processors to be used
        # procLim = 1
        # Number of processors for each individual (EQUAL or SMALLER than procLim)
        # nProc = 1

        # create sim object for this generation and it's population
        sim = RunYALES2(x, gen)

        out['F'] = np.column_stack(sim.obj)

        np.save("checkpoint-gen%i" % gen, algorithm)
        np.save("checkpoint", algorithm)

        print('GEN%i COMPLETE' % gen)

        # objectives unconstrainted
        # g1 = 2*(x[:, 0]-0.1) * (x[:, 0]-0.9) / 0.18
        # g2 = - 20*(x[:, 0]-0.4) * (x[:, 0]-0.6) / 4.8
        # out["G"] = np.column_stack([g1, g2])
    #
    # # --------------------------------------------------
    # # Pareto-front - not necessary but used for plotting
    # # --------------------------------------------------
    # def _calc_pareto_front(self, flatten=True, **kwargs):
    #     f1_a = np.linspace(0.1 ** 2, 0.4 ** 2, 100)
    #     f2_a = (np.sqrt(f1_a) - 1) ** 2
    #
    #     f1_b = np.linspace(0.6 ** 2, 0.9 ** 2, 100)
    #     f2_b = (np.sqrt(f1_b) - 1) ** 2
    #
    #     a, b = np.column_stack([f1_a, f2_a]), np.column_stack([f1_b, f2_b])
    #     return stack(a, b, flatten=flatten)
    #
    # # --------------------------------------------------
    # # Pareto-set - not necessary but used for plotting
    # # --------------------------------------------------
    # def _calc_pareto_set(self, flatten=True, **kwargs):
    #     x1_a = np.linspace(0.1, 0.4, 50)
    #     x1_b = np.linspace(0.6, 0.9, 50)
    #     x2 = np.zeros(50)
    #
    #     a, b = np.column_stack([x1_a, x2]), np.column_stack([x1_b, x2])
    #     return stack(a, b, flatten=flatten)


# problem = MyProblem()

# TEST PROBLEM
from pymoo.factory import get_problem
problem = get_problem("bnh")