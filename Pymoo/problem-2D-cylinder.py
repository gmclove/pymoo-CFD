import numpy as np
import subprocess
from pymoo.model.problem import Problem
from pymoo.util.misc import stack

from Pymoo.callback import callback
from YALES2 import PreProc, PostProc, ExecuteSims
# from RunOpenFOAMv4.RunOpenFOAMv4 import RunOFv4

from GMSHapi import GMSHapi


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                                    # omega freq
                         xl=np.array([0.1, 0.1]),
                         xu=np.array([3, 1])
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        ###### Initialize Generation ######
        case = '2D_cylinder'
        gen = callback.gen
        # geometry variables index
        # geoVarsI = [0, 1]
        # GMSHapi(self.x, self.gen, geoVarsI)

        # Maximum processors to be used
        procLim = 1
        # Number of processors for each individual (EQUAL or SMALLER than procLim)
        nProc = 1

        # print(x)

        # create sim object for this generation and it's population
        # sim = RunOFv4(case, x, gen, solver, mesher, procLim, nProc)
        # obj = sim.runGen()

        PreProc()
        # ExecuteSims()
        sim = subprocess.run(["sh", "sim.sh"])
        PostProc()


        #         print('Objectives')
        #         print('  Lift    Drag')
        #         for i in range(len(obj[:][0])):
        #             print('gen %i:'%i, end=' ')
        #             for j in range(len(obj[0][:])):
        #                 print('%.6f' %obj[i][j], end=' ')
        #             print('\n')

        # out["F"] = np.column_stack([sigmaX, sigmaY])
        out['F'] = obj

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
