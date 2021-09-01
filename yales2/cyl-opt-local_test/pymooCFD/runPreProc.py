import os
import numpy as np
from pymooCFD.setupOpt import *
from pymooCFD.util.handleData import loadCP

def runGen0():
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

def mapGen0():
    preProcDir = f'{plotsDir}/preProc'
    try:
        os.mkdir(preProcDir)
    except OSError as err:
        print(err)
        print(f'{preProcDir} already exists')
    ########################################################################################################################
    algorithm = loadCP()
    X = algorithm.pop.get('X')
    F = algorithm.pop.get('F')
    ########################################################################################################################
    print('VARS')
    print(X)
    print('OBJ')
    print(F)
    ########################################################################################################################
    ##### SCATTER PLOTS #######
    ###########################
    X = np.array(X)
    F = np.array(F)
    from pymoo.visualization.scatter import Scatter
    # https://pymoo.org/visualization/scatter.html
    ##### Function Space ######
    f_space = Scatter(title = 'Objective Space',
                        labels = obj_labels)
    f_space.add(F)
    # if pf is not None:
    #     f_space.add(pf)
    f_space.save(f'{preProcDir}/obj-space.png')
    ##### Variable Space ######
    f_space = Scatter(title = 'Design Space',
                        labels = obj_labels)
    f_space.add(X)
    # if pf is not None:
    #     f_space.add(pf)
    f_space.save(f'{preProcDir}/var-space.png')

    ##### Variable vs. Objective Plots ######
    # extract objectives and variables columns and plot them against each other
    for x_i, x in enumerate(X.transpose()):
        for f_i, f in enumerate(F.transpose()):
            plot = Scatter(title=f'{var_labels[x_i]} vs. {obj_labels[f_i]}',
                            labels=[var_labels[x_i], obj_labels[x_i]]
                            )
            xy = np.column_stack((x,f))
            plot.add(xy)
            plot.save(f'{preProcDir}/{var_labels[x_i].replace(" ", "_")}-vs-{obj_labels[f_i].replace(" ", "_")}.png')

    # if there are more than 2 objectives create array of scatter plots comparing
    # the trade-off between 2 objectives at a time
    if len(F.transpose()) > 2:
        ####### Pair Wise Objective Plots #######
        # Pairwise Scatter Plots of Function Space
        plot = Scatter(tight_layout=True)
        plot.add(F, s=10)
        plot.add(F[-1], s=30, color="red")
        plot.save(f'{preProcDir}/pairwise-scatter.png')

if __name__ == "__main__":
    runGen0()
    mapGen0()
